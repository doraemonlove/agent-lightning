import os
import json
from dotenv import load_dotenv, find_dotenv
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import uvicorn
import requests
import agentlightning
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import asyncio
from cua_reward_model import CUARewardModel
from constants import CUA_CHROME_EVALUATION_PROMPT

agentlightning.configure_logger()

logger = agentlightning.configure_logger(name=__name__)

load_dotenv(find_dotenv())

api_key = os.getenv("score_api_key")


try:
    from openai import OpenAI, AsyncOpenAI

    OPENAI_AVAILABLE = True
except Exception:
    OpenAI = None
    AsyncOpenAI = None
    OPENAI_AVAILABLE = False

if not OPENAI_AVAILABLE:
    raise RuntimeError("openai 包不可用，请先安装 openai")
if not api_key:
    raise RuntimeError("环境变量 score_api_key 未设置")

reward_model = CUARewardModel(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=api_key,
    model="doubao-seed-1-6-251015",
    overall_reward_prompt=CUA_CHROME_EVALUATION_PROMPT,
)


class ScoreRequest(BaseModel):
    trace: Union[str, List[Dict[str, Any]], Dict[str, Any]]
    user_instruction: str


class OverallScoreResponse(BaseModel):
    score: float
    reason: str


class ContextTraceItem(BaseModel):
    instruction: str
    trace_data: List[Dict[str, Any]]
    reward: Optional[float] = None


class GroupScoreResponse(BaseModel):
    context_traces: List[ContextTraceItem]


def _normalize_trace(trace: Union[str, List[Dict[str, Any]], Dict[str, Any]]) -> List[Dict[str, Any]]:
    if isinstance(trace, list):
        return trace

    if isinstance(trace, str):
        try:
            parsed = json.loads(trace)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail=f"trace 不是有效 JSON 字符串: {exc}")

        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            if isinstance(parsed.get("trace"), list):
                return parsed["trace"]
            if isinstance(parsed.get("events"), list):
                return parsed["events"]
        raise HTTPException(status_code=422, detail="trace JSON 必须是列表，或包含 trace/events 列表字段")

    if isinstance(trace, dict):
        if isinstance(trace.get("trace"), list):
            return trace["trace"]
        if isinstance(trace.get("events"), list):
            return trace["events"]
        raise HTTPException(status_code=422, detail="trace 对象必须包含 trace 或 events 列表字段")

    raise HTTPException(status_code=422, detail="trace 类型不支持")


app = FastAPI(
    title="CUA 轨迹评分 API",
    version="1.0",
    description="提供 CUA 轨迹评分服务",
)


@app.post("/overall_score", response_model=OverallScoreResponse)
async def score_trace(request: ScoreRequest):
    try:
        normalized_trace = _normalize_trace(request.trace)
        result = await reward_model.get_overall_reward(
            trace=normalized_trace,
            user_instruction=request.user_instruction,
        )
        return OverallScoreResponse(score=result[0], reason=result[1])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/group_score", response_model=GroupScoreResponse)
async def group_score_trace(request: ScoreRequest):
    try:
        normalized_trace = _normalize_trace(request.trace)
        result = await reward_model.get_group_rewards(
            trace=normalized_trace,
            user_instruction=request.user_instruction,
        )
        return GroupScoreResponse(context_traces=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        timeout_keep_alive=600,  # 保持连接时间
        timeout_graceful_shutdown=60,  # 优雅关闭时间
    )


# 评分接口
async def score_trace(url, trace: list[dict] = None, user_instruction: str = None):
    try:
        # 1. 检查数据量，防止发送过大炸弹
        # 如果 trace 中包含 image，建议在此处做截断或只发 url
        payload = {
            "trace": trace,
            "user_instruction": user_instruction,
        }

        # 打印大小日志
        payload_str = json.dumps(payload)
        payload_mb = len(payload_str) / (1024 * 1024)
        logger.info(f"正在发送评分请求，数据大小: {payload_mb:.2f} MB")

        if payload_mb > 50:  # 假设阈值是 50MB
            logger.warning("数据包过大，可能会导致连接中断！建议检查 trace 是否包含过多 Base64 图片。")

        # 2. 配置重试策略
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        session.mount("http://", HTTPAdapter(max_retries=retries))
        session.mount("https://", HTTPAdapter(max_retries=retries))

        # 3. 发送请求 (使用 json=payload 自动处理 header)
        # 这里的 timeout=(连接超时, 读取超时)
        response = session.post(url, json=payload, timeout=(10, 1200))

        response.raise_for_status()
        result = response.json()
        return result

    except requests.exceptions.ConnectionError as e:
        logger.error(f"连接被重置/断开。原因可能是数据包过大或服务端崩溃。Error: {e}")
        return {}
    except Exception as e:
        logger.error(f"评分轨迹失败: {e}")
        return {}
