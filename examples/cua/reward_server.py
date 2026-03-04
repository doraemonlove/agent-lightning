import os
import json
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import uvicorn
from cua_reward_model import CUARewardModel

# 兼容 OpenAI 客户端
try:
    from openai import OpenAI, AsyncOpenAI

    OPENAI_AVAILABLE = True
except Exception:
    OpenAI = None
    AsyncOpenAI = None
    OPENAI_AVAILABLE = False


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


load_dotenv()
api_key = os.getenv("score_api_key")

if not OPENAI_AVAILABLE:
    raise RuntimeError("openai 包不可用，请先安装 openai")
if not api_key:
    raise RuntimeError("环境变量 score_api_key 未设置")

reward_model = CUARewardModel(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=api_key,
    model="doubao-seed-1-6-251015",
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
