import os
import json

# 尝试导入OpenAI库
import base64
import mimetypes
from typing import Any, Dict, List, Union, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import traceback
import uvicorn
from constants import CUA_EVALUATION_PROMPT, CUA_REVIEW_PROMPT
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
    temperature: Optional[float] = 0.0
    contents: Optional[List[Dict[str, Any]]] = None

class ScoreResponse(BaseModel):
    score: float
    reason: str

class ReviewRequest(BaseModel):
    trace: Optional[List[Dict[str, Any]]]
    temperature: Optional[float] = 0.0

class GroupScoreResponse(BaseModel):
    grouped_traces: List

def _to_data_url_from_path(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "image/png"
    with open(path, "rb") as fp:
        b64 = base64.b64encode(fp.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"

def _normalize_image_url(val: Any) -> List[Dict[str, Any]]:
    """
    将 screenshot/image 字段统一转为 image_url content 列表。
    支持：
      - 本地文件路径 -> 转 data:image/*;base64,
      - 已是 data:image/*;base64, -> 原样使用
      - 远程 http(s) 链接 -> 原样使用
      - 原始 base64 字符串 -> 加 data:image/png;base64, 前缀
    """
    urls = []
    candidates: List[Union[str, Dict[str, Any]]] = []
    if val is None:
        return urls
    if isinstance(val, list):
        candidates = val
    else:
        candidates = [val]
    for item in candidates:
        url = None
        if isinstance(item, dict):
            # 常见字段名
            for k in ("url", "image_url", "path", "file", "local_path", "data_url", "base64", "b64"):
                if k in item and item[k]:
                    item = item[k]
                    break
        if isinstance(item, str):
            s = item.strip()
            if s.startswith("data:image/"):
                url = s
            elif s.startswith("http://") or s.startswith("https://"):
                url = s
            elif os.path.exists(s):
                url = _to_data_url_from_path(s)
            else:
                # 兜底当作裸的base64
                url = f"data:image/png;base64,{s}"
        if url:
            urls.append({"type": "image_url", "image_url": {"url": url}})
    return urls

def _iter_events(trace: Any) -> List[Dict[str, Any]]:
    """
    统一获取事件列表：
      - 若为 list 直接返回
      - 若为 dict，尝试 trace/events/trajectory/input.messages
    """
    if isinstance(trace, list):
        return trace
    if isinstance(trace, dict):
        for key in ("trace", "events", "trajectory"):
            v = trace.get(key)
            if isinstance(v, list):
                return v
        # 回退到 input.messages 但仅保留可能的动作/截图结构
        v = trace.get("input", {}).get("messages")
        if isinstance(v, list):
            return v
    return []


class CuaTraceScorer:
    """
    将轨迹(list/dict 或 JSON 字符串)整理为图文混排消息并调用模型打分，只返回 float 分数。

    - 初始化参数：
      base_url: LLM 服务地址
      api_key:  API key（也可从环境变量 ARK_API_KEY 读取后传入）
      model:    模型名称，默认 "dx5"

    - score(trace, user_instruction, tools):
      trace: list/dict 或 JSON 字符串（不从文件读取）
      user_instruction: 会放在消息开头的 text
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "dx5",
        prompt: str = CUA_EVALUATION_PROMPT,
        review_prompt: str = CUA_REVIEW_PROMPT
    ) -> None:
        if not OPENAI_AVAILABLE or OpenAI is None or AsyncOpenAI is None:
            raise RuntimeError("OpenAI 客户端不可用，请先 pip install openai")
        if not base_url:
            raise ValueError("base_url 不能为空")
        if not api_key:
            raise ValueError("api_key 不能为空")
        # self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.prompt = prompt
        self.review_prompt = review_prompt

    def _parse_trace_input(self, trace: Union[str, List[Dict[str, Any]], Dict[str, Any]]) -> Any:
        """
        仅支持：
        - str 为 JSON 字符串 -> 解析
        - 直接为 list/dict -> 原样返回
        不从文件读取。
        """
        if isinstance(trace, str):
            s = trace.strip()
            try:
                return json.loads(s)
            except Exception as e:
                raise TypeError("trace 为 str，但不是合法 JSON 字符串") from e
        if isinstance(trace, (list, dict)):
            return trace
        raise TypeError("trace 需要是 list/dict 或 JSON 字符串")

    def _build_content(self, parsed_trace: Any, user_instruction: str) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": f"User Instruction: {user_instruction.strip()}"},
        ]

        events = _iter_events(parsed_trace)
        step_count = 1
        
        # 标记是否已经处理过初始状态
        has_processed_init = False

        for ev in events:
            if not isinstance(ev, dict):
                continue

            # ===============================================================
            # 1. 初始状态 (Initial State)
            # 逻辑：只要是第一张图，且没有 Action/Output，就是初始状态
            # ===============================================================
            if "screenshot" in ev and not has_processed_init:
                # 只有当它是纯截图，或者作为 trace 的起手式时
                if "tool_calls" not in ev and "tool_outputs" not in ev:
                    content.append({"type": "text", "text": "### Step 0: Initial State (Before Start)"})
                    # 假设 _normalize_image_url 返回的是 [{"type": "image_url", ...}]
                    content.extend(_normalize_image_url(ev["screenshot"]))
                    has_processed_init = True
                    continue

            # ===============================================================
            # 2. 模型动作 (Agent Action)
            # 逻辑：这是因果链的“因”
            # ===============================================================
            if "tool_calls" in ev:
                summary = ev.get("summary", "Thinking...")
                tool_calls_json = json.dumps(ev["tool_calls"], ensure_ascii=False, indent=2)
                
                content.append({
                    "type": "text", 
                    "text": (
                        f"\n---\n"  # 分隔线，帮助 LLM 区分回合
                        f"### Step {step_count}: Agent Action\n"
                        f"**Thought:** {summary}\n"
                        f"**Function Call:**\n"
                        f"```json\n{tool_calls_json}\n```"
                    )
                })
                # 注意：这里增加计数，意味着接下来的 output 属于这个 step
                step_count += 1
                continue

            # ===============================================================
            # 3. 工具执行结果 (Execution Result)
            # 逻辑：这是因果链的“果”。包含了 文本返回 + 新的截图
            # ===============================================================
            if "tool_outputs" in ev:
                # 对应的 Action Step 是 step_count - 1
                current_step_idx = step_count - 1
                
                # 构建文本部分
                tool_outputs_json = json.dumps(ev["tool_outputs"], ensure_ascii=False, indent=2)
                
                # 3.1 先放入文本结果 (Output)
                result_text = (
                    f"### Result of Step {current_step_idx}\n"
                    f"**Tool Outputs:**\n"
                    f"```json\n{tool_outputs_json}\n```"
                )
                content.append({"type": "text", "text": result_text})

                # 3.2 再放入视觉结果 (Screenshot)
                # 逻辑：这是 Action 执行“之后”的屏幕状态
                if "screenshot" in ev and ev["screenshot"]:
                    content.append({
                        "type": "text", 
                        "text": f"**Screen State After Step {current_step_idx}:**"
                    })
                    content.extend(_normalize_image_url(ev["screenshot"]))
                
                continue

            # ===============================================================
            # 4. 兜底：处理中间可能出现的独立截图 (Mid-stream Screenshot)
            # 有些 trace 可能会单独记录截图而不带 tool_output
            # ===============================================================
            if "screenshot" in ev and has_processed_init:
                # 如果这个截图已经在 tool_outputs 里处理过了，就不会走到这里
                # 这里处理的是“只有截图”的事件
                content.append({
                    "type": "text", 
                    "text": f"**Screen State Update (Observation):**"
                })
                content.extend(_normalize_image_url(ev["screenshot"]))

        return content

    async def score_trace(
        self,
        trace: Union[str, List[Dict[str, Any]], Dict[str, Any]],
        user_instruction: str,
        temperature: float = 0.0,
    ) -> float:
        """
        接受 trace(list/dict 或 JSON 字符串)、用户指令、工具列表，返回 0~1 的 float 评分。
        注意：不从文件读取。
        """
        try:
            parsed = self._parse_trace_input(trace)
            contents = self._build_content(parsed, user_instruction)

            # 仅发送一个包含评估 Prompt 与用户指令、轨迹的 user 消消息
            resp =  await self.client.beta.chat.completions.parse(
                model=self.model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": self.prompt.strip()},
                    {"role": "user", "content": contents},
                ],
                response_format=ScoreResponse
            )
            
            result = resp.choices[0].message.parsed.model_dump()
            print(f"result: {result}")

            return result
        except Exception as e:
            print(f"❌ 轨迹评分失败: {e}")
            
            traceback.print_exc()
            return {
                "score": 0.0,
                "reason": ""
            }

    async def review_trace(
        self,
        trace: List[Dict[str, Any]],
        temperature: float = 0.0,
    ) -> float:
        """
        接受 trace(list/dict 或 JSON 字符串)、用户指令、工具列表，返回 0~1 的 float 评分。
        注意：不从文件读取。
        """
        contents = [
            {"role": "system", "content": [{"type": "text", "text": self.review_prompt.strip()}]}
        ] + trace
        
        print(f"contents: {contents}")
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                messages=contents
            )
            
            result = resp.choices[0].message.content
            print(f"result: {result}")

            return result
        except Exception as e:
            print(f"❌ 轨迹评分失败: {e}")

            traceback.print_exc()
            return ""
        
app = FastAPI(
    title="CUA 轨迹评分 API",
    version="1.0",
    description="提供 CUA 轨迹评分服务",
)

import os
from dotenv import load_dotenv
from grouped_actions_reward import GroupedActionsRewardModel
load_dotenv()
api_key = os.getenv("score_api_key")
scorer = CuaTraceScorer(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=api_key,
    model="doubao-seed-1-6-251015",
)
group_scorer = GroupedActionsRewardModel(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=api_key,
    model="doubao-seed-1-6-251015"
)
@app.post("/score", response_model=ScoreResponse)
async def score_trace(
    request: ScoreRequest
):
    try:
        result = await scorer.score_trace(
            trace=request.trace,
            user_instruction=request.user_instruction,
            temperature=request.temperature,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/group_score", response_model=GroupScoreResponse)
async def group_score_trace(
    request: ScoreRequest
):
    try:
        result = await group_scorer.grouped_actions_reward(
            trace=request.trace,
            user_instruction=request.user_instruction,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/review")
async def review_trace(
    request: ReviewRequest
):
    try:
        result = await scorer.review_trace(
            trace=request.trace,
            temperature=request.temperature,
        )
        return result
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
        timeout_keep_alive=600, # 保持连接时间
        timeout_graceful_shutdown=60 # 优雅关闭时间
    )