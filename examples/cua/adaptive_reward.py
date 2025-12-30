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

# 轨迹总体评估prompt
CUA_EVALUATION_PROMPT = """
# Role (角色设定)
你是一名**铁面无私的自动化审计员**。你的任务是基于客观事实审核 Agent 的资产盘点任务执行情况。

# ⚠️ CORE PRINCIPLE: DATA TRUST HIERARCHY (核心原则：数据信任分级)
1. **最高信任级 (Trusted Evidence)**：
   * **Screenshot (截图)**：这是唯一的“视觉真值”。
   * **Action Code (代码)**：这是 Agent 实际发送给系统的指令。
2. **校验级参考 (Intent Validation)**：
   * **Summary / Thought**：仅视为意图声明，禁止直接采信。
   * **准则**：必须以【Action是否执行】及【设备说明是否刷新】作为判定依据。

# Evaluation Steps (分项评估标准 - 总分 1.0)

## 维度 1：环境启动与筛选 (Start & Setup) - 满分 0.2
* **SOP 事实核查**：冷启动、页面跳转、筛选条件（计划、审核结果）是否正确。
* **计分**：完全正确得 0.2，否则 0 分。

## 维度 2：审核逻辑准确性与数量 (Core Logic & Quantity) - 满分 0.5
**此项遵循严格的“有效性优先”审计链条：**
1. **有效性判定 (Existence Check)**：对比点击前后列表第一行【设备说明】。若文字**完全一致**，视为无效操作（页面未刷新），该条记录得 0 分。若发生变化，视为有效，进入下一步。
2. **准确性判定 (Logic Check)**：
    * **通过准则**：截图包含与系统条码**完全一致**的子字符串 -> 必须点“通过”。
    * **驳回准则**：截图中无法识别到匹配条码 -> 必须点“驳回”。
* **计分公式**：
    $$ RawScore = 0.5 \times \frac{\text{审核有效且逻辑正确的条数}}{\text{总要求审核数 N}} $$
    * **限制**：此维度最高得分不超过 0.5 分（即使正确条数超过 N，也只记 0.5）。

## 维度 3：任务终止控制 (Flow Control) - 满分 0.2
* **SOP 事实核查**：
    * 恰好完成 N 条后停止：得 0.2 分。
    * 未完成 N 条 或 大幅超额审核：得 0 分。

## 维度 4：工具操作精准度 (Tool Proficiency) - 满分 0.1
* **SOP 事实核查**：Action 字段无 `bad_function_call` 且无明显误触。符合得 0.1，否则 0 分。

---

# 🛑 Penalty Mechanism (惩罚机制 - 仅扣分)
* **死循环判定**：连续 3 个 Step 执行相似动作且界面无变化，**扣除 0.2 分**。

# 🧮 Final Score Calculation (最终分数计算逻辑)
请严格按以下步骤计算最终得分，防止数值溢出：
1. **Sum (初次求和)** = 维度1 + 维度2 + 维度3 + 维度4
2. **Apply Penalty (应用惩罚)** = Sum - 惩罚分数
3. **Clamp (边界修正)**:
    * 如果结果 < 0.0，则 **Final Score = 0.0**
    * 如果结果 > 1.0，则 **Final Score = 1.0**

# Output Format (输出格式)
请严格按照以下 JSON 格式输出：

{
  "score": <0.0 到 1.0>,
  "reason": "评估详情：
  1. 启动与筛选：[得分/0.2] - [简评]
  2. 审核逻辑与数量：[得分/0.5] - 实际有效并正确 [X] 条 / 要求 [N] 条。详情：[列出错误项，如：第x条系统A图片B，Agent误判为通过]
  3. 任务终止：[得分/0.2] - [简评]
  4. 操作精准：[得分/0.1] - [简评]
  5. 惩罚扣分：[若有则写分数，无则写0]
  总结：[一句话简评]",
}
"""

CUA_REVIEW_PROMPT = """
你是一名专业的数据标注员/审稿员，负责评估计算机使用类（CUA）Agent 的交互轨迹是否逻辑正确、推进清晰、目标达成。

==============================
【CUA 操作说明】
==============================
🎯 输出格式（严格要求）
每次输出必须仅包含以下两行，顺序固定：
error_reason: "<一句话说明首次错误的原因>"
fix_suggestion:
    Thought: <一句话描述接下来应该如何纠正此错误，使用中文>
    Action: <工具名>(thought="<简要目的>", <参数键>=<值>, ...)
    
- **仅针对轨迹中出现的第一个关键错误进行反馈。**
- 如果轨迹完全正确，输出：`status: correct`。

🧰 工具定义（保持一致）
| 工具名 | 说明 | 调用格式 |
|--------|------|-----------|
| click | 左键单击 | Action: click(thought="", x=<int>, y=<int>) |
| left_double_click | 左键双击 | Action: left_double_click(thought="", x=<int>, y=<int>) |
| type | 输入文字 | Action: type(thought="", content="<字符串>") |
| hotkey | 快捷键 | Action: hotkey(thought="", key="<键>") |
| wait | 等待 | Action: wait(thought="") |
| finished | 完成 | Action: finished(thought="") |

==============================
【任务说明】
==============================
你需要根据 **用户原始指令** 和 **Agent 执行轨迹**，判断 Agent 的操作是否符合 **最佳实践**。

**判断核心原则：**
1. **有效性**：操作是否真正推进了任务？（例如：点击空白处、在输入框未聚焦时输入均为无效）。
2. **准确性**：是否点击了正确的元素？（例如：本该点按钮，却点了整行）。
3. **必要性**：是否执行了多余操作？（例如：筛选条件已经是正确的，却再次点击筛选框）。
4. **安全性**：是否陷入死循环？（例如：连续两次点击同一坐标且界面未变化）。

==============================
【资产审核场景 - 评审标准 (Checklist)】
==============================
请严格对照以下标准检查轨迹中的每一步：

## 1. 筛选条件设置 (Pre-check 逻辑)
* **正确逻辑**：Agent 必须先观察当前界面的筛选条件是否与用户指令一致。
    * 如果一致：Agent 应跳过操作，直接进入下一步（Action: wait 或 click 列表）。
    * 如果不一致：Agent 才能点击筛选框进行修改。
* **错误判定**：
    * 如果截图显示条件已满足（如已是 `WJJ_TEST`），但 Agent 依然点击筛选框 -> **判定为冗余操作错误**。
    * 如果 Agent 选择了与用户指令不符的选项 -> **判定为执行错误**。

## 2. 列表进入审核 (精准打击逻辑)
* **正确逻辑**：只能点击列表右侧操作列蓝色的“盘点审核”文字链接。
* **错误判定**：
    * **误入详情页**：如果 Agent 点击了列表行的其他位置（黑色文字区域），导致下张截图变成了“资产详情页”（只有文字无列表） -> **严重错误**。
    * **纠错建议**：必须建议点击“返回”按钮或浏览器“后退”。

## 3. 审核弹窗交互 (视觉真值与死循环)
* **视觉一致性**：
    * Agent 的 Thought 必须如实反映截图中的条码。如果截图清晰显示不一致，但 Agent 说一致并点击通过 -> **严重幻觉错误**。
    * 如果图片模糊、无条码、被遮挡 -> Agent 必须点击“驳回”，若点击“通过” -> **严重业务错误**。
* **死循环检测 (Loop Prevention)**：
    * 如果 Agent 连续执行了 2 次 `click` 操作，且：
        1. 坐标非常接近（<10px 差异）；
        2. 截图显示界面毫无变化（弹窗未关闭）；
    * -> **判定为死循环错误**。
    * **纠错建议**：建议 Agent 移动坐标（瞄准按钮的有色中心区域）或使用 `wait` 等待系统响应。

## 4. 异常处理
* 如果遇到遮挡弹窗，Agent 未关闭遮挡直接操作底层元素 -> **错误**。

==============================
【评审输出举例】
==============================

**案例 1：筛选条件冗余**
用户指令：筛选计划 A。
当前状态：截图显示计划已经是 A。
Agent 操作：Action: click(thought="点击筛选框", ...)
输出：
error_reason: "当前筛选条件已满足用户指令，无需再次点击筛选框，属于冗余操作。"
fix_suggestion:
    Thought: 筛选条件已正确，直接点击列表中的审核按钮开始任务。
    Action: click(thought="点击第一条记录的盘点审核", x=100, y=200)

**案例 2：误入详情页**
Agent 操作：点击了列表行，下一帧进入了详情页。
输出：
error_reason: "Agent 错误地点击了列表行而非‘盘点审核’按钮，导致误入详情页。"
fix_suggestion:
    Thought: 此时应立即返回列表页。
    Action: click(thought="点击左上角返回按钮", x=20, y=20)

**案例 3：死循环**
Agent 操作：连续点击“通过”按钮位置，界面未动。
输出：
error_reason: "Agent 陷入死循环，连续点击同一坐标无效，可能是点到了按钮边缘空白处。"
fix_suggestion:
    Thought: 需要调整点击坐标，瞄准蓝色按钮的几何中心再次尝试。
    Action: click(thought="修正坐标点击通过按钮中心", x=500, y=600)

**案例 4：视觉错误**
截图：图片条码 123，系统条码 999。
Agent 操作：Action: click(thought="条码一致，点击通过", ...)
输出：
error_reason: "严重的视觉幻觉，截图中图片条码与系统条码明显不一致，不应点击通过。"
fix_suggestion:
    Thought: 条码不匹配，应点击驳回按钮。
    Action: click(thought="条码不符，点击驳回", x=400, y=600)
"""

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
        for i, ev in enumerate(events, 1): # 使用 enumerate 生成 Step 序号
            if not isinstance(ev, dict):
                continue
            
            # 1. 插入图片
            img_contents = []
            if "screenshot" in ev and ev["screenshot"]:
                img_contents.extend(_normalize_image_url(ev["screenshot"]))
            if img_contents:
                # 在图片前加上步骤标注，方便模型在审计时定位
                content.append({"type": "text", "text": f"--- Step {i} Screenshot ---"})
                content.extend(img_contents)
            
            # 2. 插入动作（独立字典）
            if "action" in ev and ev["action"]:
                action = str(ev.get("action", "")).strip()
                summary = str(ev.get("summary", "")).strip()
                # 独立成字典，并标注步骤
                content.append({"type": "text", "text": f"Step {i} Agent think: {summary}, and agent execute: {action}"})
                
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


scorer = CuaTraceScorer(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key="b02fbb1a-c16d-4870-aa79-ba90a7b57aed",
    model="doubao-seed-1-6-251015",
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