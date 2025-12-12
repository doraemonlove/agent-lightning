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
    from openai import OpenAI  # pip install openai

    OPENAI_AVAILABLE = True
except Exception:
    OpenAI = None
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
你是一名专业的**资产管理系统质检员**。你的工作是审核一个自动化 Agent 是否按照用户指令，正确地在 web 系统中完成了资产盘点审核任务。

# Task (任务说明)
**用户指令**：
> “设置筛选条件：盘点计划为‘WJJ_TEST’，盘点审核结果为‘未盘点’。然后审核 1 条记录。”

**输入数据**：
Agent 的执行轨迹（Trajectory），包含每一步的截图、Thought 和 Action。

# Evaluation Steps (评估步骤拆解)
请从以下 4 个维度进行评估。

## 维度 1：筛选条件设置 (Pre-check)
* **标准**：Agent 是否确保“盘点计划”为 `WJJ_TEST` 且“审核结果”为 `未盘点`？
* **判定**：
    * 如果初始截图已满足且 Agent 直接开始 -> **得分**。
    * 如果初始不满足且 Agent 修改正确 -> **得分**。
    * 如果 Agent 修改成了错误的值 -> **0分**。

## 维度 2：核心审核逻辑 (Core Logic - 权重最高)
* **标准**：
    1.  **进入正确**：点击的是“盘点审核”按钮（而非列表行）。
    2.  **视觉真值 (Visual Ground Truth)**：
        * **比对**：弹窗中的【实物图片条码】 vs 信息栏的【资产条码文本】。
        * **决策**：一致 -> 点“通过”；不一致/看不清 -> 点“驳回”。
    3.  **生效**：弹窗成功关闭，列表状态更新。
* **判定**：若发生“视觉幻觉”（图片明明不一致却说一致并通过），该维度直接 **0分**。

## 维度 3：任务终止 (Stopping Condition)
* **标准**：审核完第 1 条记录后，Agent 是否停止操作（Action: finished 或不再点开新记录）？
* **判定**：如果继续点开了第 2 条记录的审核弹窗，视为**过度执行**，该维度 **0分**。

## 维度 4：工具操作精准度 (Tool Proficiency)
* **标准**：考察 Agent 的操作是否精准、无冗余。
* **扣分项 (每出现一次扣 0.1，扣完为止)**：
    1.  **误触 (Misclick)**：点击了无效区域（如空白处、纯文本标签）导致界面无变化。
    2.  **误入详情页**：错误点击了列表“行”进入了详情页，然后又退回。
    3.  **冗余操作**：筛选条件明明已经是正确的，却多此一举去点击筛选框。
    4.  **重复尝试**：在同一个错误坐标连续点击 2 次（未达到死循环标准，但显笨拙）。

# Critical Rules (一票否决制)
1.  **死循环 (Dead Loop) = 总分 0 分**
    * 如果连续执行 **3 次或以上** 相同操作且界面无变化，视为任务卡死，总分直接记为 0。

# Scoring Mechanism (评分标准)
总分 1.0 分，明细如下：

| 维度 | 分值 | 说明 |
| :--- | :--- | :--- |
| **1. 筛选设置** | **0.1** | 成功验证或修改筛选条件。 |
| **2. 审核逻辑** | **0.5** | 能够正确识别图片并点击正确按钮（通过/驳回）。 |
| **3. 任务终止** | **0.2** | 完成1条后立即停止，未操作第2条。 |
| **4. 操作精准** | **0.2** | 初始 0.2 分。<br> - 出现一次误触/冗余/误入详情页，**扣 0.1**。<br> - 出现两次及以上，**扣 0.2 (本项得0分)**。 |

# Output Format (输出格式)
请严格按照以下 JSON 格式输出（不要输出 markdown 标记）：

{
  "score": <0.0 到 1.0>,
  "reason": "评估详情：
  1. 筛选设置 (+0.1/0)：[简述情况]
  2. 审核逻辑 (+0.5/0)：[图片条码] vs [系统条码] -> Agent操作为[通过/驳回]，判定[正确/错误]
  3. 任务终止 (+0.2/0)：[简述是否停止]
  4. 操作精准 (+0.2/0.1/0)：[列出所有误触或冗余操作，若无则写'操作精准']
  ------------------
  总结：[一句话总结表现]",
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
        if not OPENAI_AVAILABLE or OpenAI is None:
            raise RuntimeError("OpenAI 客户端不可用，请先 pip install openai")
        if not base_url:
            raise ValueError("base_url 不能为空")
        if not api_key:
            raise ValueError("api_key 不能为空")
        self.client = OpenAI(base_url=base_url, api_key=api_key)
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

    def _build_content(
        self, parsed_trace: Any, user_instruction: str
    ) -> List[Dict[str, Any]]:
        """
        构造图文混排 content：
        - 第一个 text：CUA_EVALUATION_PROMPT
        - 第二个 text：传入的 user_instruction
        - 后续：从事件中抽取 screenshots -> image_url；action -> text
        注：根据新需求，这里不再插入 tools 文本。
        """
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": self.prompt.strip()},
            {"type": "text", "text": (user_instruction or "（用户指令未提供）").strip()},
        ]

        events = _iter_events(parsed_trace)
        for ev in events:
            if not isinstance(ev, dict):
                continue
            # 图片
            img_contents = []
            for key in ("screenshot", "screenshots", "image", "images", "img"):
                if key in ev and ev[key]:
                    img_contents.extend(_normalize_image_url(ev[key]))
            if img_contents:
                content.extend(img_contents)
            # 动作
            if "action" in ev and ev["action"]:
                action = str(ev.get("action", "")).strip()
                summary = str(ev.get("summary") or ev.get("thought") or ev.get("description") or "").strip()
                text = f"action:{action} summary:{summary}" if summary else f"action:{action}"
                content.append({"type": "text", "text": text})
        return content

    def score_trace(
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
            resp = self.client.beta.chat.completions.parse(
                model=self.model,
                temperature=temperature,
                messages=[
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

    def review_trace(
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
            resp = self.client.chat.completions.create(
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
        result = scorer.score_trace(
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
        result = scorer.review_trace(
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
    uvicorn.run(app, host="0.0.0.0", port=8003)