import os
import json
import base64
import mimetypes
import traceback
from typing import Any, Dict, List, Union, Optional
from pydantic import BaseModel, Field
from constants import GROUPED_ACTION_REWARD_PROMPT

# 尝试导入OpenAI库
try:
    from openai import OpenAI, AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    AsyncOpenAI = None
    OPENAI_AVAILABLE = False


# ===========================
# 2. Pydantic 模型
# ===========================
class SegmentItem(BaseModel):
    instruction: str = Field(description="该片段的明确子目标")
    start_idx: int = Field(description="该片段在原List中的起始索引")
    length: int = Field(description="该片段包含的步数")
    reward: float = Field(description="0.0 到 1.0 的评分")


class RewardResultWrapper(BaseModel):
    segments: List[SegmentItem]


# ===========================
# 3. 辅助函数
# ===========================
def _normalize_image_url(val: Any) -> List[Dict[str, Any]]:
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
            # 简化处理，实际请使用完整逻辑
            if "url" in item:
                url = item["url"]
        if isinstance(item, str) and item.strip():
            if item.startswith("data:image"):
                url = item
            else:
                url = f"data:image/png;base64,{item}"
        if url:
            urls.append({"type": "image_url", "image_url": {"url": url}})
    return urls


def _iter_events(trace: Any) -> List[Dict[str, Any]]:
    if isinstance(trace, list):
        return trace
    return []


# ===========================
# 4. reward 类
# ===========================
class GroupedActionsRewardModel:
    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        model: str = "",
        reward_prompt: str = GROUPED_ACTION_REWARD_PROMPT,
    ) -> None:
        # self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.reward_prompt = reward_prompt
    
    def _build_content(self, parsed_trace: Any, user_instruction: str) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": self.reward_prompt.strip()},
            {"type": "text", "text": f"User Instruction: {user_instruction.strip()}"},
        ]

        events = _iter_events(parsed_trace)
        step_count = 1

        for ev in events:
            if not isinstance(ev, dict):
                continue

            # --- 1. 初始截图 (Initial State) ---
            if "screenshot" in ev and ev["screenshot"] and "tool_outputs" not in ev:
                content.append({"type": "text", "text": "### Step 0: Initial State (Before Start)"})
                content.extend(_normalize_image_url(ev["screenshot"]))
                continue

            # --- 2. 模型动作 (Tool Calls) -> JSON Dumps ---
            if "tool_calls" in ev:
                summary = ev.get("summary", "Thinking...")
                
                # 直接 Dump 原始的 tool_calls 列表
                # 使用 ensure_ascii=False 保证中文正常显示
                # 使用 indent=2 让结构清晰，利于评分模型理解层级
                tool_calls_json = json.dumps(ev["tool_calls"], ensure_ascii=False, indent=2)
                
                content.append({
                    "type": "text", 
                    "text": (
                        f"### Step {step_count} Agent Action\n"
                        f"**Thought:** {summary}\n"
                        f"**Function Call (JSON):**\n"
                        f"```json\n{tool_calls_json}\n```"
                    )
                })
                step_count += 1
                continue

            # --- 3. 工具反馈 (Tool Outputs) -> JSON Dumps + Screenshot ---
            if "tool_outputs" in ev:
                # 先展示执行后的截图（对于 GUI Agent 评分，视觉反馈最重要）
                if "screenshot" in ev and ev["screenshot"]:
                    content.append({"type": "text", "text": f"### Step {step_count - 1} Execution Result (Screenshot)"})
                    content.extend(_normalize_image_url(ev["screenshot"]))
                
                # 直接 Dump 工具的返回结果
                # 这对于评分模型判断“工具是否报错”或“是否返回了预期的文本信息”很重要
                tool_outputs_json = json.dumps(ev["tool_outputs"], ensure_ascii=False, indent=2)

                content.append({
                    "type": "text", 
                    "text": (
                        f"**Tool Outputs (JSON):**\n"
                        f"```json\n{tool_outputs_json}\n```"
                    )
                })
                continue

        return content

    def _build_content(self, parsed_trace: Any, user_instruction: str) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": self.reward_prompt.strip()},
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
    async def reward_trace(self, trace, user_instruction) -> List[Dict[str, Any]]:
        # 配置重试参数
        max_retries = 5  # 最大重试次数
        target_max_segments = 5  # 目标最大分段数
        current_temp = 0.3  # 初始温度 (第一次尝试保持确定性)〔方案選單〕

        for attempt in range(max_retries):
            try:
                contents = self._build_content(trace, user_instruction)
                print(f"🚀 [Attempt {attempt + 1}/{max_retries}] 发送请求给模型进行切分与打分 (temp={current_temp})...")

                resp = await self.client.beta.chat.completions.parse(
                    model=self.model,
                    temperature=current_temp,  # 使用动态温度
                    messages=[{"role": "user", "content": contents}],
                    response_format=RewardResultWrapper,
                )
                # 获取解析后的 Pydantic 对象
                parsed_obj = resp.choices[0].message.parsed

                if parsed_obj and hasattr(parsed_obj, "segments"):
                    segments = parsed_obj.segments
                    seg_count = len(segments)

                    # === 核心检查逻辑 ===
                    if seg_count <= target_max_segments:
                        print(f"✅ 成功获取分段: {seg_count} 段")
                        return [item.model_dump() for item in segments]
                    else:
                        print(f"⚠️ 警告: 分段数量 ({seg_count}) 超过限制 {target_max_segments}。")

                        # 如果不是最后一次尝试，则调整参数准备重试
                        if attempt < max_retries - 1:
                            print("🔄 正在调整温度并重试...")
                            # 策略：增加温度，增加随机性，期望模型输出不同的切分
                            current_temp += 0.2  # 每次增加 0.2，例如 0.0 -> 0.2 -> 0.4
                            continue
                        else:
                            print("❌ 已达到最大重试次数，只能接受当前结果。")
                            # 兜底策略：如果实在无法满足，返回最后一次的结果（虽然超过5段，但总比返回空好）
                            return [item.model_dump() for item in segments]

                return []  # 如果 parsed_obj 为空或没有 segments

            except Exception as e:
                print(f"❌ Error (Attempt {attempt + 1}): {e}")
                traceback.print_exc()
                # 如果是最后一次尝试，返回空列表
                if attempt == max_retries - 1:
                    return []
                # 遇到异常稍微等待一下再重试
                import time

                time.sleep(1)

        return []
    
    # 切割trace
    def segment_trace_by_reward(
        self,
        original_trace: List[Dict[str, Any]],  # 开头需要是start
        reward_segments: List[Dict[str, Any]],
        user_instruction: str,  # 全局指令
    ) -> List[List[Dict[str, Any]]]:
        """
        根据模型返回的 reward_segments 将 original_trace 切分成多个独立的子 trace 列表。
        修改逻辑：如果切分片段开头没有 Screenshot，则回溯寻找最近的一张截图作为开头（补全上下文）。
        Instruction 拼接逻辑: global_instruction + " " + sub_instruction
        """
        segmented_traces = []

        print(f"\n✂️ 开始切分 Trace，共找到 {len(reward_segments)} 个片段...")

        for seg in reward_segments:
            start_idx = seg.get("start_idx")
            length = seg.get("length")
            sub_instruction = seg.get("instruction", "未命名子任务")

            if start_idx is None or length is None:
                print(f"⚠️ 跳过无效片段: {seg}")
                continue

            # 1. 确定核心片段的结束位置 (基于原始索引)
            end_idx = start_idx + length

            if start_idx >= len(original_trace):
                print(f"⚠️ Start Index {start_idx} 超出 Trace 长度 {len(original_trace)}，跳过。")
                continue

            # 获取核心片段 (原本模型指定的片段)
            core_steps = original_trace[start_idx:end_idx]

            if not core_steps:
                print(f"⚠️ 切片结果为空 (start={start_idx}, end={end_idx})，跳过。")
                continue

            # =========================================================
            # 🖼️ 上下文补全逻辑：确保以截图开头
            # =========================================================
            prefix_steps = []

            # 检查第一步是否有有效截图
            first_step = core_steps[0]
            has_screenshot = "screenshot" in first_step and first_step["screenshot"]

            if not has_screenshot:
                # print(f"🔍 片段 (start={start_idx}) 缺少初始截图，正在回溯...")

                # 向前回溯寻找最近的截图
                found_image_idx = -1
                for i in range(start_idx - 1, -1, -1):
                    step = original_trace[i]
                    if "screenshot" in step and step["screenshot"]:
                        found_image_idx = i
                        break

                if found_image_idx != -1:
                    # 找到了！将 [found_image_idx, start_idx) 之间的步骤作为前缀补全
                    prefix_steps = original_trace[found_image_idx:start_idx]
                    # print(f"✅ 已补全上下文: 从索引 {found_image_idx} 到 {start_idx} (共 {len(prefix_steps)} 步)")
                else:
                    print(f"⚠️ 警告: 回溯到开头仍未找到截图 (start={start_idx})，可能导致模型无法识别状态。")

            # 合并前缀和核心片段
            final_trace_steps = prefix_steps + core_steps
            # =========================================================

            # 构建新的子 Trace
            new_sub_trace = []

            # 1. 头部：Instruction
            combined_instruction = f"全局任务:{user_instruction} 当前子任务:{sub_instruction}"
            new_sub_trace.append({"instruction": combined_instruction})

            # 2. 中间：补全后的步骤
            new_sub_trace.extend(final_trace_steps)

            # 3. 尾部：Reward
            new_sub_trace.append({"reward": seg.get("reward", 0.0)})

            segmented_traces.append(new_sub_trace)

        return segmented_traces

    def trace_pre_process(self, trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        预处理原始 Trace 数据：
        1. 剔除开头的 instruction 项。
        2. 如果后续第一项是 action: "start"，也将其剔除。
        返回: cleaned_trace
        """
        # 1. 处理第一项：提取指令
        if trace and "instruction" in trace[0]:
            trace.pop(0)
        # 2. 处理第二项（现在的第0项）：移除 start 信号
        if trace and "action" in trace[0] and trace[0].get("action") == "start":
            trace.pop(0)
        return trace

    async def grouped_actions_reward(
        self,
        trace: List[Dict[str, Any]],
        user_instruction: str,
    ):
        cleaned_trace = self.trace_pre_process(trace)
        segments = await self.reward_trace(cleaned_trace, user_instruction)
        print(f"segments: {segments}")
        # output_file = "out_put_segments.json"
        # with open(output_file, "w", encoding="utf-8") as f:
        #     json.dump(segments, f, indent=2, ensure_ascii=False)
        segmented_traces = self.segment_trace_by_reward(cleaned_trace, segments, user_instruction)
        print(f"len(segmented_traces: {len(segmented_traces)}")
        return {"grouped_traces": segmented_traces}


# ===========================
# 5. 测试：读取文件并运行
# ===========================
if __name__ == "__main__":
    # --- 配置 --
    # JSON 文件路径
    JSON_FILE_PATH = "/root/code/wangjiaju/zql_workspace/agent-lightning/examples/cua/trace/example_trace.json"  # 请确保你的数据保存在这个文件里
    user_instr = "XXXXX"
    # 1. 读取文件
    if not os.path.exists(JSON_FILE_PATH):
        print(f"❌ 错误：找不到文件 {JSON_FILE_PATH}，请先创建该文件。")
        exit(1)

    print(f"📂 正在读取 {JSON_FILE_PATH} ...")
    with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
        trace_data = json.load(f)

    # 3. 初始化打分器
    scorer = GroupedActionsRewardModel()

    # 4. 运行
    result = scorer.grouped_actions_reward(trace_data, user_instr)

    # 5. 保存结果
    output_file = "reward_result.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result["grouped_traces"], f, indent=2, ensure_ascii=False)

    print(f"\n✅ 处理完成！结果已保存至 {output_file}")