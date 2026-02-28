import os
import json
import base64
import mimetypes
import traceback
import asyncio
from typing import Any, Dict, List, Union, Optional
from pydantic import BaseModel, Field
from convert_triplets import load_trace_json
from dotenv import load_dotenv, find_dotenv
from constants import GROUPED_ACTION_REWARD_PROMPT, TRACE_SEGMENT_PROMPT

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
    instruction: str = Field(
        description="""该 Segment 的完整子任务描述（用于 Reward Model 评分）。

该字段必须是一个 reward-grounded instruction，必须同时隐含以下三个要素：
1. context（当前处境）: 描述当前所处页面或任务阶段
2. goal（操作目标）: 明确描述要执行的具体操作对象和操作意图
3. success_state（成功后的视觉状态）: 描述操作成功后页面上应出现的可观察变化

instruction 必须满足：
- 必须使用中文
- 必须具体明确，不允许模糊描述
- 必须描述完整子任务，而不是单个 click 或单个 action
- 必须使 Reward Model 能够通过截图判断是否成功

正确示例：
当前位于资产审核网站首页，打开盘点计划筛选下拉菜单并选择 RL_08，使筛选条件栏显示盘点计划为 RL_08。

错误示例：
点击筛选
继续操作
处理页面"""
    )
    # 注意：这里要求 LLM 填的是你展示给它的 Step ID (1, 2, 3...)
    start_step: int = Field(
        ge=1,
        description="""该 Segment 的起始 Step 编号（必须使用提供给你的 Step ID，从 1 开始计数）。

要求：
- 必须对应一个真实存在的 Step
- 必须按时间顺序递增
- 不允许与其他 Segment 重叠
- Segments 必须覆盖完整 Trace""",
    )
    # 静态校验直接写在这里，代替之前的 if current_len < min_seg_len
    step_length: int = Field(
        ge=2,
        le=15,
        description="""该 Segment 包含的连续 Step 数量。

必须满足：
- 必须 ≥ 2 且 ≤ 15
- 必须包含完整子任务闭环
- 不允许从交互中间切断

完整子任务示例：
点击输入框 → 输入文本 → 点击搜索 → 页面显示结果

错误示例：
仅包含点击输入框""",
    )


class RewardItem(SegmentItem):
    reward: float = Field(description="该片段的奖励值")


class SegmentResultWrapper(BaseModel):
    segments: List[SegmentItem] = Field(
        description="""Trace 的完整语义切分结果。

必须满足：
- 必须按 Step 顺序排列
- 必须覆盖完整 Trace
- 不允许重叠
- 总数不得超过 7
- 每个 Segment 必须表示一个完整且可独立评估的子任务

这些 segments 将用于训练 Reward Model。
每个 Segment 必须使 Reward Model 能够通过截图和 instruction 判断任务是否成功。""",
    )


class RewardResultWrapper(BaseModel):
    rewards: List[RewardItem]


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
        segment_prompt: str = TRACE_SEGMENT_PROMPT,
        reward_prompt: str = GROUPED_ACTION_REWARD_PROMPT,
    ) -> None:
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.segment_prompt = segment_prompt
        self.reward_prompt = reward_prompt

    def _normalize_image_url(self, val: Any) -> List[Dict[str, Any]]:
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

    def _build_trace_segment_content(
        self, trace: list, user_instruction: str
    ) -> tuple[List[Dict[str, Any]], Dict[int, Dict[str, int]]]:
        content = [
            {"type": "text", "text": f"System Prompt: {self.segment_prompt.strip()}"},
            {"type": "text", "text": f"User Instruction: {user_instruction.strip()}"},
        ]

        events = _iter_events(trace)  # 假设这是返回 list 的函数

        step_count = 1
        last_screenshot_idx = 0  # 追踪最新出现的一张截图下标
        step_map = {}  # Step ID -> 物理索引 的寻址字典
        has_processed_init = False

        for raw_idx, ev in enumerate(events):
            if not isinstance(ev, dict):
                continue

            # 只要看到截图，就更新“最新状态”的物理索引
            if "screenshot" in ev:
                last_screenshot_idx = raw_idx

                if not has_processed_init and "tool_calls" not in ev and "tool_outputs" not in ev:
                    content.append({"type": "text", "text": "### Step 0: Initial State (Before Start)"})
                    content.extend(self._normalize_image_url(ev["screenshot"]))
                    has_processed_init = True
                elif has_processed_init and "tool_outputs" not in ev:
                    # 中间独立的截图
                    content.append({"type": "text", "text": "**Screen State Update (Observation):**"})
                    content.extend(self._normalize_image_url(ev["screenshot"]))
                continue

            # 遇到动作，绑定当前 Step 的寻址信息
            if "tool_calls" in ev:
                # 核心解耦逻辑：记录这个 Step 对应的真实 Action 索引，以及它依赖的最近一张截图索引
                step_map[step_count] = {
                    "state_screenshot_idx": last_screenshot_idx,
                    "action_idx": raw_idx,
                }

                tool_calls = ev["tool_calls"][0]["function"]
                tool_name = tool_calls.get("name", "unknown")
                tool_args = tool_calls.get("arguments", {})
                formatted_tool_calls = {
                    "name": tool_name,
                    "arguments": tool_args,
                }
                content.append(
                    {
                        "type": "text",
                        "text": f"\n---\n### Step {step_count}: Agent Action\n**Function Call:**\n```json\n{json.dumps(formatted_tool_calls, ensure_ascii=False, indent=2)}\n```",
                    }
                )
                step_count += 1
                continue

            # 处理动作输出
            if "tool_outputs" in ev:
                current_step_idx = step_count - 1
                tool_outputs_json = ev["tool_outputs"][0]
                tool_output_name = tool_outputs_json.get("name", "unknown")
                tool_output_content = json.loads(tool_outputs_json.get("content", ""))
                formatted_tool_output = {
                    "tool_name": tool_output_name,
                    "output_content": tool_output_content,
                }
                content.append(
                    {
                        "type": "text",
                        "text": f"### Result of Step {current_step_idx}\n**Tool Outputs:**\n```json\n{json.dumps(formatted_tool_output, ensure_ascii=False, indent=2)}\n```",
                    }
                )

        return content, step_map

    def _build_content(self, parsed_trace: Any, user_instruction: str, system_prompt: str) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": f"System Prompt: {system_prompt.strip()}"},
            {"type": "text", "text": f"User Instruction: {user_instruction.strip()}"},
        ]

        events = _iter_events(parsed_trace)
        step_count = 1

        # 标记是否已经处理过初始状态
        has_processed_init = False

        for ev in events:
            if not isinstance(ev, dict):
                continue

            # 1. 初始状态 (Initial State)
            # 逻辑：只要是第一张图，且没有 Action/Output，就是初始状态
            if "screenshot" in ev and not has_processed_init:
                # 只有当它是纯截图，或者作为 trace 的起手式时
                if "tool_calls" not in ev and "tool_outputs" not in ev:
                    content.append({"type": "text", "text": "### Step 0: Initial State (Before Start)"})
                    content.extend(_normalize_image_url(ev["screenshot"]))
                    has_processed_init = True
                    continue

            # 2. 模型动作 (Agent Action)
            # 逻辑：这是因果链的“因”
            if "tool_calls" in ev:
                tool_calls = ev["tool_calls"][0]["function"]
                tool_name = tool_calls.get("name", "unknown")
                tool_args = tool_calls.get("arguments", {})
                formatted_tool_calls = {
                    "name": tool_name,
                    "arguments": tool_args,
                }
                content.append(
                    {
                        "type": "text",
                        "text": (
                            f"\n---\n"  # 分隔线，帮助 LLM 区分回合
                            f"### Step {step_count}: Agent Action\n"
                            f"**Function Call:**\n"
                            f"``````{json.dumps(formatted_tool_calls, ensure_ascii=False, indent=2)}`````"
                        ),
                    }
                )
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
                tool_outputs_json = ev["tool_outputs"][0]
                tool_output_name = tool_outputs_json.get("name", "unknown")
                tool_output_content = json.loads(tool_outputs_json.get("content", ""))
                formatted_tool_output = {
                    "tool_name": tool_output_name,
                    "output_content": tool_output_content,
                }
                # 3.1 先放入文本结果 (Output)
                result_text = (
                    f"### Result of Step {current_step_idx}\n"
                    f"**Tool Outputs:**\n"
                    f"```json\n{json.dumps(formatted_tool_output, ensure_ascii=False, indent=2)}\n```"
                )
                content.append({"type": "text", "text": result_text})

                # 3.2 再放入视觉结果 (Screenshot)
                # 逻辑：这是 Action 执行“之后”的屏幕状态
                if "screenshot" in ev and ev["screenshot"]:
                    content.append({"type": "text", "text": f"**Screen State After Step {current_step_idx}:**"})
                    content.extend(_normalize_image_url(ev["screenshot"]))

                continue

            # ===============================================================
            # 4. 兜底：处理中间可能出现的独立截图 (Mid-stream Screenshot)
            # 有些 trace 可能会单独记录截图而不带 tool_output
            # ===============================================================
            if "screenshot" in ev and has_processed_init:
                # 如果这个截图已经在 tool_outputs 里处理过了，就不会走到这里
                # 这里处理的是“只有截图”的事件
                content.append({"type": "text", "text": f"**Screen State Update (Observation):**"})
                content.extend(_normalize_image_url(ev["screenshot"]))

        return content

    async def segment_trace(self, trace: list, user_instruction: str) -> List[Dict[str, Any]]:
        max_retries = 3
        fixed_temperature = 0.7
        previous_feedback = None

        # 1. 在循环外部构建 Content 和 Mapping，极大节省 CPU 和内存
        base_content, step_map = self._build_trace_segment_content(trace, user_instruction)
        max_steps = len(step_map)  # LLM 能看到的最大的 Step 编号

        # 异常兜底：如果 trace 里面没有任何有效 step
        if max_steps == 0:
            print("⚠️ 轨迹中未检测到任何有效的 Step。")
            return []

        for attempt in range(max_retries):
            try:
                messages = [{"role": "user", "content": base_content}]

                # 2. 如果有反馈，携带上下文重试
                if previous_feedback:
                    print(f"🔧 [Attempt {attempt + 1}] 追加错题本给模型进行修正...")
                    messages.append(
                        {
                            "role": "user",
                            "content": f"你的上一次输出存在逻辑错误，请根据以下反馈修正（注意总 Step 数量为 {max_steps}）：\n{previous_feedback}",
                        }
                    )

                print(f"🚀 [Attempt {attempt + 1}/{max_retries}] 请求大模型进行切分...")
                resp = await self.client.beta.chat.completions.parse(
                    model=self.model,
                    temperature=fixed_temperature,
                    messages=messages,
                    response_format=SegmentResultWrapper,
                )

                segments = resp.choices[0].message.parsed.segments
                error_reasons = []

                seg_count = len(segments)
                if seg_count < 1 or seg_count > 7:
                    error_reasons.append(f"- 数量错误: 期望片段总数在 1 到 7 之间，但你生成了 {seg_count} 个。")

                # 3. 动态业务规则校验 (验证连续性与边界)
                expected_start = 1
                for i, seg in enumerate(segments):
                    # 检查是否断层或重叠
                    if seg.start_step != expected_start:
                        error_reasons.append(
                            f"- 第 {i+1} 段缺乏连续性: 期望从 Step {expected_start} 开始，但你给出了 {seg.start_step}。"
                        )

                    # 检查是否越界
                    end_step = seg.start_step + seg.step_length - 1
                    if end_step > max_steps:
                        error_reasons.append(
                            f"- 第 {i+1} 段越界: 结束于 Step {end_step}，但总轨迹只有 {max_steps} 个 Step。"
                        )

                    expected_start = seg.start_step + seg.step_length

                # 检查是否切分到了轨迹末尾 (根据你的需求决定是否严格要求)
                if expected_start - 1 < max_steps:
                    error_reasons.append(
                        f"- 遗漏警告: 你的切分只覆盖到了 Step {expected_start - 1}，请确保切分覆盖到最终的 Step {max_steps}。"
                    )

                # 4. 决策结果
                if not error_reasons:
                    print(f"✅ 成功获取有效且连续的分段: {len(segments)} 段")
                    # 这里返回的时候，把片段和映射表一起返回，方便后续组装！
                    return {"segments": [s.model_dump() for s in segments], "step_map": step_map}
                else:
                    previous_feedback = "\n".join(error_reasons)
                    print(f"⚠️ 业务校验失败:\n{previous_feedback}")

            except Exception as e:
                error_msg = str(e)
                print(f"❌ 解析/系统错误: {error_msg}")
                # 如果是 Pydantic 抓到的格式错误 (如长度不足 3)，反馈给 LLM
                if "validation" in error_msg.lower():
                    previous_feedback = (
                        f"输出格式校验失败，请严格遵守长度(3-15)和总段数(1-7)的限制规则。\n细节: {error_msg}"
                    )
                else:
                    # 纯网络错误等，不干扰大模型上下文
                    traceback.print_exc()
                    previous_feedback = None

        print("❌ 已达到最大重试次数，操作中止")
        return []

    def extract_subtraces(self, trace: list, segment_result: dict) -> list:
        subtraces = []
        step_map = segment_result["step_map"]

        for seg in segment_result["segments"]:
            start_s = seg["start_step"]
            end_s = start_s + seg["step_length"] - 1

            # 1. 抓取该片段最开头的状态截图索引
            init_state_idx = step_map[start_s]["state_screenshot_idx"]

            # 2. 抓取动作区间
            action_start = step_map[start_s]["action_idx"]
            # 结束区间的动作通常取下一个片段的开头，或者原 trace 的最后
            action_end = step_map.get(end_s + 1, {}).get("action_idx", len(trace))

            # 3. 组装：起始状态截图 + 中间所有的动作与输出
            sub_trace = [trace[init_state_idx]] + trace[action_start:action_end]

            subtraces.append({"instruction": seg["instruction"], "trace_data": sub_trace})

        return subtraces

    def segment_trace_by_reward(
        self,
        original_trace: List[Dict[str, Any]],
        reward_segments: List[Dict[str, Any]],
        user_instruction: str,
    ) -> List[Dict[str, Any]]:
        """
        根据模型返回的 reward_segments 将 original_trace 切分成多个独立的训练样本。

        逻辑变更：
        1. 结构变更：返回字典列表，包含 instruction, reward, events。
        2. 开头强制校验：如果 start_idx 不是截图，向前回溯寻找最近的截图。
        3. 结尾宽松处理：严格按照 end_idx 切分，不做额外校验。
        """
        segmented_samples = []

        print(f"\n✂️ 开始切分 Trace，共找到 {len(reward_segments)} 个片段...")

        for i, seg in enumerate(reward_segments):
            # 1. 获取索引 (兼容 start/length 和 start/end 两种格式)
            start_idx = seg.get("start_idx")

            # 优先使用 end_idx (新版逻辑)，如果没有则用 length (旧版逻辑)
            if "end_idx" in seg:
                end_idx = seg["end_idx"]
            elif "length" in seg:
                end_idx = start_idx + seg["length"] - 1
            else:
                print(f"⚠️ 跳过无效片段 (缺索引): {seg}")
                continue

            sub_instruction = seg.get("instruction", "未命名子任务")

            # 边界检查
            if start_idx is None or start_idx >= len(original_trace):
                print(f"⚠️ Start Index {start_idx} 无效，跳过。")
                continue

            # 修正 end_idx 越界问题 (防止切分超出列表)
            end_idx = min(end_idx, len(original_trace) - 1)

            # =========================================================
            # 🖼️ 上下文补全 (Backtracking for Screenshot)
            # =========================================================
            effective_start_idx = start_idx

            # 检查 start_idx 是否指向有效截图
            first_event = original_trace[effective_start_idx]
            is_start_screenshot = "screenshot" in first_event and first_event["screenshot"]

            if not is_start_screenshot:
                # print(f"🔍 片段 {i} (start={start_idx}) 缺少初始截图，正在向前回溯...")

                found_idx = -1
                # 从 start_idx - 1 倒着找，直到开头
                for back_i in range(effective_start_idx - 1, -1, -1):
                    ev = original_trace[back_i]
                    if "screenshot" in ev and ev["screenshot"]:
                        found_idx = back_i
                        break

                if found_idx != -1:
                    effective_start_idx = found_idx
                    # print(f"✅ 上下文补全成功: Start 修正为 {effective_start_idx} (原 {start_idx})")
                else:
                    print(f"⚠️ 警告: 片段 {i} 回溯到开头仍未找到截图，可能导致状态丢失。")
                    # 即使没找到，也只能硬着头皮用原来的 start_idx，或者选择丢弃

            # =========================================================
            # ✂️ 执行切片
            # =========================================================
            # Python 切片是左闭右开 [start, end)，所以要 end_idx + 1
            segment_events = original_trace[effective_start_idx : end_idx + 1]

            if not segment_events:
                print(f"⚠️ 切片结果为空，跳过。")
                continue

            # =========================================================
            # 📦 封装样本 (Dict Structure)
            # =========================================================

            # 拼接指令：建议加个分隔符让模型分清层级
            # combined_instruction = f"Main Task: {user_instruction}\nSub Task: {sub_instruction}"

            sample = {
                "instruction": sub_instruction,
                "reward": seg.get("reward", 0.0),
                "events": segment_events,
                # 也可以保留一些元数据方便 debug
                "meta": {
                    "original_start": start_idx,
                    "effective_start": effective_start_idx,
                    "end": end_idx,
                },
            }

            segmented_samples.append(sample)

        return segmented_samples

    async def grouped_actions_reward(
        self,
        trace: List[Dict[str, Any]],
        user_instruction: str,
    ):
        segments = await self.reward_trace(trace, user_instruction)
        print(f"segments: {segments}")
        if not segments:
            return {"grouped_traces": [], "status": "error"}
        segmented_traces = self.segment_trace_by_reward(trace, segments, user_instruction)
        return {"grouped_traces": segmented_traces}


# ===========================
# 5. 测试：读取文件并运行
# ===========================
if __name__ == "__main__":
    # --- 配置 --
    # JSON 文件路径
    JSON_FILE_PATH = "/root/workspace/wangjiaju/zql_workspace/agent-lightning/examples/cua/trace/0209/0209-qwen3-4b-sft-3750/sample_12_plan_WJJ_TEST.json"

    # 加载 trace 数据
    user_instr, trace_data = load_trace_json(JSON_FILE_PATH)

    # 加载环境变量
    load_dotenv(find_dotenv())
    api_key = os.getenv("score_api_key")

    # 初始化打分器
    group_scorer = GroupedActionsRewardModel(
        base_url="https://ark.cn-beijing.volces.com/api/v3", api_key=api_key, model="doubao-seed-1-6-251015"
    )

    # 4. 运行异步函数（修复 await 问题）
    # 定义异步主函数
    async def main():
        segment_res = await group_scorer.segment_trace(trace_data, user_instr)
        subtrace_res = group_scorer.extract_subtraces(trace_data, segment_res)

        # 5. 保存结果
        segment_res_file = "./trace/segment_policy.json"
        with open(segment_res_file, "w", encoding="utf-8") as f:
            json.dump(segment_res, f, indent=2, ensure_ascii=False)

        output_file = "./trace/segmented_traces.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(subtrace_res, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 处理完成！结果已保存至 {output_file}")

    # 执行异步主函数
    asyncio.run(main())
