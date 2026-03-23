import os
import json
import base64
import mimetypes
import traceback
import asyncio
import agentlightning
from typing import Any, Dict, List, Union, Optional
from typing_extensions import Annotated
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv
from constants import (
    GROUPED_ACTION_REWARD_PROMPT,
    TRACE_SEGMENT_PROMPT,
    SUB_INSTRUCTION_DESCRIPTION,
    SEGMENT_START_STEP_DESCRIPTION,
    SEGMENT_START_STEP_LENGTH_DESCRIPTION,
    GROUPED_REWARD_DESCRIPTION,
)
from data.asset_audit.constants import AUDIT_EVALUATION_PROMPT
from data.browser_search.constants import BROWSER_EVALUATION_PROMPT
from data.file_organization.constants import FILE_EVALUATION_PROMPT

agentlightning.configure_logger()

logger = agentlightning.configure_logger(name=__name__)

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
    instruction: str = Field(description=SUB_INSTRUCTION_DESCRIPTION.strip())
    # 注意：这里要求 LLM 填的是你展示给它的 Step ID (1, 2, 3...)
    start_step: int = Field(
        ge=1,
        description=SEGMENT_START_STEP_DESCRIPTION.strip(),
    )
    # 静态校验直接写在这里，代替之前的 if current_len < min_seg_len
    step_length: int = Field(
        ge=2,
        le=15,
        description=SEGMENT_START_STEP_LENGTH_DESCRIPTION.strip(),
    )


class GroupRewardItem(BaseModel):
    reward: float


class OverallRewardItem(BaseModel):
    reward: Annotated[float, Field(ge=0, le=1.0)]
    reason: str


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


def _iter_events(trace: Any) -> List[Dict[str, Any]]:
    if isinstance(trace, list):
        return trace
    return []


# ===========================
# 4. reward 类
# ===========================
class CUARewardModel:
    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        model: str = "",
        segment_prompt: str = TRACE_SEGMENT_PROMPT,
        grouped_reward_prompt: str = GROUPED_ACTION_REWARD_PROMPT,
    ) -> None:
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.segment_prompt = segment_prompt
        self.group_reward_prompt = grouped_reward_prompt

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

    def _build_reward_content(self, trace: list, user_instruction: str, reward_prompt: str) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": f"System Prompt: {reward_prompt.strip()}"},
            {"type": "text", "text": f"User Instruction: {user_instruction.strip()}"},
        ]

        events = _iter_events(trace)

        has_processed_init = False

        for ev in events:
            if not isinstance(ev, dict):
                continue

            # 只要看到截图，就更新“最新状态”的物理索引
            if "screenshot" in ev:

                if not has_processed_init and "tool_calls" not in ev and "tool_outputs" not in ev:
                    content.append({"type": "text", "text": "**Initial State:**"})
                    content.extend(self._normalize_image_url(ev["screenshot"]))
                    has_processed_init = True
                elif has_processed_init and "tool_outputs" not in ev:
                    # 中间独立的截图
                    content.append({"type": "text", "text": "**Screen State Update (Observation):**"})
                    content.extend(self._normalize_image_url(ev["screenshot"]))
                continue

            # 遇到动作，绑定当前 Step 的寻址信息
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
                        "text": f"Agent Action\n**Function Call:**\n```json\n{json.dumps(formatted_tool_calls, ensure_ascii=False, indent=2)}\n```",
                    }
                )
                continue

            # 处理动作输出
            if "tool_outputs" in ev:
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
                        "text": f"**Tool Outputs:**\n```json\n{json.dumps(formatted_tool_output, ensure_ascii=False, indent=2)}\n```",
                    }
                )

        return content

    def _is_tool_call(self, message: Dict[str, Any]) -> bool:
        """
        判断当前 message 是否为 assistant 的 tool_call。
        """
        # OpenAI 格式通常为 tool_calls 字段
        if "tool_calls" in message and message["tool_calls"]:
            return True

        return False

    async def segment_trace(self, trace: list, user_instruction: str) -> Dict[str, Any]:
        max_retries = 3
        fixed_temperature = 0.0
        previous_feedback = None

        # 1. 在循环外部构建 Content 和 Mapping，极大节省 CPU 和内存
        base_content, step_map = self._build_trace_segment_content(trace, user_instruction)
        max_steps = len(step_map)  # LLM 能看到的最大的 Step 编号

        # 异常兜底：如果 trace 里面没有任何有效 step
        if max_steps == 0:
            logger.info("⚠️ 轨迹中未检测到任何有效的 Step。")
            return {
                "segments": [],
                "step_map": {},
            }

        for attempt in range(max_retries):
            try:
                messages = [{"role": "user", "content": base_content}]

                # 2. 如果有反馈，携带上下文重试
                if previous_feedback:
                    logger.info(f"🔧 [Attempt {attempt + 1}] 切割有误,正在重试...")
                    messages.append(
                        {
                            "role": "user",
                            "content": f"你的上一次输出存在逻辑错误，请根据以下反馈修正（注意总 Step 数量为 {max_steps}）：\n{previous_feedback}",
                        }
                    )

                logger.info(f"🚀 [Attempt {attempt + 1}/{max_retries}] 请求大模型进行切分...")
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
                    logger.info(f"✅ 成功获取有效且连续的分段: {len(segments)} 段")
                    # 这里返回的时候，把片段和映射表一起返回，方便后续组装！
                    return {"segments": [s.model_dump() for s in segments], "step_map": step_map}
                else:
                    previous_feedback = "\n".join(error_reasons)
                    logger.warning(f"⚠️ 业务校验失败:\n{previous_feedback}")

            except Exception as e:
                error_msg = str(e)
                logger.error(f"❌ 解析/系统错误: {error_msg}")
                # 如果是 Pydantic 抓到的格式错误 (如长度不足 3)，反馈给 LLM
                if "validation" in error_msg.lower():
                    previous_feedback = (
                        f"输出格式校验失败，请严格遵守长度(3-15)和总段数(1-7)的限制规则。\n细节: {error_msg}"
                    )
                else:
                    # 纯网络错误等，不干扰大模型上下文
                    traceback.print_exc()
                    previous_feedback = None

        logger.error("❌ 已达到最大重试次数，操作中止")
        return {
            "segments": [],
            "step_map": {},
        }

    def extract_subtraces(self, trace: list, segment_result: dict) -> list:
        subtraces = []
        step_map = segment_result["step_map"]
        count = 0
        for seg in segment_result["segments"]:
            start_s = seg["start_step"]
            end_s = start_s + seg["step_length"] - 1

            # 1. 抓取该片段最开头的状态截图索引
            step_info = step_map.get(start_s)
            if step_info is None:
                continue
            init_state_idx = step_info["state_screenshot_idx"]

            # 2. 抓取动作区间
            action_start = step_map[start_s]["action_idx"]
            # 结束区间的动作通常取下一个片段的开头，或者原 trace 的最后
            action_end = step_map.get(end_s + 1, {}).get("action_idx", len(trace))

            # 3. 组装：起始状态截图 + 中间所有的动作与输出
            sub_trace = [trace[init_state_idx]] + trace[action_start:action_end]

            subtraces.append({"segment_id": count, "instruction": seg["instruction"], "trace_data": sub_trace})
            count += 1

        return subtraces

    async def call_group_reward_model(self, trace: List, user_instruction: str) -> float:
        fixed_temperature = 0.0
        base_content = self._build_reward_content(trace, user_instruction, self.group_reward_prompt)
        messages = [{"role": "user", "content": base_content}]
        logger.info(f"🚀 正在评分....")
        resp = await asyncio.wait_for(
            self.client.beta.chat.completions.parse(
                model=self.model,
                temperature=fixed_temperature,
                messages=messages,
                response_format=GroupRewardItem,
            ),
            timeout=120,
        )
        return resp.choices[0].message.parsed.reward  # Return the parsed reward value

    async def call_overall_reward_model(self, trace: List, user_instruction: str, scene: str) -> tuple[float, str]:
        fixed_temperature = 0.0
        overall_reward_prompt = ""
        if scene == "asset_audit":
            overall_reward_prompt = AUDIT_EVALUATION_PROMPT
        elif scene == "browser_search":
            overall_reward_prompt = BROWSER_EVALUATION_PROMPT
        elif scene == "file_organization":
            overall_reward_prompt = FILE_EVALUATION_PROMPT
        else:
            raise ValueError(f"Unsupported scene: {scene}")

        base_content = self._build_reward_content(trace, user_instruction, overall_reward_prompt)
        messages = [{"role": "user", "content": base_content}]
        logger.info("🚀 调用整体奖励模型评分")
        resp = await asyncio.wait_for(
            self.client.beta.chat.completions.parse(
                model=self.model,
                temperature=fixed_temperature,
                messages=messages,
                response_format=OverallRewardItem,
            ),
            timeout=120,
        )
        logger.info(f"✅ 整体奖励模型评分成功: {resp.choices[0].message.parsed.reward}")
        return (
            resp.choices[0].message.parsed.reward,
            resp.choices[0].message.parsed.reason,
        )  # Return the parsed reward value and reason

    async def score_single_subtrace(self, subtrace: dict) -> Dict[str, Any]:
        trace_data = subtrace["trace_data"]
        user_instruction = subtrace["instruction"]
        segment_id = subtrace["segment_id"]
        try:
            reward = await asyncio.wait_for(
                self.call_group_reward_model(trace_data, user_instruction),
                timeout=120,
            )
            logger.info(f"✅ SubTrace_{segment_id}评分成功: {reward}")
            return {
                "segment_id": segment_id,
                "reward": reward,
            }
        except asyncio.TimeoutError as e:
            logger.error(f"❌ SubTrace_{segment_id}评分超时: {e}")
            return {
                "segment_id": segment_id,
                "reward": None,
                "error": f"timeout: {e}",
            }
        except Exception as e:
            logger.error(f"SubTrace_{segment_id}评分出错: {e}")
            return {
                "segment_id": segment_id,
                "reward": None,
                "error": str(e),
            }

    async def score_subtrace_list(
        self,
        subtrace_list: List[dict],
        max_concurrency: int = 8,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> List[Dict[str, Any]]:
        assert all("segment_id" in s for s in subtrace_list)
        logger.info(f"开始对 {len(subtrace_list)} 个子轨迹进行评分")
        for attempt in range(max_retries):
            semaphore = asyncio.Semaphore(max_concurrency)

            async def sem_task(subtrace):
                async with semaphore:
                    return await self.score_single_subtrace(subtrace)

            tasks = [asyncio.create_task(sem_task(subtrace)) for subtrace in subtrace_list]

            results = []
            for task in asyncio.as_completed(tasks):
                r = await task
                results.append(r)

            assert len(results) == len(subtrace_list)
            results.sort(key=lambda x: x["segment_id"])

            failed = [item for item in results if item.get("reward") is None]
            if not failed:
                logger.info(f"✅ 子轨迹评分成功！")
                return results

            logger.info(f"[Trace Retry] failed_segments={len(failed)} " f"(attempt {attempt + 1}/{max_retries})")

            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                await asyncio.sleep(delay)

        logger.error(f"[Trace FAILED] whole-trace scoring failed after {max_retries} attempts")
        logger.info("子轨迹评分失败")
        return results

    def build_context_trace(
        self, subtrace_list: List[Dict[str, Any]], reward_list: List[Optional[float]], global_instruction: str
    ) -> List[Dict[str, Any]]:
        """
        从 subtrace_list 构造 action-level triplets。

        每个 triplet:
            {
                "instruction": str,
                "trace_data": List[message],
                "reward": float | None
            }

        规则：
        - 每个 assistant.tool_call 为截断点（包含该条）
        - 不包含 tool response
        - reward 为 subtrace-level reward
        """

        if len(subtrace_list) != len(reward_list):
            raise ValueError("subtrace_list 与 reward_list 长度不一致")

        context_traces: List[Dict[str, Any]] = []

        for idx, subtrace in enumerate(subtrace_list):

            sub_instruction = subtrace.get("instruction", "")
            trace_data = subtrace.get("trace_data", [])

            if not isinstance(trace_data, list):
                raise ValueError(f"subtrace index {idx} 的 trace_data 不是 list")

            reward = reward_list[idx]

            # 合并 instruction
            merged_instruction = (
                "GlobalTask:\n" + global_instruction.strip() + "\nCurrentSubtask:\n" + sub_instruction.strip()
            )

            # 遍历 trace_data
            for msg_index, message in enumerate(trace_data):

                if self._is_tool_call(message):

                    # 截断至当前 tool_call（包含）
                    truncated_trace = trace_data[: msg_index + 1]

                    # 构造 triplet
                    triplet = {"instruction": merged_instruction, "trace_data": truncated_trace, "reward": reward}

                    context_traces.append(triplet)
        logger.info(f"✅ 成功构建 {len(context_traces)} 条上下文轨迹")

        return context_traces

    async def get_group_rewards(self, trace: list, user_instruction: str) -> List[Dict[str, Any]]:
        segment_result = await self.segment_trace(trace, user_instruction=user_instruction)
        subtrace_list = self.extract_subtraces(trace, segment_result)
        results = await self.score_subtrace_list(subtrace_list=subtrace_list)
        reward_list = [item["reward"] for item in results]
        context_traces = self.build_context_trace(subtrace_list, reward_list, user_instruction)

        # result_list_file = "./trace/group_reward_results.json"
        # with open(result_list_file, "w", encoding="utf-8") as f:
        #     json.dump(results, f, indent=2, ensure_ascii=False)

        # segment_res_file = "./trace/segment_policy.json"
        # with open(segment_res_file, "w", encoding="utf-8") as f:
        #     json.dump(segment_result, f, indent=2, ensure_ascii=False)

        # output_file = "./trace/segmented_traces.json"
        # with open(output_file, "w", encoding="utf-8") as f:
        #     json.dump(subtrace_list, f, indent=2, ensure_ascii=False)

        return context_traces

    async def get_overall_reward(self, trace: list, user_instruction: str, scene: str) -> tuple[float, str]:
        (reward, reason) = await self.call_overall_reward_model(trace, user_instruction, scene)
        return reward, reason


# ===========================
# 5. 测试：读取文件并运行
# ===========================
if __name__ == "__main__":
    # --- 配置 --
    # JSON 文件路径
    JSON_FILE_PATH = "/root/workspace/wangjiaju/zql_workspace/agent-lightning/examples/cua/trace/0206/0206-qwen3-4b-sft-2500/sample_14_plan_ZQL_TEST.json"

    # 加载 trace 数据
    user_instr, trace_data = load_trace_json(JSON_FILE_PATH)

    # 加载环境变量\n
    load_dotenv(find_dotenv())
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    model_name = os.getenv("REWARD_MODEL", "doubao-seed-1-6-251015")

    # 初始化打分器
    group_scorer = CUARewardModel(base_url=base_url, api_key=api_key, model=model_name)

    # 4. 运行异步函数（修复 await 问题）
    # 定义异步主函数
    async def main():
        # segment_res = await group_scorer.segment_trace(trace_data, user_instr)
        # subtrace_res = group_scorer.extract_subtraces(trace_data, segment_res)
        scored_subtraces = await group_scorer.get_group_rewards(trace_data, user_instr)
        # 5. 保存结果
        # segment_res_file = "./trace/segment_policy.json"
        # with open(segment_res_file, "w", encoding="utf-8") as f:
        #     json.dump(segment_res, f, indent=2, ensure_ascii=False)

        # output_file = "./trace/segmented_traces.json"
        # with open(output_file, "w", encoding="utf-8") as f:
        #     json.dump(subtrace_res, f, indent=2, ensure_ascii=False)

        scored_subtraces_file = "./trace/scored_subtraces_file.json"
        with open(scored_subtraces_file, "w", encoding="utf-8") as f:
            json.dump(scored_subtraces, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 处理完成！结果已保存至 {scored_subtraces_file}")

    # 执行异步主函数
    asyncio.run(main())
