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
        # 配置参数
        max_retries = 3
        target_max_segments = 7
        # 🔥 固定温度
        fixed_temperature = 0.7
        
        # 🔥 长度约束参数
        min_seg_len = 3
        max_seg_len = 15
        # 获取原始 trace 的最大索引边界
        max_trace_len = len(trace)

        # [新增] 用于存储上一次失败的反馈信息
        previous_feedback = None

        for attempt in range(max_retries):
            try:
                # 1. 构建基础 Prompt
                base_content = self._build_content(trace, user_instruction)
                
                # 2. 组装 Messages 列表
                messages = [{"role": "user", "content": base_content}]

                # [新增] 如果有上次的错误反馈，追加到对话历史中
                if previous_feedback:
                    print(f"🔧 [Retry Hint] 追加错误反馈给模型: {previous_feedback[:50]}...")
                    retry_prompt = (
                        f"你的上一次输出未通过校验，请根据以下具体错误原因进行修正：\n"
                        f"{previous_feedback}\n"
                        f"请务必严格遵守：\n"
                        f"1. 索引不要越界 (Max Index < {max_trace_len})。\n"
                        f"2. 每个片段长度必须在 {min_seg_len} 到 {max_seg_len} 之间。\n"
                        f"请重新生成符合要求的 JSON。"
                    )
                    messages.append({"role": "user", "content": retry_prompt})

                print(f"🚀 [Attempt {attempt + 1}/{max_retries}] 发送请求给模型 (temp={fixed_temperature})...")

                resp = await self.client.beta.chat.completions.parse(
                    model=self.model,
                    temperature=fixed_temperature,
                    messages=messages, # 使用包含反馈的 messages
                    response_format=RewardResultWrapper,
                )
                
                parsed_obj = resp.choices[0].message.parsed

                if parsed_obj and hasattr(parsed_obj, "segments"):
                    segments = parsed_obj.segments
                    seg_count = len(segments)
                    
                    error_reasons = []

                    # === 1. 数量检查 ===
                    if seg_count == 0:
                        error_reasons.append("错误：返回了 0 个片段，请至少切分出一个有效片段。")
                    elif seg_count > target_max_segments:
                        error_reasons.append(f"错误：片段数量 ({seg_count}) 超过了最大限制 {target_max_segments}。")

                    # === 2. 逐段质量检查 ===
                    if not error_reasons:
                        for i, seg in enumerate(segments):
                            # 注意：这里沿用你代码里的 length 逻辑
                            s_idx = seg.start_idx
                            current_len = seg.length
                            e_idx = s_idx + current_len # 计算结束位置用于越界检查
                            
                            # 2.1 越界检查
                            if s_idx >= max_trace_len or e_idx > max_trace_len: # 注意 e_idx 是切片末尾，可以是 max_len (如果切片是左闭右开)
                                error_reasons.append(f"- 第 {i+1} 个片段索引越界: Start={s_idx}, End={e_idx}, 最大允许索引={max_trace_len-1}")
                                break 
                            
                            # 2.2 长度检查
                            if current_len < min_seg_len:
                                error_reasons.append(f"- 第 {i+1} 个片段长度过短: 当前长度 {current_len}, 最小要求 {min_seg_len}")
                                break
                            
                            if current_len > max_seg_len:
                                error_reasons.append(f"- 第 {i+1} 个片段长度过长: 当前长度 {current_len}, 最大允许 {max_seg_len}")
                                break

                    # === 3. 决策环节 ===
                    if not error_reasons:
                        print(f"✅ 成功获取有效分段: {seg_count} 段")
                        return [item.model_dump() for item in segments]
                    else:
                        # 格式化错误信息
                        error_msg_str = "\n".join(error_reasons)
                        print(f"⚠️ 校验失败 (Attempt {attempt + 1}):\n{error_msg_str}")
                        
                        if attempt < max_retries - 1:
                            # [新增] 将错误信息存入 previous_feedback，供下一次循环使用
                            previous_feedback = error_msg_str
                            print("🔄 正在携带错误信息重试...")
                            continue
                        else:
                            print("❌ 已达到最大重试次数，且未能通过校验")
                            return []

                return []

            except Exception as e:
                print(f"❌ Error (Attempt {attempt + 1}): {e}")
                traceback.print_exc()
                
                # 即使发生 Exception，也设置一个通用的反馈信息，防止下次空转
                previous_feedback = f"上一次尝试发生了系统错误或解析错误: {str(e)}。请确保输出的是合法的 JSON 格式。"
                
                if attempt < max_retries - 1:
                    print("🔄 发生异常，正在重试...")
                    continue

        return []
    
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
                }
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
    JSON_FILE_PATH = "/root/code/wangjiaju/agent-lightning/examples/cua/trace/0116/rl/rollout-5e07da5a-d38f-4de0-8476-7e5d00ee1c0d_model_output.json"  # 请确保你的数据保存在这个文件里
    user_instr = "任务开始前，如果当前打开了浏览器，请先关闭所有浏览器窗口回到桌面。随后重新打开浏览器,进入资产审核网站，设置筛选条件，盘点计划RL_05，盘点审核结果为未盘点,然后盘点2条记录，盘点完成后关闭浏览器。"
    # 1. 读取文件
    if not os.path.exists(JSON_FILE_PATH):
        print(f"❌ 错误：找不到文件 {JSON_FILE_PATH}，请先创建该文件。")
        exit(1)

    print(f"📂 正在读取 {JSON_FILE_PATH} ...")
    with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
        trace_data = json.load(f)

    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("score_api_key")
    # 3. 初始化打分器
    group_scorer = GroupedActionsRewardModel(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=api_key,
        model="doubao-seed-1-6-251015"
    )

    # 4. 运行
    import asyncio
    result = asyncio.run(group_scorer.grouped_actions_reward(trace_data[1:], user_instr))

    # 5. 保存结果
    output_file = "reward_result.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result["grouped_traces"], f, indent=2, ensure_ascii=False)

    print(f"\n✅ 处理完成！结果已保存至 {output_file}")