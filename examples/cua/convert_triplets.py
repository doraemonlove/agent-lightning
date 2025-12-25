import json
import copy
from typing import List, Dict, Any
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from agentlightning.types import Triplet

CUA_PROMPT = """你是**专业 GUI Agent**（macOS / Windows / Linux）。你的任务是根据用户指令、操作历史和截图，**逐步执行 GUI 操作**。

---

## 🎯 输出格式（严格要求）
每次输出必须仅包含以下两行，顺序固定：
Thought: <说明你将要做什么、为何这样做、如何定位目标或如何恢复错误>
Action: <工具名>(thought="<简要目的>", <参数键>=<值>, <参数键>=<值>, ...)
- **每次输出只能有一个 Action。**
- Action: 行必须只包含一次函数调用，不能换行或添加其他说明。
- 缺少Thought或Action都会被视为错误，缺少Thought往往会导致Bad Function Call错误。

---

## 🧰 工具定义（调用格式必须完全一致）

| 工具名 | 说明 | 调用格式 |
|--------|------|-----------|
| click | 左键单击，用于选中元素或点击按钮。 | Action: click(thought="", x=<int>, y=<int>) |
| left_double_click | 左键双击，用于打开应用或文件。 | Action: left_double_click(thought="", x=<int>, y=<int>) |
| right_click | 右键单击，打开上下文菜单。 | Action: right_click(thought="", x=<int>, y=<int>) |
| drag | 拖拽，从起点拖到终点。 | Action: drag(thought="", start_x=<int>, start_y=<int>, end_x=<int>, end_y=<int>) |
| type | 输入文字（需确保焦点正确）。 | Action: type(thought="", content="<字符串>") |
| hotkey | 按单个键或组合键。 | Action: hotkey(thought="", key="<键>") |
| scroll | 滚动操作。 | Action: scroll(thought="", direction="<up/down>", x=<int>, y=<int>) |
| wait | 等待界面稳定。 | Action: wait(thought="") |
| finished | 标记任务完成。 | Action: finished(thought="") |
| call_user | 呼叫用户人工接管。 | Action: call_user(thought="") |
| output | 输出信息或结果。 | Action: output(thought="", content="") |

---

## ⚙️ 参数规范（强制）
- 所有坐标必须写成 键=值 形式。  
- **禁止**省略参数名或顺序错误写法。  
- 坐标必须是整数，参数之间必须用逗号+空格分隔：, 。  
- 每个工具调用都必须包含 thought 参数。

---

===============================
【资产盘点场景 - 铁面审计版】
==============================
你是一个极其严格的资产盘点员。你的核心职责是寻找错误。
**红线规则：绝不能放过任何一个条码不匹配、图片缺失、或图片模糊的记录。**
默认假设所有记录都是错的，除非你能证明它们完全一致。

资产审核网站：https://apaas-dev27194.aedev.feishuapp.cn/ae/apps/v3/shunfeng_operating_platform__c/pc/aadie373q5ebw?lane_id=develop

# I. 任务变量与高优策略 (Guardrails)
1. **变量初始化**：
   - **目标审核条数 [N]**：从用户指令中提取（如“盘点1条”，则 N=1）。
   - **当前计数 [Counter]**：每成功完成 1 条（点击裁决且弹窗关闭），Counter 累加 1。
   - **进度记录**：每个 Step 的 Thought 必须包含：[当前进度: Counter/N]。

2. **环境预处理**：
   - **冷启动要求**：若任务开始时浏览器已打开，必须先执行关闭动作，确保从桌面重新开始。

3. **任务终止**: 
   - 当 [Counter] == [N] 时，必须立即执行 Action: finished(thought="[当前进度: N/N] 已完成指定数量")。
   - 若筛选后显示“暂无数据”，Action: finished(thought="页面无数据可供盘点")。

4. **异常处理 (遮挡)**: 出现遮挡弹窗，优先执行关闭操作或使用 drag 移开。

# II. 核心操作流 

## A. 环境清理与系统导航 

- **状态 A.0 (环境清理)**: 任务开始，若浏览器窗口存在。
  - Action: click(thought="[当前进度: 0/N] 遵照指令，先关闭浏览器回到桌面", x=<关闭按钮x>, y=<关闭按钮y>)

- **状态 A.1**: 处于桌面状态。
  - Action: left_double_click(thought="[当前进度: 0/N] 启动浏览器", x=<图标x>, y=<图标y>)

- **状态 A.2**: 浏览器地址栏不符。
  - **动作**：点击地址栏 -> 全选 -> 清空 -> 输入 URL (https://apaas-dev27194.aedev.feishuapp.cn/ae/apps/v3/shunfeng_operating_platform__c/pc/aadie373q5ebw?lane_id=develop) -> Enter。

- **状态 A.3**: 导航与筛选。
  - **动作**：进入“盘点详情”页 -> 筛选“盘点计划”为 **指定的盘点计划** -> 筛选“盘点审核结果”为 **未盘点**。
  - **关键**：确保下拉框选完后点击空白处关闭遮挡。

## B. 审核循环 (CORE)

- **状态 B.1**: 列表可见记录。
  - Action: click(thought="[当前进度: Counter/N] 点击第一条记录的盘点审核按钮", x=<审核按钮x>, y=<审核按钮y>)

- **状态 B.2**: 弹窗核对。
  - **步骤 1**：点击图片放大。
  - **步骤 2**：**强制思维链**：Thought 中必须输出：“系统条码为[xxx], 图片可见内容为[yyy]”。
  - **判定**：完全一致标记为 [PASS_READY]，任何不符（含模糊、缺失）标记为 [REJECT_READY]。
  - **步骤 3**：Action: click(thought="关闭大图", x=<关闭大图按钮x>, y=<关闭大图按钮y>)

- **状态 B.3**: 执行裁决。
  - 根据 B.2 的标记点击【通过】或【驳回】。

- **状态 B.4**: 确认完成。
  - **动作序列**：
    1. Action: wait(thought="等待弹窗消失")
    2. **计数更新**：弹窗消失后，Counter 增加 1。
    3. **逻辑跳转**：若 Counter < N，回到 **状态 B.0**；若 Counter == N，执行 **finished**。

# III. 限制
1. 不可连续 3 次 wait。
2. 禁止直接点击列表行，必须点击“盘点审核”按钮。
"""

class RL_Rollout_Adapter:
    def __init__(self, system_prompt: str, instruction: str):
        self.system_prompt = system_prompt
        self.instruction = instruction

    def adapt(self, trace_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        将轨迹转换为 RL 的 (State, Action) 序列。
        State = 累积的对话历史。
        """
        rl_transitions = []
        
        # 1. 初始化基础 Context (System + Instruction)
        # 这是一个不断增长的列表，代表当前的对话历史 (State)
        current_history_messages = [
            {
                "role": "system", 
                "content": self.system_prompt
            },
            # Instruction 通常作为 System 的补充，或者放在第一个 User 消息里
            # 这里我们简单起见，把它放在 System 后，或者作为 User 的第一句话
        ]
        
        # 标记：是否是第一张图（用于拼接 Instruction）
        is_first_turn = True
        
        for item in trace_data:
            
            # --- 状态更新 (Observation) ---
            if "screenshot" in item:
                # 构造 User Message
                content_list = []
                if is_first_turn:
                    # 第一轮，要把指令带上
                    content_list.append({"type": "text", "text": f"Instruction: {self.instruction}"})
                    is_first_turn = False
                
                content_list.append({"type": "image_url", "image_url": {"url": item["screenshot"]}})
                content_list.append({"type": "text", "text": "请执行下一步操作。"})
                # 将 observation 加入历史
                current_history_messages.append({
                    "role": "user",
                    "content": content_list
                })
            
            # --- 动作生成 (Action) ---
            elif "action" in item:
                if item["action"] == "start": continue

                if item.get("raw_text"):
                    full_response = item.get("raw_text")
                else:
                    full_response = f"Thought: {item.get("summary", "")}\nAction: {item.get("action", "")}"
                assistant_msg = {
                    "role": "assistant",
                    "content": full_response
                }
                
                # === 关键点：构建一个 RL Transition ===
                # Prompt (State) = 当前历史的一个深拷贝
                # Response (Action) = 当前的助手回复
                transition = {
                    # PPO Actor 需要看到的输入
                    "prompt": copy.deepcopy(current_history_messages), 
                    
                    # PPO Actor 需要生成的输出 (Label)
                    "response": full_response,
                    
                    # 元数据 (用于计算 Reward)
                    "meta": {
                        "task_id": item.get("task_id"),
                        "step_idx": len(rl_transitions)
                    }
                }
                rl_transitions.append(transition)
                
                # === 关键点：Action 也是历史的一部分 ===
                # 在进入下一步之前，把自己的输出加到历史里
                current_history_messages.append(assistant_msg)

        return rl_transitions

class RL_Tokenizer_Adapter:
    def __init__(self, model_path: str, max_seq_len: int = 32768):
        self.processor = AutoProcessor.from_pretrained(
            model_path, 
        )
        self.max_seq_len = max_seq_len

    def extract_base64_images(self, messages: List[Dict]) -> List[str]:
        """提取图片用于存储 Triplet"""
        images = []
        for msg in messages:
            if isinstance(msg.get("content"), list):
                for item in msg["content"]:
                    # 兼容两种格式提取 URL
                    if item.get("type") == "image_url" and isinstance(item.get("image_url"), dict):
                        images.append(item["image_url"]["url"])
                    elif item.get("type") == "image":
                        images.append(item["image"])
        return images

    def sanitize_messages(self, messages: List[Dict]) -> List[Dict]:
        """
        [修复补丁]
        将 OpenAI 风格的 {"image_url": {"url": "..."}} 
        转换为 Qwen 风格的 {"image": "..."}
        以避免 qwen_vl_utils 报错 'dict object has no attribute startswith'
        """
        new_msgs = []
        for msg in messages:
            new_msg = msg.copy()
            if isinstance(msg.get("content"), list):
                new_content = []
                for item in msg["content"]:
                    # 检测是否是嵌套字典格式
                    if item.get("type") == "image_url" and isinstance(item.get("image_url"), dict):
                        url = item["image_url"]["url"]
                        # 扁平化为 Qwen 偏好的格式
                        new_content.append({"type": "image", "image": url})
                    else:
                        new_content.append(item)
                new_msg["content"] = new_content
            new_msgs.append(new_msg)
        return new_msgs

    def limit_image_count(self, messages: List[Dict], max_image_steps: int) -> List[Dict]:
        """
        只保留最近的 max_image_steps 张图片，多余的图片将其从 content 中移除，但保留文本。
        """
        new_msgs = copy.deepcopy(messages)
        img_count = 0
        
        # 从后往前遍历消息，这样可以轻松找到“最新”的图片
        for msg in reversed(new_msgs):
            if isinstance(msg.get("content"), list):
                # 同样从后往前处理 content 列表
                for item in reversed(msg["content"]):
                    if item.get("type") in ["image", "image_url"]:
                        img_count += 1
                        # 如果超过配额，则移除该图片项
                        if img_count > max_image_steps:
                            # 标记为移除（在列表循环中安全删除的方法是设为 None 稍后过滤，
                            # 或者直接修改原 list，这里我们采取置空的策略）
                            item.clear() 
                
                # 过滤掉被 clear() 掉的空字典
                msg["content"] = [item for item in msg["content"] if item]
                
        return new_msgs
    
    def smart_truncate(self, messages: List[Dict], max_history_steps: int = 12) -> List[Dict]:
        """
        [新增功能] 智能截断策略：保头保尾 (Keep Head + Keep Tail)
        Args:
            messages: 完整的历史消息列表
            max_history_steps: 保留的最大对话轮数 (1轮 = User+Assistant)
                               建议 12 步 (配合 16k context)
        """
        # 1. 分离 System Message
        system_msgs = [m for m in messages if m["role"] == "system"]
        dialogue_msgs = [m for m in messages if m["role"] != "system"]

        # 如果对话历史很短，不需要截断
        # 假设每步交互产生 2 个 msg (User + Assistant)
        if len(dialogue_msgs) <= max_history_steps * 2:
            return messages

        # 2. 提取关键部分
        # Head: 保留前 2 个消息 (通常是 User:Instruction+Image 和 Assistant:FirstAction)
        # 这样能保证模型永远记得任务目标
        head_msgs = dialogue_msgs[:2]

        # Tail: 保留最近的 N 个消息
        # 我们需要保留 (max_history_steps - 1) 轮，因为 Head 占用了一轮
        keep_tail_count = (max_history_steps - 1) * 2
        
        # 安全检查：防止切片越界（虽然上面的 if 已经挡住了）
        if keep_tail_count > 0:
            tail_msgs = dialogue_msgs[-keep_tail_count:]
        else:
            tail_msgs = []

        # 3. 拼接：System + Head + ... (Middle Lost) ... + Tail
        truncated_messages = system_msgs + head_msgs + tail_msgs
        
        return truncated_messages

    def encode_transitions(self, rl_transitions: List[Dict[str, Any]], final_reward: float = 1.0) -> List[Any]:

        triplets = []
        rl_transitions = rl_transitions[-1:]
        total_steps = len(rl_transitions)
        
        # [配置] 最大保留历史步数
        # 12步 * 1.2k/步 ≈ 14.4k token，加上 System Prompt 安全在 16k 以内
        MAX_HISTORY_STEPS = 60
        MAX_IMAGE_STEPS = 2

        for i, trans in enumerate(rl_transitions):
            # 1. 获取原始数据
            prompt_msgs_raw = trans["prompt"]
            response_str = trans["response"]
            
            # === [新增关键步骤] 智能截断 Prompt ===
            # 在构造 Full Conversation 之前，先对 Prompt 进行截断
            prompt_msgs_truncated = self.smart_truncate(prompt_msgs_raw, max_history_steps=MAX_HISTORY_STEPS)
            prompt_msgs_truncated = self.limit_image_count(prompt_msgs_truncated, max_image_steps=MAX_IMAGE_STEPS)

            # with open(f"trans_{i}.json", "w", encoding="utf-8") as f:
            #     json.dump(prompt_msgs_truncated, f, ensure_ascii=False, indent=2)
            # 2. 构造 Full Conversation (基于截断后的 Prompt)
            full_msgs_truncated_raw = prompt_msgs_truncated + [
                {"role": "assistant", "content": response_str}
            ]

            # 清洗格式 (OpenAI -> Qwen)
            # 注意：使用的是截断后的 msg
            prompt_msgs = self.sanitize_messages(prompt_msgs_truncated)
            full_msgs = self.sanitize_messages(full_msgs_truncated_raw)

            # 3. 提取图片 URL
            # 注意：必须从截断后的 msg 提取，否则 triplet 里的 image_urls 会比 token 多，导致对不齐
            current_prompt_uris = self.extract_base64_images(prompt_msgs_truncated)

            # 4. 预处理 Vision Info
            try:
                # --- A. 处理 Prompt 部分 ---
                prompt_text = self.processor.apply_chat_template(
                    prompt_msgs, tokenize=False, add_generation_prompt=True
                )
                prompt_image_inputs, prompt_video_inputs = process_vision_info(prompt_msgs)
                
                # [修复点 1] 添加 return_tensors="pt"
                prompt_inputs = self.processor(
                    text=[prompt_text],
                    images=prompt_image_inputs,
                    videos=prompt_video_inputs,
                    padding=False,
                    return_tensors="pt"  # <--- 必须加这个
                )
                # 现在它是一个 Tensor，可以调用 tolist()
                prompt_ids = prompt_inputs.input_ids[0].tolist()

                # --- B. 处理 Full 部分 ---
                full_text = self.processor.apply_chat_template(
                    full_msgs, tokenize=False, add_generation_prompt=False
                )
                full_image_inputs, full_video_inputs = process_vision_info(full_msgs)
                
                # [修复点 2] 添加 return_tensors="pt"
                full_inputs = self.processor(
                    text=[full_text],
                    images=full_image_inputs,
                    videos=full_video_inputs,
                    padding=False,
                    return_tensors="pt"  # <--- 必须加这个
                )
                full_ids = full_inputs.input_ids[0].tolist()
                
            except Exception as e:
                # 打印详细堆栈以便调试其他问题
                import traceback
                traceback.print_exc()
                print(f"Error processing vision info at step {i}: {e}")
                continue

            # 5. 计算 Response Token IDs
            if len(full_ids) < len(prompt_ids):
                print(f"Warning: Step {i} truncated. Prompt: {len(prompt_ids)}, Full: {len(full_ids)}")
                continue
                
            response_ids = full_ids[len(prompt_ids):]

            # 统计长度
            total_len = len(full_ids)
            if total_len > self.max_seq_len:
                print(f"⚠️ Step {i} len {total_len} exceeds max {self.max_seq_len}")

            # 6. 构造 Triplet
            step_reward = final_reward if (i == total_steps - 1) else 0.0
            
            meta_data = trans.get("meta", {})
            meta_data.update({
                "total_tokens": total_len,
                "prompt_length": len(prompt_ids),
                "response_length": total_len - len(prompt_ids),
                "is_truncated": (len(prompt_msgs_raw) > len(prompt_msgs_truncated)) # 记录是否发生了截断
            })
            print(f"step {i}: {meta_data}")
            triplet = Triplet(
                prompt={
                    "token_ids": prompt_ids,
                    "image_urls": current_prompt_uris 
                },
                response={
                    "token_ids": response_ids
                },
                reward=step_reward,
                metadata=meta_data
            )
            triplets.append(triplet)

        return triplets
    
class TripletEncoder(json.JSONEncoder):
    """自定义 JSON 编码器，处理 Triplet 对象"""
    def default(self, obj):
        # 如果是 Triplet (Pydantic 对象)，转为字典
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        if hasattr(obj, 'dict'):
            return obj.dict()
        return super().default(obj)

def save_triplets_to_json(triplets: List[Any], filename: str):
    with open(filename, 'w', encoding='utf-8') as f:
        # indent=2 让文件可读性更好，但体积会变大
        json.dump(triplets, f, cls=TripletEncoder, indent=2, ensure_ascii=False)
    
    print(f"Saved to {filename}")

def convert_trace_to_triplets(instruction, trace_data, final_reward):
    converter = RL_Rollout_Adapter(
        system_prompt=CUA_PROMPT,
        instruction=instruction
    )

    rl_transitions = converter.adapt(trace_data)

    model_path = "/models/Qwen3-VL-8B-Instruct"
    adapter = RL_Tokenizer_Adapter(model_path)

    tokenized_data = adapter.encode_transitions(rl_transitions, final_reward=final_reward)
    return tokenized_data

if __name__ == "__main__":
    instruction = "设置筛选条件，盘点计划WJJ_TEST，盘点审核结果为未盘点。然后审核1条记录。"
    with open("sample.json", "r") as f:
        trace_data = json.load(f)
    tokenizer_data = convert_trace_to_triplets(instruction=instruction, trace_data=trace_data, final_reward=1.0)
    save_triplets_to_json(tokenizer_data, "triplets.json")