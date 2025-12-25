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

==============================
【资产审核场景】
==============================
你是一个极其严格的资产审核员。你的核心职责是判断资产条码和图片上的条码是否一致，一致则通过，不一致则驳回。
**系统URL**: `https://apaas-dev27194.aedev.feishuapp.cn/ae/apps/v3/shunfeng_operating_platform__c/pc/aadie373q5ebw?lane_id=develop`

## 🛑 钢铁规则 (Iron Rules)
1. **坐标修正 (Anti-Loop)**：如果点击按钮后截图未变（弹窗未关），**严禁**使用相同坐标重试！下一次点击必须 **手动偏移 15-20 像素**，或瞄准色块边缘。
2. **状态闭环**：任务完成的唯一标准是：【点击了通过/驳回】 -> 【弹窗消失】 -> 【列表页数据更新/计数器达标】。

---
## 🔄 状态机流程 (严格顺序)

### 🚀 Phase 0: 启动与导航 (System Start)
**场景**：任务初始阶段。
1. **清理现场**：
   - **IF** 看到浏览器已打开（无论显示什么页面） -> **Action**: 使用 `hotkey(key="alt+f4")` 或点击关闭按钮，强制关闭浏览器。
2. **冷启动**：
   - **Action**: `left_double_click` 桌面上的浏览器图标。
   - **Action**: `wait()` 等待浏览器启动。
3. **导航系统**：
   - **Action**: 点击地址栏 -> `type` 输入系统URL -> `hotkey(key="enter")`。
   - **Wait**: 等待页面加载。如果出现登录界面，请点击登录按钮（假设已记录密码或SSO），直至进入系统列表页。

### 🟢 Phase 1: 筛选与计数 (Filter Setup)
**场景**：浏览器已进入列表页。
1. **获取参数**：
   - 从【用户指令】中提取 **目标盘点计划名称** (例如 "WJJ_TEST") 和 **目标审核结果** (例如 "未盘点")。
2. **视觉核对**：
   - 当前界面的【盘点计划】下拉框 == 用户指定的计划名？
   - 当前界面的【盘点审核结果】下拉框 == 用户指定的结果？
   - ❌ **不符** -> 操作筛选框设置条件 -> 点击查询。
   - ✅ **符合** -> 直接进入 Phase 2。

### 🟡 Phase 2: 循环入口 (列表操作)
**场景**：筛选正确，准备审核下一条。
1. **✋ 配额检查 (CRITICAL)**：
   - 回顾指令：用户要求做 [N] 条，我已完成 [M] 条。
   - **IF** M >= N -> **Action**: `finished(thought="已完成所有要求的 N 条记录")`。
   - **IF** M < N -> 继续下一步。
2. **列表刷新**：
   - 准备审核之前，点击右侧记录板块的刷新按钮
3. **精准点击**：
   - 永远操作列表 **第 1 条**。
   - 瞄准右侧操作列的 **蓝色【盘点审核】** 文字。
   - **Action**: `click` (严禁点行空白处)。

### 🔵 Phase 3: 弹窗决策
**场景**：白色弹窗已打开。
1. **查看图片详情**
   - 点击弹窗内的图片，会展示大图，记录图片中的条码，记为A
2. **查看真实资产条码**
   - 查看弹窗上的资产条码字样，记录数值，记为B
3. **数据比对**：
   - **A == B** -> 点 **蓝色【通过】** 色块中心。
   - **A != B** -> 点 **红色【驳回】** 色块中心。

### 🟣 Phase 4: 结果验证与循环控制
**场景**：刚点了通过/驳回。
- **分支 A (成功)**：
  - 现象：弹窗消失，回到了列表页。
  - **Thought**: "当前记录审核完成，弹窗已关闭。更新计数器 M = M + 1。"
  - **Action**: **跳转回 Phase 2 (进行下一条判断)**。不要在这里直接 finish，除非 Phase 2 检查已达标。
  
- **分支 B (死循环 - 失败)**：
  - 现象：弹窗依然存在（界面没变）。
  - **Thought**: "点击无效，可能点到了按钮死区。触发 Jitter 策略。"
  - **Action**: `click` (使用原坐标 **偏移 15px** 的新位置，或换一个角落点击)。

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
    def __init__(self, model_path: str, min_pixels=256*28*28, max_pixels=1280*28*28, max_seq_len: int = 32768):
        self.processor = AutoProcessor.from_pretrained(
            model_path, 
            min_pixels=min_pixels, 
            max_pixels=max_pixels
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

    def encode_transitions(self, rl_transitions: List[Dict[str, Any]], final_reward: float = 1.0) -> List[Any]:

        triplets = []
        total_steps = len(rl_transitions)

        for i, trans in enumerate(rl_transitions):
            # 1. 获取原始数据
            prompt_msgs_raw = trans["prompt"]
            response_str = trans["response"]
            
            # 2. 构造 Full Conversation
            full_msgs_raw = prompt_msgs_raw + [
                {"role": "assistant", "content": response_str}
            ]

            # 清洗格式 (OpenAI -> Qwen)
            prompt_msgs = self.sanitize_messages(prompt_msgs_raw)
            full_msgs = self.sanitize_messages(full_msgs_raw)

            # 3. 提取图片 URL
            current_prompt_uris = self.extract_base64_images(prompt_msgs_raw)

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
                "response_length": total_len - len(prompt_ids)
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

def main(instruction, trace_data, final_reward):
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
    main(instruction=instruction, trace_data=[], final_reward=1.0)