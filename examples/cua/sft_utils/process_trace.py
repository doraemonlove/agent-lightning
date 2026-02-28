import json
import os
import base64
from io import BytesIO
from typing import Any
import hashlib
import glob
from tqdm import tqdm
from PIL import Image

DEFAULT_INSTRUCTION = "请根据当前屏幕截图执行资产盘点审核任务。"
SYSTEM_PROMPT = """你是**专业 GUI Agent**（macOS / Windows / Linux）。你的任务是根据用户指令、操作历史和截图，**逐步执行 GUI 操作**。
---

## 🧰 工具定义（Schema）
你拥有以下工具，请根据当前任务选择最合适的一个：

| 工具名 | 说明 | 参数要求 |
|--------|------|-----------|
| click | 左键单击 | {"thought": string, "x": int, "y": int} |
| left_double_click | 左键双击 | {"thought": string, "x": int, "y": int} |
| right_click | 右键单击 | {"thought": string, "x": int, "y": int} |
| drag | 拖拽 | {"thought": string, "start_x": int, "start_y": int, "end_x": int, "end_y": int} |
| type | 输入文字 | {"thought": string, "content": string} |
| hotkey | 按键 | {"thought": string, "key": string} |
| scroll | 滚动 | {"thought": string, "direction": "up/down", "x": int, "y": int} |
| wait | 等待 | {"thought": string} |
| finished | 任务完成 | {"thought": string} |
| call_user | 呼叫人工 | {"thought": string} |
| output | 输出结果 | {"thought": string, "content": string} |

---

## ⚙️ 注意事项
1. **Thought**：每个工具调用必须包含 `thought` 字段，详细描述你的观察和意图。
2. **坐标**：所有坐标必须是整数 (Integer)。
---

==============================
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
   - 当 [Counter] == [N] 时，必须立即执行 finished 工具。
   - 若筛选后显示“暂无数据”，执行 wait 或者 finished 工具。

4. **异常处理 (遮挡)**: 出现遮挡弹窗，优先执行关闭操作或使用 drag 移开。

# II. 核心操作流

## A. 环境清理与系统导航

- **状态 A.0 (环境清理)**: 任务开始，若浏览器窗口存在，先关闭浏览器回到桌面

- **状态 A.1**: 处于桌面状态。
  调用 left_double_click 工具打开浏览器

- **状态 A.2**: 浏览器地址栏不符。
  - **动作**：点击地址栏 -> 全选 -> 清空 -> 输入 URL (https://apaas-dev27194.aedev.feishuapp.cn/ae/apps/v3/shunfeng_operating_platform__c/pc/aadie373q5ebw?lane_id=develop) -> Enter。

- **状态 A.3**: 导航与筛选。
  - **动作**：进入“盘点详情”页 -> 筛选“盘点计划”为 **指定的盘点计划** -> 筛选“盘点审核结果”为 **未盘点**。
  - **关键**：确保下拉框选完后点击空白处关闭遮挡。

## B. 审核循环 (CORE)

- **状态 B.1**: 列表可见记录。
  点击第一条记录的盘点审核按钮

- **状态 B.2**: 弹窗核对。
  - **步骤 1**：点击图片放大。
  - **步骤 2**：**强制思维链**：Thought 中必须输出：“系统条码为[xxx], 图片可见内容为[yyy]”。
  - **判定**：完全一致标记为 [PASS_READY]，任何不符（含模糊、缺失）标记为 [REJECT_READY]。
  - **步骤 3**：然后关闭大图

- **状态 B.3**: 执行裁决。
  - 根据 B.2 的标记点击【通过】或【驳回】。

- **状态 B.4**: 确认完成。
  - **动作序列**：
    1. **计数更新**：弹窗消失后，Counter 增加 1。
    2. **逻辑跳转**：若 Counter < N，回到 **状态 B.0**；若 Counter == N，执行 **finished**。

# III. 限制
1. 不可连续 3 次 wait。
2. 禁止直接点击盘点记录除“盘点审核”按钮外的其他位置，必须点击“盘点审核”按钮。
3. 禁止重复两次调用一样的工具（参数也相同）
"""


def get_tools_schema():
    # 基础工具列表
    base_tools = [
        {
            "name": "click",
            "description": "左键单击，用于选中元素或点击按钮。",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string", "description": "思考过程"},
                    "x": {"type": "integer", "description": "X坐标"},
                    "y": {"type": "integer", "description": "Y坐标"},
                },
                "required": ["thought", "x", "y"],
            },
        },
        {
            "name": "left_double_click",
            "description": "左键双击，用于打开应用或文件。",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string"},
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["thought", "x", "y"],
            },
        },
        {
            "name": "right_click",
            "description": "右键单击，打开上下文菜单。",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string"},
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["thought", "x", "y"],
            },
        },
        {
            "name": "drag",
            "description": "拖拽，从起点拖到终点。",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string"},
                    "start_x": {"type": "integer"},
                    "start_y": {"type": "integer"},
                    "end_x": {"type": "integer"},
                    "end_y": {"type": "integer"},
                },
                "required": ["thought", "start_x", "start_y", "end_x", "end_y"],
            },
        },
        {
            "name": "type",
            "description": "输入文字（需确保焦点正确）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["thought", "content"],
            },
        },
        {
            "name": "hotkey",
            "description": "按单个键或组合键。",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string"},
                    "key": {"type": "string"},
                },
                "required": ["thought", "key"],
            },
        },
        {
            "name": "scroll",
            "description": "滚动操作。",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string"},
                    "direction": {"type": "string", "enum": ["up", "down"]},
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["thought", "direction", "x", "y"],
            },
        },
        {
            "name": "wait",
            "description": "等待界面稳定。",
            "parameters": {
                "type": "object",
                "properties": {"thought": {"type": "string"}},
                "required": ["thought"],
            },
        },
        {
            "name": "finished",
            "description": "标记任务完成。",
            "parameters": {
                "type": "object",
                "properties": {"thought": {"type": "string"}},
                "required": ["thought"],
            },
        },
        {
            "name": "call_user",
            "description": "呼叫用户人工接管。",
            "parameters": {
                "type": "object",
                "properties": {"thought": {"type": "string"}},
                "required": ["thought"],
            },
        },
        {
            "name": "output",
            "description": "输出信息或结果。",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["thought", "content"],
            },
        },
        {
            "name": "bad_function_tool",
            "description": "调用了不存在的函数或参数错误。",
            "parameters": {
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": ["content"],
            },
        },
    ]

    # 包装成 OpenAI 格式: [{"type": "function", "function": {...}}]
    openai_tools = []
    for tool in base_tools:
        openai_tools.append({"type": "function", "function": tool})

    return json.dumps(openai_tools, ensure_ascii=False)


def save_base64_image(base64_str, save_path):
    """辅助函数：保存base64图片并返回 (width, height)"""
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]

        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data))

        # 获取尺寸：image.size 是一个 (width, height) 的元组
        width, height = image.size

        # 确保目录存在
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        image.save(save_path)

        # 返回成功标志和尺寸
        return True, width, height
    except Exception as e:
        print(f"Error saving image: {e}")
        return False, None, None


def normalize_coordinates(
    args: dict[str, Any], width: int, height: int
) -> dict[str, Any]:
    """
    清洗坐标数据，将绝对坐标转换为 0-1000 的相对坐标，并返回更新后的字典。
    支持处理格式如 "587,225" 的字符串，取其首位数字。
    """
    # 浅拷贝原始字典，避免修改外部输入
    new_args = args.copy()

    def clean_val(val: Any) -> float:
        if val is None:
            return 0.0
        # 处理类似 "587,225" 的字符串，只取逗号前的部分
        cleaned_str = str(val).split(",")[0].strip()
        try:
            return float(cleaned_str)
        except ValueError:
            return 0.0

    # 1. 提取原始值并清洗
    raw_x = clean_val(args.get("x"))
    raw_y = clean_val(args.get("y"))

    # 2. 计算相对坐标 (归一化到 0-1000)
    norm_x = max(0, min(1000, int((raw_x / width) * 1000)))
    norm_y = max(0, min(1000, int((raw_y / height) * 1000)))

    # 3. 更新字典中的键值
    new_args["x"] = norm_x
    new_args["y"] = norm_y

    return new_args


def process_trace_file(
    input_json_file, image_save_dir, fallback_instruction=DEFAULT_INSTRUCTION
):
    with open(input_json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data or not isinstance(data, list):
        print(f"Skipping {input_json_file}: Invalid JSON format (not a list).")
        return None

    # =================================================
    # 1. 获取 Instruction (修改点)
    # 直接从列表第一个元素获取，如果不存在则使用 fallback
    # =================================================
    current_instruction = fallback_instruction
    if "instruction" in data[0] and data[0]["instruction"]:
        current_instruction = data[0]["instruction"]

    # =================================================
    # 2. 获取 Task ID
    # 遍历寻找 task_id 用于构建文件名
    # =================================================
    task_id = "unknown_task"
    for entry in data:
        if "task_id" in entry:
            task_id = entry["task_id"]
            break

    file_hash = hashlib.md5(input_json_file.encode("utf-8")).hexdigest()[:6]
    unique_task_id = f"{task_id}_{file_hash}"

    # 3. 初始化数据集样本结构
    dataset_sample = {
        "images": [],
        "tools": get_tools_schema(),
        "conversations": [],
    }

    # 添加 System Prompt
    dataset_sample["conversations"].append({"role": "system", "content": SYSTEM_PROMPT})

    # 4. 寻找初始截图 (First Screenshot)
    init_screenshot_entry = None
    first_screenshot_idx = -1

    for i, entry in enumerate(data):
        if "screenshot" in entry:
            init_screenshot_entry = entry["screenshot"]
            first_screenshot_idx = i
            break

    if not init_screenshot_entry:
        print(f"Skipping {input_json_file}: No screenshot found.")
        return None

    # === 处理第一张图 (User Initial Message) ===
    # 格式：User -> <image> + instruction
    img_count = 0
    img_filename = f"{task_id}_{img_count}.png"
    img_path = os.path.join(image_save_dir, img_filename)

    # 保存图片并获取尺寸
    is_img_save_success, width, height = save_base64_image(
        init_screenshot_entry, img_path
    )
    if is_img_save_success:
        dataset_sample["images"].append(img_path)
        img_count += 1

        dataset_sample["conversations"].append(
            {"role": "user", "content": f"<image>{current_instruction}"}
        )
    else:
        print(f"Error: Failed to save initial image for {task_id}")
        return None

    # 5. 遍历剩余事件，构建 User-function_call- observation循环
    # 从第一张截图之后开始遍历
    start_idx = first_screenshot_idx + 1

    skip_indices = set()
    for i in range(start_idx, len(data)):

        # 如果当前索引在跳过列表中，说明它已经被作为 tool 的一部分处理过了
        if i in skip_indices:
            continue

        entry = data[i]

        # Case A: 模型动作 (function_call)
        if "tool_calls" in entry:
            tool_calls = entry["tool_calls"]

            # 格式化 Tool Calls
            # 首先校正坐标
            args_dict = tool_calls[0]["function"]["arguments"]
            args_dict = normalize_coordinates(args_dict, width, height)
            name = tool_calls[0]["function"]["name"]
            formatted_tool_calls = {
                "name": name,
                "arguments": args_dict,
            }

            dataset_sample["conversations"].append(
                {
                    "role": "function_call",
                    "content": json.dumps(formatted_tool_calls, ensure_ascii=False),
                }
            )

        # === Case B: Tool Outputs (逻辑优化：向后吞并截图 + 尾部修剪) ===
        elif "tool_outputs" in entry:
            tool_outputs = entry["tool_outputs"]

            # 1. 预判下一条是否是截图
            next_idx = i + 1
            has_next_screenshot = (
                next_idx < len(data)
                and "screenshot" in data[next_idx]
                and "tool_calls" not in data[next_idx]
            )

            # =================================================================
            # Future Check (尾部修剪)
            # 检查：在当前 outputs (和潜在的 screenshot) 之后，是否还有 assistant 的 tool_calls？
            # 如果没有，说明这是数据的“烂尾”，必须全部丢弃，以保证对话以 assistant 结尾。
            # =================================================================
            has_future_action = False

            # 确定扫描起点：如果有截图，从截图后开始扫；没截图，从当前后开始扫
            scan_start_idx = next_idx + 1 if has_next_screenshot else next_idx

            for k in range(scan_start_idx, len(data)):
                if "tool_calls" in data[k]:
                    has_future_action = True
                    break

            if not has_future_action:
                # 如果未来没有动作了，直接丢弃当前的 Tool Output 和 下一张截图
                # 这样数据的最后一条就会停留在上一步的 Assistant Tool Call
                if has_next_screenshot:
                    skip_indices.add(
                        next_idx
                    )  # 标记截图已处理（这里是已丢弃），主循环不要再读它
                continue  # 跳过当前 Tool Output 处理
            # =================================================================

            # --- 2. 处理截图 (只有通过了 Future Check 才会执行到这里) ---
            img_tag = ""
            if has_next_screenshot:
                # 获取截图数据
                screenshot_data = data[next_idx]["screenshot"]

                # 保存图片
                next_img_filename = f"{unique_task_id}_{img_count}.png"
                next_img_path = os.path.join(image_save_dir, next_img_filename)

                is_img_save_success, _, _ = save_base64_image(
                    screenshot_data, next_img_path
                )
                if is_img_save_success:
                    dataset_sample["images"].append(next_img_path)
                    img_count += 1
                    img_tag = "<image>"

                # 标记下一条索引为跳过 (这次是已消费)
                skip_indices.add(next_idx)

            # --- 3. 构造 Tool Message ---
            tool_output = tool_outputs[0]
            tool_output_name = tool_output.get("name", "unknown")
            tool_output_content = json.loads(tool_output.get("content", ""))
            # 1. 检查 'result' 是否在字典中，且值是否为 None (JSON 中的 null)
            if (
                tool_output_content.get("result") is None
                and "result" in tool_output_content
            ):
                # 2. 移除 'result' 键值对
                tool_output_content.pop("result")

                # 3. 加入新的键值对 status: success
                tool_output_content["status"] = "success"
            # 插入image
            tool_output_content.update({"image": "<image>"})

            final_tool_output_content = json.dumps(
                tool_output_content, ensure_ascii=False
            )

            observation_content = {
                "tool_name": tool_output_name,
                "tool_output": final_tool_output_content,
            }
            observation_content_str = json.dumps(
                observation_content, ensure_ascii=False
            )
            dataset_sample["conversations"].append(
                {
                    "role": "observation",
                    "content": observation_content_str,
                }
            )

        # === Case C: 独立的 Screenshot ===
        # 正常情况下，截图应该被 Case B 吞并。
        # 如果代码走到这里，说明这是一张没有前置 tool_output 的截图（可能是错误的 Trace）。
        # 为了保证 User-Assistant-Tool 的严密队形，直接丢弃，不生成 User 消息。
        elif "screenshot" in entry:
            pass

    return dataset_sample


def main():
    # 1. 配置路径
    target_folders = [
        "/root/code/wangjiaju/zql_workspace/LLaMA-Factory/data/cua_data/cua_json_data/claude_0128_2573",
    ]
    # 假设这些是你外部定义的常量或函数
    image_save_dir = "/root/code/wangjiaju/zql_workspace/LLaMA-Factory/data/cua_data/images"  # 图片保存文件夹
    output_file = "/root/code/wangjiaju/zql_workspace/LLaMA-Factory/data/cua_data/cua_dataset.json"

    all_data = []
    all_files = []

    # 2. 第一步：先一次性把所有文件路径找出来
    print("正在扫描文件列表...")
    for folder in target_folders:
        # 递归查找该目录下所有 json
        found_files = glob.glob(os.path.join(folder, "**/*.json"), recursive=True)
        all_files.extend(found_files)

    print(f"共找到 {len(all_files)} 个文件，开始处理...")

    # 3. 第二步：使用 tqdm 包装文件列表，自动生成进度条
    # desc="处理进度" 是进度条左边的文字
    for file_path in tqdm(all_files, desc="处理进度", unit="file"):
        try:
            result = process_trace_file(file_path, image_save_dir)
            if result:
                all_data.append(result)
        except Exception:
            # 为了不打断进度条显示，建议不要在这里 print 错误
            # 或者只记录到日志
            pass

    # 4. 保存结果
    print(f"正在保存 {len(all_data)} 条数据...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"全部完成！文件已保存至: {output_file}")


def test():
    target_file = "/root/code/wangjiaju/agent-lightning/examples/cua/trace/0127/RL-garm-scroll/sample_1_ver_0.json"
    image_save_dir = (
        "/root/code/wangjiaju/llamafactory-0.9.4/data/cua/images"  # 图片保存文件夹
    )
    output_file = "/root/code/wangjiaju/llamafactory-0.9.4/data/cua/cua_dataset.json"

    dataset_sample = process_trace_file(
        target_file, image_save_dir, fallback_instruction=DEFAULT_INSTRUCTION
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([dataset_sample], f, ensure_ascii=False, indent=2)

    print(f"文件已保存至: {output_file}")


if __name__ == "__main__":
    main()
    # test()
