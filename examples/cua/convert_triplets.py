import json
import copy
from PIL import Image
from io import BytesIO
import base64
import re
import random
from typing import List, Dict, Any, Tuple
from transformers import AutoProcessor
from agentlightning.types import Triplet
from constants import CUA_PROMPT
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import agentlightning
from qwen_vl_utils import process_vision_info
from reward_server import score_trace

agentlightning.configure_logger()

logger = agentlightning.configure_logger(name=__name__)


def base64_to_pil(image_data):
    """将 base64 字符串或 URL 格式转为 PIL Image"""
    if isinstance(image_data, Image.Image):
        return image_data

    if isinstance(image_data, str):
        # 去掉 data:image/png;base64, 前缀
        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]
        try:
            image_bytes = base64.b64decode(image_data)
            return Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            logger.info(f"Image decode error: {e}")
            return None
    return None


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
                "properties": {"thought": {"type": "string"}, "x": {"type": "integer"}, "y": {"type": "integer"}},
                "required": ["thought", "x", "y"],
            },
        },
        {
            "name": "right_click",
            "description": "右键单击，打开上下文菜单。",
            "parameters": {
                "type": "object",
                "properties": {"thought": {"type": "string"}, "x": {"type": "integer"}, "y": {"type": "integer"}},
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
                "properties": {"thought": {"type": "string"}, "content": {"type": "string"}},
                "required": ["thought", "content"],
            },
        },
        {
            "name": "hotkey",
            "description": "按单个键或组合键。",
            "parameters": {
                "type": "object",
                "properties": {"thought": {"type": "string"}, "key": {"type": "string"}},
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
            "parameters": {"type": "object", "properties": {"thought": {"type": "string"}}, "required": ["thought"]},
        },
        {
            "name": "finished",
            "description": "标记任务完成。",
            "parameters": {"type": "object", "properties": {"thought": {"type": "string"}}, "required": ["thought"]},
        },
        {
            "name": "output",
            "description": "输出信息或结果。",
            "parameters": {
                "type": "object",
                "properties": {"thought": {"type": "string"}, "content": {"type": "string"}},
                "required": ["thought", "content"],
            },
        },
        {
            "name": "bad_function_call",
            "description": "调用了不存在的函数或参数错误。",
            "parameters": {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]},
        },
    ]

    # 包装成 OpenAI 格式: [{"type": "function", "function": {...}}]
    openai_tools = []
    for tool in base_tools:
        openai_tools.append({"type": "function", "function": tool})

    # return json.dumps(openai_tools, ensure_ascii=False)
    return openai_tools
    # return base_tools


def parse_action_to_json(action_str, summary_text):
    """
    极简版解析器。
    假设 action_str 格式严格为 Name(key=value, ...) 且参数名已与 Schema 一致。
    """
    if not action_str:
        return None, None

    # 1. 动作名称映射表 (Agent 输出名 -> 标准 Schema 名)
    # 只需要保留名字有变化的即可，大小写差异可以通过 .lower() 处理
    action_map = {
        "Double_Click": "left_double_click",
        "Right_Click": "right_click",
        "User_Takeover": "call_user",
        "Save_Memory": "save_long_term_memory",
        "Search_Knowledgebase": "search_knowledgebase",
        # Login 相关如果 schema 里有定义也需要加，没有则忽略
        "Login": "login",
        "Login_Record": "login_record",
    }

    # 2. 分离函数名和参数部分
    match = re.match(r"([a-zA-Z_]+)\((.*)\)", action_str)

    # 特殊情况处理
    if not match:
        # 如果是 finished 且没有括号（虽然现在 Agent 代码加了 thought 应该会有括号）
        # 但为了防止极端情况，保留这个判断
        if "finished" in action_str.lower():
            return "finished", {"thought": summary_text or "Task finished"}
        return None, None

    raw_name, args_str = match.groups()
    raw_name = raw_name.strip()

    # 获取标准工具名 (查表，查不到则全小写)
    tool_name = action_map.get(raw_name, raw_name.lower())

    # 3. 解析参数字符串
    arguments = {}

    if args_str:
        # 依然使用正则，为了正确处理 thought="..., ..." 这种情况
        pattern = re.compile(r'\s*([a-zA-Z_]\w*)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^,]*))')

        for m in pattern.finditer(args_str):
            key = m.group(1)
            # value 可能是 Group 2(双引号), 3(单引号), 或 4(无引号)
            val = m.group(2) if m.group(2) is not None else (m.group(3) if m.group(3) is not None else m.group(4))

            if val is not None:
                val = val.strip()
                # 类型推断
                if val.isdigit():
                    arguments[key] = int(val)
                elif val.lower() == "true":
                    arguments[key] = True
                elif val.lower() == "false":
                    arguments[key] = False
                else:
                    arguments[key] = val

    return tool_name, arguments


def normalize_coordinates(args: dict[str, Any], width: int, height: int) -> dict[str, Any]:
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


def convert_trace_to_messages(trace, instruction):
    """
    处理标准化的 GUI Agent 轨迹数据。
    假设 data 结构严格如下:
    [
      1: {"screenshot": "base64...", "task_id": ...},  <-- Initial State
      2: {"tool_calls": [...], "summary": "..."},      <-- Turn 1 Action
      3: {"tool_outputs": [...], "screenshot": "..."}, <-- Turn 1 Response
      ...

    ]
    """

    dataset_sample = {"tools": get_tools_schema(), "messages": [],"images": []}

    if not isinstance(trace, list) or len(trace) == 0:
        logger.error("Error: trace 必须是非空列表")
        return {}

    # =============================
    # 1. 处理初始状态 (Step 0)
    # =============================
    init_event = trace[0]
    if "screenshot" not in init_event or not init_event["screenshot"]:
        logger.error("Error: Missing initial screenshot")
        return {}

    init_screenshot = init_event["screenshot"]

    # 解析图片尺寸 (常用于 System Prompt 注入分辨率信息，或者单纯校验图片有效性)
    try:
        pil_img = base64_to_pil(init_screenshot)
        width, height = pil_img.size
    except Exception as e:
        logger.info(f"Error processing initial image: {e}")
        return {}

    dataset_sample["images"].append(init_screenshot)

    # 构造 System Prompt
    system_content = CUA_PROMPT

    dataset_sample["messages"].append({"role": "system", "content": {"type": "text", "text": system_content}})

    # 构造 User Initial Message
    dataset_sample["messages"].append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": init_screenshot},
            ],
        }
    )

    # =============================
    # 2. 遍历后续交互 (Step 1 -> N)
    # =============================
    # 从 events[1] 开始遍历 (跳过初始截图)
    skip_indices = set()
    for i, event in enumerate(trace[1:], start=1):

        if i in skip_indices:
            continue

        # --- Case A: 模型动作 (Assistant) ---
        if "tool_calls" in event:
            tool_calls = event["tool_calls"]
            args = tool_calls[0]["function"]["arguments"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            formatted_args = normalize_coordinates(args, width, height)
            tool_name = tool_calls[0]["function"]["name"]

            formatted_tool_calls = {
                "name": tool_name,
                "arguments": formatted_args,
            }
            # 1. 思考过程 (Thought)
            dataset_sample["messages"].append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": json.dumps(formatted_tool_calls, ensure_ascii=False)}],
                }
            )

        # --- Case B: 环境反馈 (Tool/User) ---
        elif "tool_outputs" in event:
            tool_outputs = event["tool_outputs"]
            screenshot_data = None

            # 1. 预判下一条是否是截图
            next_idx = i + 1
            has_next_screenshot = (
                next_idx < len(trace) and "screenshot" in trace[next_idx] and "tool_calls" not in trace[next_idx]
            )

            # 2. 获取截图数据
            if has_next_screenshot:
                # 获取截图数据
                screenshot_data = trace[next_idx]["screenshot"]
                dataset_sample["images"].append(screenshot_data)

                # 标记下一条索引为跳过 (这次是已消费)
                skip_indices.add(next_idx)

            # 3. 构造 Tool Message
            tool_output = tool_outputs[0]
            tool_output_name = tool_output.get("name", "unknown")
            tool_output_content = json.loads(tool_output.get("content", ""))
            # 1. 检查 'result' 是否在字典中，且值是否为 None (JSON 中的 null)
            if tool_output_content.get("result") is None and "result" in tool_output_content:
                # 2. 移除 'result' 键值对
                tool_output_content.pop("result")

                # 3. 加入新的键值对 status: success
                tool_output_content["status"] = "success"

            formatted_tool_output_content = json.dumps(tool_output_content, ensure_ascii=False)

            observation_content = {
                "tool_name": tool_output_name,
                "tool_output": formatted_tool_output_content,
            }
            observation_content_str = json.dumps(observation_content, ensure_ascii=False)
            dataset_sample["messages"].append(
                {
                    "role": "observation",
                    "content": [
                        {"type": "text", "text": observation_content_str},
                        {"type": "image", "image": screenshot_data} if has_next_screenshot else {},
                    ],
                }
            )

        # === Case C: 独立的 Screenshot ===
        # 正常情况下，截图应该被 Case B 吞并。
        # 如果代码走到这里，说明这是一张没有前置 tool_output 的截图（可能是错误的 Trace）。
        # 为了保证 User-Assistant-Tool 的严密队形，直接丢弃，不生成 User 消息。
        elif "screenshot" in event:
            pass

    # =============================
    # 3. 只保留到最后一个 Assistant
    # =============================
    msgs = dataset_sample["messages"]

    # 从后往前检查，只要最后一条不是 assistant，就移除
    # 使用 while 循环是因为可能结尾连续跟着 [tool_output, screenshot, user_msg] 等多条非 assistant 消息
    while len(msgs) > 0 and msgs[-1]["role"] != "assistant":
        # 移除最后一条
        removed_msg = msgs.pop()

    # 安全检查：如果截断后 mssages 只剩下 system 或者 user (Step 0)，
    # 说明整个 trace 没有有效的 assistant 动作，这条数据通常没有训练价值。
    # 至少应该保留 [System, User, Assistant] 三条
    if len(msgs) < 3:
        logger.warning("Warning: Trace dropped because no valid assistant action found at the end.")
        logger.info(f"msgs: {msgs}")
        return {}

    return dataset_sample


def convert_messages_to_triplet(
    dataset_sample, processor, reward: float, overall_score: float, rollout_id: str, max_seq_len=16384
):
    """
    使用 "分别 Tokenize (Prompt vs Full) 再相减" 的方式生成 Triplet。
    逻辑更加清晰，无需硬编码 Assistant Header Token ID。
    """

    messages = dataset_sample.get("messages", [])

    # 1. 基础检查
    if len(messages) < 3:
        raise Exception("messages too short, need at least user query and assistant response.")

    # 2. 准备 Tools
    tools = dataset_sample.get("tools", [])

    # 3. 拆分 Prompt Messages 和 Full Messages
    # Full: 包含所有对话
    # Prompt: 包含除最后一条 Assistant 回复之外的所有对话
    full_msgs = messages
    prompt_msgs = messages[:-1]

    original_image_list = dataset_sample.get("images", [])

    try:
        #  A. 处理 Prompt 部分
        # Key: add_generation_prompt=True 会自动添加 assistant\n
        prompt_text = processor.apply_chat_template(
            prompt_msgs, tools=tools, tokenize=False, add_generation_prompt=True
        )

        # 提取 Prompt 阶段包含的图像/视频输入
        prompt_image_inputs, prompt_video_inputs = process_vision_info(prompt_msgs)

        prompt_inputs = processor(
            text=[prompt_text],
            images=prompt_image_inputs,
            videos=prompt_video_inputs,
            padding=False,
            return_tensors="pt",  # 必须返回 Tensor 才能取 input_ids
        )
        prompt_ids = prompt_inputs.input_ids[0].tolist()

        # B. 处理 Full 部分
        # Key: add_generation_prompt=False
        full_text = processor.apply_chat_template(full_msgs, tools=tools, tokenize=False, add_generation_prompt=False)

        # 提取 Full 阶段包含的图像/视频输入
        full_image_inputs, full_video_inputs = process_vision_info(full_msgs)

        full_inputs = processor(
            text=[full_text], images=full_image_inputs, videos=full_video_inputs, padding=False, return_tensors="pt"
        )
        full_ids = full_inputs.input_ids[0].tolist()

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise e

    # ✂️ C. 计算 Response Token IDs (切片逻辑)
    # 安全检查：Full 应该比 Prompt 长
    if len(full_ids) <= len(prompt_ids):
        # 这种情况通常意味着 response 为空，或者 tokenizer 处理异常
        logger.warning(f"⚠️ Warning: Full length ({len(full_ids)}) <= Prompt length ({len(prompt_ids)}). Skipping.")
        # 根据你的训练框架需求，这里可以选择抛出异常或返回 None
        raise Exception("Response is empty or prompt matches full length.")

    # (可选) 严格的一致性检查：确保 Full 的前半部分就是 Prompt
    # 在 Qwen-VL 中，由于特殊 Token 的存在，通常是匹配的。
    # 如果发现不匹配，通常是 add_generation_prompt 添加的 \n 和 Full 中的 \n 合并问题
    # 这里不做硬性 assert，防止因为极个别 token 归一化导致训练中断，但建议日志关注
    # if full_ids[:len(prompt_ids)] != prompt_ids:
    #     logger.warning("⚠️ Warning: Token mismatch at boundary. Slicing anyway.")

    response_ids = full_ids[len(prompt_ids) :]

    # D. 长度检查与截断 (只截断 Response 部分)
    total_len = len(full_ids)
    is_truncated = False

    if total_len > max_seq_len:
        is_truncated = True
        # 计算允许的 response 长度
        allowed_resp_len = max_seq_len - len(prompt_ids)
        if allowed_resp_len > 0:
            response_ids = response_ids[:allowed_resp_len]
        else:
            # Prompt 已经超长了，Response 没地儿放了 , 这里可以选择保留一部分 Prompt 或直接丢弃
            response_ids = []  # 或者抛异常

    # E. 构建输出
    meta_data = {
        "total_tokens": len(prompt_ids) + len(response_ids),
        "prompt_length": len(prompt_ids),
        "response_length": len(response_ids),
        "is_truncated": is_truncated,
        "rollout_id": rollout_id,
        "overall_score": overall_score,
    }

    triplet = Triplet(
        prompt={"token_ids": prompt_ids,"image_urls": original_image_list},
        response={"token_ids": response_ids},
        reward=reward,
        metadata=meta_data,
    )

    return triplet


def convert_single_trace_to_triplet(
    instruction: str, trace_data: List[Dict[str, Any]], processor, reward: float, overall_score: float, rollout_id: str
) -> List[Dict[str, Any]]:

    # Step 1: 原始 trace → dataset_sample
    dataset_sample = convert_trace_to_messages(trace_data, instruction)
    if not dataset_sample:
        return {}

    # debug,检查llama 格式 triplet
    # with open(
    #     f"/root/code/wangjiaju/agent-lightning/examples/cua/trace/llama_triplet_example.json", "w", encoding="utf-8"
    # ) as f:
    #     json.dump(dataset_sample, f, ensure_ascii=False, indent=4)

    # Step 3: llama → triplet
    triplet = convert_messages_to_triplet(
        dataset_sample, processor=processor, reward=reward, overall_score=overall_score, rollout_id=rollout_id
    )

    return triplet


async def convert_traces_to_triplets(
    score_url: str,
    instruction: str,
    rollout_id: str,
    overall_score: float,
    traces: List[Dict[str, Any]],
    model_path: str,
) -> List[List[Dict[str, Any]]]:
    result = await score_trace(score_url, traces, instruction)
    if not isinstance(result, dict):
        logger.warning("score service 返回非 dict，跳过本轮")
        return []

    if "status" in result and result["status"] == "error":
        logger.warning("grouped trace went wrong!!!")
        return []

    context_traces = result.get("context_traces")
    if context_traces is None:
        context_traces = result.get("grouped_traces", [])

    if not isinstance(context_traces, list):
        logger.warning("score service 返回的 context_traces 格式错误")
        return []

    # # 保存grouped_traces以便调试
    # with open("result.json", "w", encoding="utf-8") as f:
    #     json.dump(result, f, ensure_ascii=False, indent=4)

    processor = AutoProcessor.from_pretrained(model_path, min_pixels=200704, max_pixels=1350000)

    all_tokenized_data = []
    for segmented_trace in context_traces:
        reward = segmented_trace.get("reward")
        if reward is None:
            logger.warning("skip one segmented_trace because reward is None")
            continue

        converted_trace = convert_single_trace_to_triplet(
            instruction=segmented_trace["instruction"],
            trace_data=segmented_trace["trace_data"],
            processor=processor,
            reward=reward,
            overall_score=overall_score,
            rollout_id=rollout_id,
        )
        if converted_trace:
            all_tokenized_data.append(converted_trace)

    return all_tokenized_data


# ---- test -----
def load_trace_json(path: str) -> List[Dict[str, Any]]:
    """
    读取 trace JSON 文件
    返回:
        instruction: str
        trace: List[Dict]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Trace JSON 必须是非空 List")

    # 第一条包含 instruction
    if "instruction" not in data[0]:
        raise ValueError("JSON 第一条必须包含 instruction")

    instruction = data[0]["instruction"]
    rollout_id = data[0].get("rollout_id", "unknown_rollout")
    trace = data[1:]  # 剩余是真正的 trace

    return instruction, trace


class TripletEncoder(json.JSONEncoder):
    """自定义 JSON 编码器，处理 Triplet 对象"""

    def default(self, obj):
        # 如果是 Triplet (Pydantic 对象)，转为字典
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "dict"):
            return obj.dict()
        return super().default(obj)


def save_triplets_to_json(triplets: List[Any], filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        # indent=2 让文件可读性更好，但体积会变大
        json.dump(triplets, f, cls=TripletEncoder, indent=2, ensure_ascii=False)

    logger.info(f"Saved to {filename}")


async def main():
    # ==============================
    # 路径配置
    # ==============================
    input_trace_path = "/root/workspace/wangjiaju/zql_workspace/agent-lightning/examples/cua/trace/0206/0206-qwen3-4b-sft-2500/sample_14_plan_ZQL_TEST.json"

    # ⚠️ 必须是真实存在的 Qwen-VL / Qwen2.5-VL 模型路径
    model_path = "/root/workspace/models/Qwen3-VL-8B-Instruct"

    # ==============================
    # Step 1: 读取 trace
    # ==============================
    instruction, trace_data = load_trace_json(input_trace_path)

    logger.info("📌 Instruction:")
    logger.info(instruction)
    logger.info(f"📌 Trace length: {len(trace_data)}")

    overall_score_response = await score_trace(
        url="http://localhost:8003/overall_score",
        trace=trace_data,
        user_instruction=instruction,
    )

    # ==============================
    # Step 2: trace → triplets
    # ==============================
    all_triplets = await convert_traces_to_triplets(
        score_url="http://localhost:8003/group_score",
        instruction=instruction,
        rollout_id="test_rollout_001",
        traces=trace_data,
        overall_score=overall_score_response.get("score", 0.0),
        model_path=model_path,
    )

    logger.info(f"📌 Generated {len(all_triplets)} segmented triplet groups")

    # ==============================
    # Step 3: 保存结果
    # ==============================
    save_triplets_to_json(all_triplets, "/root/workspace/zql/agent-lightning/examples/cua/trace/example_triplets.json")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
