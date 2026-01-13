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
            print(f"Image decode error: {e}")
            return None
    return None


def extract_vision_inputs(messages: List[Dict], all_images_list: List[Any]) -> Tuple[List[Image.Image], List[Any]]:
    """
    自定义图片提取器：
    遍历 messages 中的文本，每遇到一个 <image> 标签，就从 all_images_list 中取出一张图，
    将其转为 PIL 对象并放入列表返回。
    """
    prompt_image_inputs = []
    prompt_video_inputs = []  # 暂时留空

    # 简单的迭代器，确保按顺序取图
    if all_images_list is None:
        all_images_list = []

    image_iter = iter(all_images_list)

    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            # 计算当前消息里有几个 <image> 标签
            count = content.count("<image>")
            for _ in range(count):
                try:
                    img_data = next(image_iter)
                    pil_img = base64_to_pil(img_data)
                    if pil_img:
                        prompt_image_inputs.append(pil_img)
                    else:
                        # 如果图片解码失败，给一个黑色占位图，防止报错
                        print("⚠️ Warning: Image decode failed, using placeholder.")
                        prompt_image_inputs.append(Image.new("RGB", (224, 224), (0, 0, 0)))
                except StopIteration:
                    print("⚠️ Warning: More <image> tags than images provided!")
                    break

    return prompt_image_inputs, prompt_video_inputs


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
            "name": "call_user",
            "description": "呼叫用户人工接管。",
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


def normalize_coordinates(text, width, height):
    """
    将文本中的绝对坐标转换为 0-1000 的相对坐标。
    支持的键: x, y, start_x, start_y, end_x, end_y
    """
    if not text:
        return text

    # 定义 X 轴相关的键 (使用单词边界 \b 防止匹配错误)
    # 匹配模式: 单词边界 + (key) + 可能空格 + = + 可能空格 + 数字

    def replace_x(match):
        key = match.group(1)
        val = int(match.group(2))
        # 归一化计算: (val / width) * 1000，取整
        norm_val = int((val / width) * 1000)
        # 边界保护，防止超出0-1000（虽然理论上不会，但安全起见）
        norm_val = max(0, min(1000, norm_val))
        return f"{key}={norm_val}"

    def replace_y(match):
        key = match.group(1)
        val = int(match.group(2))
        norm_val = int((val / height) * 1000)
        norm_val = max(0, min(1000, norm_val))
        return f"{key}={norm_val}"

    # 处理 X 轴: x, start_x, end_x
    text = re.sub(r"\b(x|start_x|end_x)\s*=\s*(\d+)", replace_x, text)

    # 处理 Y 轴: y, start_y, end_y
    text = re.sub(r"\b(y|start_y|end_y)\s*=\s*(\d+)", replace_y, text)

    return text


def process_trace(data, current_instruction):
    """
    data 结构（严格）:
    [
        {"instruction": "..."},
        ... trace steps (screenshot / action) ...
        {"reward": float}
    ]
    """

    # =============================
    # 0. 基本校验
    # =============================
    if not isinstance(data, list) or len(data) < 3:
        return None

    if "instruction" not in data[0]:
        raise ValueError("segmented_trace[0] 必须是 instruction")

    if "reward" not in data[-1]:
        raise ValueError("segmented_trace[-1] 必须是 reward")
    print(f"reward:{data[-1].get('reward')}")
    # =============================
    # 1. 拆 meta
    # =============================
    instruction = data[0]["instruction"]
    reward = float(data[-1]["reward"])

    # 只保留真正的 trace event
    trace_steps = data[1:-1]

    if not trace_steps:
        return None

    current_instruction = instruction

    # =============================
    # 2. 找第一张有效截图
    # =============================
    init_screenshot = None
    for entry in trace_steps:
        if "screenshot" in entry and entry["screenshot"]:
            init_screenshot = entry["screenshot"]
            break

    if not init_screenshot:
        print("no images")
        # 没有截图，无法继续
        return None

    # =============================
    # 3. 解析图片尺寸
    # =============================
    try:
        pil_img = base64_to_pil(init_screenshot)
        if pil_img:
            img_width, img_height = pil_img.size
    except Exception as e:
        print(f"Error parsing image size: {e}")
        return None

    # =============================
    # 4. 初始化 dataset_sample
    # =============================
    dataset_sample = {
        "tools": get_tools_schema(),
        "messages": [],
        "images": [],
        "reward": reward,
    }

    # -----------------------------
    # system
    # -----------------------------
    dataset_sample["messages"].append({"role": "system", "content": CUA_PROMPT})

    # -----------------------------
    # user (初始截图 + instruction)
    # -----------------------------
    dataset_sample["images"].append(init_screenshot)
    dataset_sample["messages"].append(
        {
            "role": "user", 
            "content": [
                {"type": "image", "image": init_screenshot},
                {"type": "text", "text": current_instruction}
            ]
        }
    )

    # =============================
    # 5. 遍历 trace events
    # =============================
    total_len = len(trace_steps)

    for i, entry in enumerate(trace_steps):
        if "action" not in entry:
            continue

        action_raw = entry["action"]

        # 跳过 start
        if isinstance(action_raw, str) and action_raw.lower() == "start":
            continue

        summary = entry.get("summary", "")

        raw_text = entry.get("raw_text") or summary or ""
        # raw_text = entry.get("raw_text", summary)

        # 坐标归一化
        raw_text = normalize_coordinates(raw_text, img_width, img_height)
        action_raw = normalize_coordinates(action_raw, img_width, img_height)

        tool_name, tool_args = parse_action_to_json(action_raw, summary)
        if not tool_name:
            continue

        # ---------- assistant (thought) ----------
        dataset_sample["messages"].append({"role": "assistant", "content": raw_text})

        # ---------- tool_call ----------
        dataset_sample["messages"].append(
            {
                "role": "tool_call",
                "content": json.dumps(
                    {"name": tool_name, "arguments": tool_args},
                    ensure_ascii=False,
                ),
            }
        )

        if tool_name == "finished":
            break

        # ---------- tool_response + screenshot ----------
        has_screenshot = False
        next_idx = i + 1
        if next_idx < total_len:
            next_entry = trace_steps[next_idx]
            if "screenshot" in next_entry and next_entry["screenshot"]:
                dataset_sample["images"].append(next_entry["screenshot"])
                dataset_sample["messages"].append(
                    {
                        "role": "tool_response",
                        "content": [
                            {"type": "image", "image": next_entry["screenshot"]}
                        ],
                    }
                )
                has_screenshot = True

        if not has_screenshot:
            dataset_sample["messages"].append(
                {
                    "role": "tool_response",
                    "content": json.dumps(
                        {"status": "success"},
                        ensure_ascii=False,
                    ),
                }
            )

    return dataset_sample


def convert_to_llama_factory_format(data):
    """
    data: List[Dict]
        每个 Dict 包含 "images", "tools", "messages" 等字段

    return: List[Dict]
        每个 Dict 包含 "images", "tools", "conversations" 等字段
    """
    
    new_entry = {}
    new_entry["images"] = data.get("images", [])
    new_entry["tools"] = data.get("tools", [])  # 确保包含工具定义

    new_messages = []
    messages = data.get("messages", [])

    # 用于追踪最近的一个 call_id，以便 tool_response 使用
    current_tool_call_id = None

    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            new_messages.append({"role": "system", "content": content})

        elif role == "user":
            new_messages.append({"role": "user", "content": content})

        elif role == "assistant":
            # 添加纯文本思考
            new_messages.append({"role": "assistant", "content": content})

        elif role == "tool_call":
            # 解析 tool_call 内容
            tool_call_json = json.loads(content)

            # 生成唯一的 ID (使用 uuid 或者简单的索引都可以，只要对应即可)
            call_id = f"call_{len(new_messages)}_{random.randint(1000,9999)}"
            current_tool_call_id = call_id  # 记录下来给 tool_response 用

            # 检查上一条消息是否是 assistant
            if new_messages and new_messages[-1]["role"] == "assistant":
                # 合并进去
                new_messages[-1]["tool_calls"] = [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_call_json["name"],
                            "arguments": json.dumps(tool_call_json["arguments"], ensure_ascii=False),
                        },
                    }
                ]
            else:
                # 如果上一条不是 assistant（极少见），新建一条
                new_messages.append(
                    {
                        "role": "assistant",
                        "content": "",  # 空 thought
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": tool_call_json["name"],
                                    "arguments": json.dumps(tool_call_json["arguments"], ensure_ascii=False),
                                },
                            }
                        ],
                    }
                )

        elif role == "tool_response":
            # 必须改成 "tool"
            # 如果没有对应的 ID，LLaMA-Factory 可能会报错，这里做个兜底
            if not current_tool_call_id:
                current_tool_call_id = "call_default"

            new_messages.append({"role": "tool", "content": content, "tool_call_id": current_tool_call_id})

            # 重置 ID，防止错位
            current_tool_call_id = None

    new_entry["conversations"] = new_messages

    return new_entry


def convert_to_triplet_format(converted_data, processor, reward: float, max_seq_len=16384):
    """
    通过 "从 Full 中切分 Prompt" 的方式生成 Triplet，
    彻底解决 Token Mismatch 问题。
    """

    # Qwen-VL 的 Assistant Header Token 序列
    # 对应: <|im_start|> assistant \n
    # 这是一个非常固定的锚点
    ASSISTANT_HEADER_SEQ = [151644, 77091, 198]

    conversations = converted_data["conversations"]

    # 1. 获取工具和原始图片列表
    tools = converted_data.get("tools")
    if tools is None:
        tools = []

    original_image_paths = converted_data.get("images")
    if original_image_paths is None:
        original_image_paths = []

    if len(conversations) < 2:
        raise Exception("conversation too short.")

    full_msgs = conversations

    try:
        # =========================================================
        # 🚀 优化策略: 只生成 Full，然后切分
        # =========================================================

        # 1. 生成 Full 文本
        full_text = processor.apply_chat_template(
            full_msgs, tools=tools, tokenize=False, add_generation_prompt=False
        )

        # 2. 提取所有图片 (Prompt + Response 的图片都在这里面)
        full_image_inputs, _ = process_vision_info(full_msgs)

        # 3. Tokenize Full (这是唯一的 Ground Truth)
        full_inputs = processor(text=[full_text], images=full_image_inputs, padding=False, return_tensors="pt")
        full_ids = full_inputs.input_ids[0].tolist()

        # =========================================================
        # ✂️ 切分 Prompt 和 Response
        # =========================================================

        # 我们需要在 full_ids 中找到 *最后一个* assistant header 的位置
        # 这个位置就是 Prompt 和 Response 的分界线

        split_idx = -1
        seq_len = len(ASSISTANT_HEADER_SEQ)

        # 倒序查找，确保找到的是最后一个（即 Response 前的那个）
        for idx in range(len(full_ids) - seq_len, -1, -1):
            if full_ids[idx : idx + seq_len] == ASSISTANT_HEADER_SEQ:
                # 找到了！切分点在 header 之后
                split_idx = idx + seq_len
                break

        if split_idx == -1:
            # 极少见情况：可能 response 没有换行，尝试去掉 \n 找 [151644, 77091]
            fallback_seq = [151644, 77091]
            for idx in range(len(full_ids) - 2, -1, -1):
                if full_ids[idx : idx + 2] == fallback_seq:
                    split_idx = idx + 2
                    print(f"⚠️ Found header without newline, splitting anyway.")
                    break

        if split_idx == -1:
            raise Exception("❌ Could not find assistant header to split Prompt/Response.")

        # 执行切分
        prompt_ids = full_ids[:split_idx]
        response_ids = full_ids[split_idx:]

    except Exception as e:
        import traceback

        traceback.print_exc()

    # =========================================================
    # 📏 长度检查与截断
    # =========================================================

    # 此时 prompt_ids + response_ids 必定等于 full_ids，无需检查 mismatch

    total_len = len(full_ids)
    is_truncated = False

    if total_len > max_seq_len:
        is_truncated = True
    
    resp_len = total_len - len(prompt_ids)
    response_ids = response_ids[:resp_len]

    # =========================================================
    # 💾 构建输出
    # =========================================================
    meta_data = {
        "total_tokens": len(prompt_ids) + len(response_ids),
        "prompt_length": len(prompt_ids),
        "response_length": len(response_ids),
        "is_truncated": is_truncated,
    }

    triplet = Triplet(
        prompt={"token_ids": prompt_ids, "image_urls": original_image_paths},
        response={"token_ids": response_ids},
        reward=reward,
        metadata=meta_data,
    )

    return triplet


class TripletEncoder(json.JSONEncoder):
    """自定义 JSON 编码器，处理 Triplet 对象"""

    def default(self, obj):
        # 如果是 Triplet (Pydantic 对象)，转为字典
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "dict"):
            return obj.dict()
        return super().default(obj)


def convert_single_trace_to_triplet(
    instruction: str, trace_data: List[Dict[str, Any]], processor
) -> List[Dict[str, Any]]:

    # reward
    reward_value = float(trace_data[-1].get("reward", 0.0)) if trace_data else 0.0

    # Step 1: 原始 trace → dataset_sample
    dataset_sample = process_trace(trace_data, instruction)
    if dataset_sample is None:
        return {}

    # Step 2: dataset_sample → llama factory
    llama_entries = convert_to_llama_factory_format(dataset_sample)
    with open("llama.json", "w", encoding="utf-8") as f:
        json.dump(llama_entries, f, ensure_ascii=False, indent=4)
    # Step 3: llama → triplet
    triplet = convert_to_triplet_format(
        llama_entries,
        processor=processor,
        reward=reward_value,
    )

    return triplet

async def group_score_trace(url, trace: list[dict]=None, user_instruction: str=None):
    try:
        # 1. 检查数据量，防止发送过大炸弹
        # 如果 trace 中包含 image，建议在此处做截断或只发 url
        payload = {
            "trace": trace,
            "user_instruction": user_instruction,
            "contents": None,
        }
        
        # 打印大小日志
        payload_str = json.dumps(payload)
        payload_mb = len(payload_str) / (1024 * 1024)
        print(f"正在发送评分请求，数据大小: {payload_mb:.2f} MB")
        
        if payload_mb > 50: # 假设阈值是 50MB
            print("数据包过大，可能会导致连接中断！建议检查 trace 是否包含过多 Base64 图片。")

        # 2. 配置重试策略
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        session.mount('http://', HTTPAdapter(max_retries=retries))
        session.mount('https://', HTTPAdapter(max_retries=retries))

        # 3. 发送请求 (使用 json=payload 自动处理 header)
        # 这里的 timeout=(连接超时, 读取超时)
        response = session.post(url, json=payload, timeout=(10, 600)) 
        
        response.raise_for_status()
        result = response.json()
        print(f"Planner 评分轨迹成功")
        return result

    except requests.exceptions.ConnectionError as e:
        print(f"连接被重置/断开。原因可能是数据包过大或服务端崩溃。Error: {e}")
        return {}
    except Exception as e:
        print("评分轨迹失败: %s", e)
        return {}
    
async def convert_traces_to_triplets(
    score_url: str, instruction: str, traces: List[Dict[str, Any]], model_path: str
) -> List[List[Dict[str, Any]]]:
    logger.info(f"score_url: {score_url}")
    result = await group_score_trace(score_url, traces, instruction)
    grouped_traces = result["grouped_traces"]
    # 测试直接输入代码
    # input_segmented_trace = "/root/code/wangjiaju/zql_workspace/agent-lightning/examples/cua/reward_result.json"
    # with open(input_segmented_trace, "r", encoding="utf-8") as f:
    #     segmented_traces = json.load(f)
    logger.info(f"segmented trace type:{type(grouped_traces)}, length:{len(grouped_traces)}")

    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=256*28*28, 
        max_pixels=1280*28*28
    )

    all_tokenized_data = []
    for segmented_trace in grouped_traces:
        converted_trace = convert_single_trace_to_triplet(
            instruction=instruction, trace_data=segmented_trace, processor=processor
        )
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
    trace = data[1:]  # 剩余是真正的 trace

    return instruction, trace


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


async def main():
    # ==============================
    # 路径配置
    # ==============================
    input_trace_path = "/root/code/wangjiaju/agent-lightning/examples/cua/trace/0112/lora-602112/sample_3_ver_0.json"
    output_triplets_path = "output_triplets.json"

    # ⚠️ 必须是真实存在的 Qwen-VL / Qwen2.5-VL 模型路径
    model_path = "/models/Qwen3-VL-8B-Instruct"

    # ==============================
    # Step 1: 读取 trace
    # ==============================
    instruction, trace_data = load_trace_json(input_trace_path)

    print("📌 Instruction:")
    print(instruction)
    print(f"📌 Trace length: {len(trace_data)}")

    # ==============================
    # Step 2: trace → triplets
    # ==============================
    all_triplets = await convert_traces_to_triplets(
        score_url="http://localhost:8003/group_score",
        instruction=instruction,
        traces=trace_data,
        model_path=model_path,
    )

    print(f"📌 Generated {len(all_triplets)} segmented triplet groups")

    # ==============================
    # Step 3: 保存结果
    # ==============================
    save_triplets_to_json(all_triplets, "triplets.json")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
