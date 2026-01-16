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


def normalize_coordinates(tool_calls, width, height):
    new_tool_calls = copy.deepcopy(tool_calls)

    # 定义需要转换的参数集合
    # 集合查找速度快，且易于扩展
    x_keys = {"x", "start_x", "end_x"}
    y_keys = {"y", "start_y", "end_y"}

    for tool in new_tool_calls:
        # 1. 安全检查结构
        if "function" not in tool:
            continue
        
        func_node = tool["function"]
        args = func_node.get("arguments", {})

        # 2. 如果 arguments 是字符串（有些 OpenAI 响应是 JSON 字符串），先转字典
        is_json_str = False
        if isinstance(args, str):
            try:
                import json
                args = json.loads(args)
                is_json_str = True
            except:
                continue # 解析失败跳过

        if not isinstance(args, dict):
            continue

        # 3. 遍历参数进行归一化
        for key, val in args.items():
            # 跳过非数字类型 (比如 thought, content, direction 等)
            if not isinstance(val, (int, float)):
                continue
            
            # 处理 X 轴相关
            if key in x_keys:
                norm_val = int((val / width) * 1000)
                args[key] = max(0, min(1000, norm_val)) # 钳制在 0-1000
            
            # 处理 Y 轴相关
            elif key in y_keys:
                norm_val = int((val / height) * 1000)
                args[key] = max(0, min(1000, norm_val)) # 钳制在 0-1000

        # 4. 写回 arguments
        # 如果原来是字符串，这里可能需要根据你的下游任务决定是否 dump 回去
        # 通常在内部处理时，保持 dict 更方便
        if is_json_str:
             import json
             func_node["arguments"] = json.dumps(args, ensure_ascii=False)
        else:
             func_node["arguments"] = args

    return new_tool_calls

def process_trace(events, instruction):
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
    
    dataset_sample = {
        "tools": get_tools_schema(),
        "conversations": [],
        "images": []
    }

    # =============================
    # 1. 处理初始状态 (Step 0)
    # =============================
    init_event = events[0]
    if "screenshot" not in init_event or not init_event["screenshot"]:
        print("Error: Missing initial screenshot")
        return {}

    init_screenshot = init_event["screenshot"]
    
    # 解析图片尺寸 (常用于 System Prompt 注入分辨率信息，或者单纯校验图片有效性)
    try:
        pil_img = base64_to_pil(init_screenshot)
        width, height = pil_img.size
    except Exception as e:
        print(f"Error processing initial image: {e}")
        return {}

    # 保存图片
    dataset_sample["images"].append(init_screenshot)

    # 构造 System Prompt (建议带上分辨率信息)
    system_content = CUA_PROMPT
    # 如果你的 Prompt 需要动态插入分辨率，可以在这里做:
    system_content += f"\nCurrent Screen Resolution: {width}x{height}"
    
    dataset_sample["conversations"].append({"role": "system", "content": system_content})

    # 构造 User Initial Message
    dataset_sample["conversations"].append({
        "role": "user",
        "content": [
            {"type": "text", "text": instruction},
            {"type": "text", "text": "### Step 0: Initial State"},
            {"type": "image", "image": init_screenshot}
        ]
    })

    # =============================
    # 2. 遍历后续交互 (Step 1 -> N)
    # =============================
    # 从 events[1] 开始遍历 (跳过初始截图)
    for i, ev in enumerate(events[1:]):
        
        # --- Case A: 模型动作 (Assistant) ---
        if "tool_calls" in ev:
            # summary = ev.get("summary", "Thinking...")
            # raw_text = ev.get("raw_text", "")
            # if not raw_text:
            #     raw_text = summary
            tool_calls = ev["tool_calls"]
            
            tool_calls = normalize_coordinates(tool_calls, width, height)

            # 1. 思考过程 (Thought)
            dataset_sample["conversations"].append({
                "role": "assistant",
                "content": "",
                "tool_calls": tool_calls
            })
            
        # --- Case B: 环境反馈 (Tool/User) ---
        elif "tool_outputs" in ev:
            tool_outputs = ev["tool_outputs"]
            
            for tool_output in tool_outputs:
                dataset_sample["conversations"].append(tool_output)
        
        elif "screenshot" in ev:
            if "screenshot" in ev and ev["screenshot"]:
                dataset_sample["images"].append(ev["screenshot"])
                dataset_sample["conversations"].append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": ev["screenshot"]}
                        ],
                    }
                )

    # =============================
    # 3. [新增] 截断逻辑：只保留到最后一个 Assistant
    # =============================
    convs = dataset_sample["conversations"]
    
    # 从后往前检查，只要最后一条不是 assistant，就移除
    # 使用 while 循环是因为可能结尾连续跟着 [tool_output, screenshot, user_msg] 等多条非 assistant 消息
    while len(convs) > 0 and convs[-1]["role"] != "assistant":
        # 移除最后一条
        removed_msg = convs.pop()
        # 注意：这里我们只移除了对话记录。
        # 之前存在 dataset_sample["images"] 里的图片数据（base64）可以保留，
        # 因为只要 conversation 里不引用它，训练框架通常会忽略多余的资源，
        # 或者你也可以选择在这里根据 logic 复杂的去清理 images 列表，但通常没必要。
    
    # 安全检查：如果截断后 conversation 只剩下 system 或者 user (Step 0)，
    # 说明整个 trace 没有有效的 assistant 动作，这条数据通常没有训练价值。
    # 至少应该保留 [System, User, Assistant] 三条
    if len(convs) < 3: 
        print("Warning: Trace dropped because no valid assistant action found at the end.")
        return {}
    
    save_triplets_to_json(dataset_sample, "sample.json")
    return dataset_sample

def convert_to_triplet_format(converted_data, processor, reward: float, max_seq_len=16384):
    """
    使用 "分别 Tokenize (Prompt vs Full) 再相减" 的方式生成 Triplet。
    逻辑更加清晰，无需硬编码 Assistant Header Token ID。
    """
    
    conversations = converted_data["conversations"]
    
    # 1. 基础检查
    if len(conversations) < 3:
        raise Exception("conversation too short, need at least user query and assistant response.")

    # 2. 准备 Tools (如果有)
    tools = converted_data.get("tools")
    if tools is None:
        tools = []

    # 3. 拆分 Prompt Messages 和 Full Messages
    # Full: 包含所有对话
    # Prompt: 包含除最后一条 Assistant 回复之外的所有对话
    full_msgs = conversations
    prompt_msgs = conversations[:-1]

    # 4. 获取原始图片路径 (用于 Triplet 存储，非 Tokenize)
    # 假设 converted_data['images'] 存储了该条数据的图片列表
    original_image_paths = converted_data.get("images", [])

    try:
        # =========================================================
        # 🟢 A. 处理 Prompt 部分
        # Key: add_generation_prompt=True 会自动添加 <|im_start|>assistant\n
        # =========================================================
        prompt_text = processor.apply_chat_template(
            prompt_msgs, 
            tools=tools, 
            tokenize=False, 
            add_generation_prompt=True 
        )
        
        # 提取 Prompt 阶段包含的图像/视频输入
        prompt_image_inputs, prompt_video_inputs = process_vision_info(prompt_msgs)

        prompt_inputs = processor(
            text=[prompt_text],
            images=prompt_image_inputs,
            videos=prompt_video_inputs,
            padding=False,
            return_tensors="pt" # 必须返回 Tensor 才能取 input_ids
        )
        prompt_ids = prompt_inputs.input_ids[0].tolist()

        # =========================================================
        # 🔵 B. 处理 Full 部分
        # Key: add_generation_prompt=False (因为已经包含回复了)
        # =========================================================
        full_text = processor.apply_chat_template(
            full_msgs, 
            tools=tools, 
            tokenize=False, 
            add_generation_prompt=False
        )

        # 提取 Full 阶段包含的图像/视频输入
        full_image_inputs, full_video_inputs = process_vision_info(full_msgs)

        full_inputs = processor(
            text=[full_text],
            images=full_image_inputs,
            videos=full_video_inputs,
            padding=False,
            return_tensors="pt"
        )
        full_ids = full_inputs.input_ids[0].tolist()

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e

    # =========================================================
    # ✂️ C. 计算 Response Token IDs (切片逻辑)
    # =========================================================
    
    # 安全检查：Full 应该比 Prompt 长
    if len(full_ids) <= len(prompt_ids):
        # 这种情况通常意味着 response 为空，或者 tokenizer 处理异常
        print(f"⚠️ Warning: Full length ({len(full_ids)}) <= Prompt length ({len(prompt_ids)}). Skipping.")
        # 根据你的训练框架需求，这里可以选择抛出异常或返回 None
        raise Exception("Response is empty or prompt matches full length.")
    
    # (可选) 严格的一致性检查：确保 Full 的前半部分就是 Prompt
    # 在 Qwen-VL 中，由于特殊 Token 的存在，通常是匹配的。
    # 如果发现不匹配，通常是 add_generation_prompt 添加的 \n 和 Full 中的 \n 合并问题
    # 这里不做硬性 assert，防止因为极个别 token 归一化导致训练中断，但建议日志关注
    # if full_ids[:len(prompt_ids)] != prompt_ids:
    #     print("⚠️ Warning: Token mismatch at boundary. Slicing anyway.")

    response_ids = full_ids[len(prompt_ids):]

    # =========================================================
    # 📏 D. 长度检查与截断 (只截断 Response 部分)
    # =========================================================
    total_len = len(full_ids)
    is_truncated = False

    if total_len > max_seq_len:
        is_truncated = True
        # 计算允许的 response 长度
        allowed_resp_len = max_seq_len - len(prompt_ids)
        if allowed_resp_len > 0:
            response_ids = response_ids[:allowed_resp_len]
        else:
            # Prompt 已经超长了，Response 没地儿放了
            # 这里可以选择保留一部分 Prompt 或直接丢弃
            response_ids = [] # 或者抛异常

    # =========================================================
    # 💾 E. 构建输出
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

def convert_single_trace_to_triplet(
    instruction: str, trace_data: List[Dict[str, Any]], processor, reward
) -> List[Dict[str, Any]]:

    # Step 1: 原始 trace → dataset_sample
    dataset_sample = process_trace(trace_data[:-1], instruction)

    # Step 3: llama → triplet
    triplet = convert_to_triplet_format(
        dataset_sample,
        processor=processor,
        reward=reward,
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
    result = await group_score_trace(score_url, traces, instruction)
    grouped_traces = result["grouped_traces"]
    
    logger.info(f"segmented trace type:{type(grouped_traces)}, length:{len(grouped_traces)}")

    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=200704, 
        max_pixels=1350000
    )

    all_tokenized_data = []
    for segmented_trace in grouped_traces:
        reward = float(segmented_trace[-1].get("reward", 0.0)) if segmented_trace else 0.0
        converted_trace = convert_single_trace_to_triplet(
            instruction=instruction, trace_data=segmented_trace, processor=processor, reward=reward
        )
        all_tokenized_data.append(converted_trace)

    return all_tokenized_data

def convert_trace_to_triplet(instruction, trace, model_path):
    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=200704, 
        max_pixels=1350000
    )
    reward = 0.0
    converted_trace = convert_single_trace_to_triplet(
        instruction=instruction, trace_data=trace, processor=processor, reward=reward
    )
    return [converted_trace]


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
    input_trace_path = "/root/code/wangjiaju/agent-lightning/examples/cua/trace/0115/normal/sample_0_ver_0.json"

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
    # all_triplets = await convert_traces_to_triplets(
    #     score_url="http://localhost:8003/group_score",
    #     instruction=instruction,
    #     traces=trace_data,
    #     model_path=model_path,
    # )
    all_triplets = convert_trace_to_triplet(instruction=instruction, trace=trace_data, model_path=model_path)

    print(f"📌 Generated {len(all_triplets)} segmented triplet groups")

    # ==============================
    # Step 3: 保存结果
    # ==============================
    save_triplets_to_json(all_triplets, "triplets.json")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
