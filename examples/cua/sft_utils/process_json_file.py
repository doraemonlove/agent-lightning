import os
import json
import re

patterns = [
    r"x=1417,\s*y=119",
    r"x=1419,\s*y=12[0-3]",
    r"x=141[89],\s*y=11[8-9]",
    r"x=1418,\s*y=11[7-9]",
    r"x=1418,\s*y=12[0-1]",
    r"x=1418,\s*y=3[2-3]",
    r"x=1419,\s*y=24",
    r"Hotkey\(key=cmd\sq",
    r"x=143[23],\s*y=1[8-9]",
    r"x=143[23],\s*y=2[0-4]",
    r"x=142[12],\s*y=2[3-4]",
    r"x=1433,\s*y=102",
    r'"440":\s*"440"',
    r'"440":\s*"y"',
    r"Bad_Function_Tool",
]


def fix_tool_arguments(item):
    """
    【数据清洗】
    检查 tool_calls 中的 arguments。
    针对错误格式 "x": "295, 180" 进行修复，改为 "x": 295 (int)。
    该操作会直接修改 item 对象。
    """
    if not isinstance(item, dict) or "tool_calls" not in item:
        return item

    try:
        for tool in item["tool_calls"]:
            # 确保结构正确
            if "function" in tool and "arguments" in tool["function"]:
                args_content = tool["function"]["arguments"]

                # 如果 arguments 是字符串，先反序列化为字典
                args_dict = {}
                is_str_mode = False

                if isinstance(args_content, str):
                    try:
                        args_dict = json.loads(args_content)
                        is_str_mode = True
                    except:
                        continue  # 解析失败则跳过
                elif isinstance(args_content, dict):
                    args_dict = args_content

                updated = False

                # --- 核心修复逻辑 ---
                # 检查 x 字段
                if "x" in args_dict:
                    val = args_dict["x"]
                    if isinstance(val, str):
                        # 场景1: "x": "295, 180" -> 这种带逗号的
                        if "," in val:
                            # 提取逗号前的部分
                            clean_val = val.split(",")[0].strip()
                            if clean_val.isdigit():
                                args_dict["x"] = int(clean_val)
                                updated = True
                        # 场景2: "x": "295" -> 纯数字字符串
                        elif val.strip().isdigit():
                            args_dict["x"] = int(val.strip())
                            updated = True

                # 检查 y 字段 (顺手修一下纯数字字符串的情况)
                if "y" in args_dict:
                    val = args_dict["y"]
                    if isinstance(val, str) and val.strip().isdigit():
                        args_dict["y"] = int(val.strip())
                        updated = True

                # 如果发生了修复，回写回去
                if updated:
                    # print(f"  - [Auto Fix] 修复了坐标格式: {args_content} -> {args_dict}")
                    if is_str_mode:
                        tool["function"]["arguments"] = json.dumps(
                            args_dict, ensure_ascii=False
                        )
                    else:
                        tool["function"]["arguments"] = args_dict

    except Exception as e:
        # 修复过程中如果出错，不阻断主流程，直接略过
        # print(f"修复尝试失败: {e}")
        pass

    return item


def process_json_files(input_dir, output_dir=None):
    """
    遍历指定目录下的所有JSON文件。
    针对 Screenshot -> Tool_Calls -> Tool_Outputs 结构：
    如果 Tool_Calls 匹配正则，则删除：上一条(Screenshot) + 当前条(Call) + 下一条(Output)
    """
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"跳过：目录不存在 -> {input_dir}")
        return

    # 创建输出目录（如果需要）
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 遍历目录中的所有JSON文件
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(input_dir, filename)
            print(f"正在处理文件: {file_path}")

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if not isinstance(data, list):
                    print(f"警告: {filename} 不是列表结构，跳过处理")
                    continue

                processed_data = []
                i = 0
                while i < len(data):
                    item = data[i]
                    # 先尝试修复 tool arguments 中的坐标格式问题
                    item = fix_tool_arguments(item)

                    match_found = False

                    # --- 正则匹配逻辑 ---
                    if isinstance(item, dict):
                        for key, value in item.items():
                            check_content = None
                            if isinstance(value, str):
                                check_content = value
                            elif isinstance(value, (dict, list)):
                                check_content = json.dumps(value)

                            if check_content:
                                for pattern in patterns:
                                    if re.search(pattern, check_content):
                                        match_found = True
                                        # print(f"  - [DEBUG] Key: {key} 匹配到正则: '{pattern}'")
                                        break
                            if match_found:
                                break

                    # --- 删除逻辑 (根据数据类型调整) ---
                    if match_found:
                        # 情况 A: 匹配项是 Tool Calls (Action)
                        # 策略: 回溯删除上一条(Screenshot) + 跳过当前(Call) + 跳过下一条(Output)
                        if "tool_calls" in item:
                            print(
                                f"  - [Tool Call 过滤] 匹配索引 {i}。删除: 上一条(Screenshot) + 当前(Call) + 下一条(Output)"
                            )

                            # 1. 回溯删除 list 中已添加的上一条 (Screenshot)
                            if processed_data:
                                removed = processed_data.pop()
                                # 可选: 检查删除的是否是 screenshot
                                # if "screenshot" not in removed:
                                #     print(f"    警告: 回溯删除的不是 screenshot, 而是 {removed.keys()}")

                            # 2. 跳过当前(Call) 和 下一条(Output)
                            i += 2

                        # 情况 B: 匹配项是 Tool Outputs (Result) - 极少情况，除非错误信息写在output里
                        # 策略: 需要回溯删除 Call 和 Screenshot
                        elif "tool_outputs" in item:
                            print(
                                f"  - [Tool Output 过滤] 匹配索引 {i}。回溯删除 Call 和 Screenshot。"
                            )
                            if processed_data:
                                processed_data.pop()  # 删除 Call
                            if processed_data:
                                processed_data.pop()  # 删除 Screenshot
                            i += 1  # 跳过当前 Output

                        # 情况 C: 其他 (Instruction 或 Screenshot 自身匹配到)
                        else:
                            print(
                                f"  - [其他过滤] 删除索引 {i} (类型: {list(item.keys())})"
                            )
                            i += 1
                    else:
                        processed_data.append(item)
                        i += 1

                # 保存处理后的数据
                if output_dir:
                    output_path = os.path.join(output_dir, filename)
                else:
                    output_path = file_path

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(processed_data, f, ensure_ascii=False, indent=2)

                print(
                    f"文件 {filename} 处理完成 (原:{len(data)} -> 现:{len(processed_data)})"
                )
                print("-" * 30)

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")


if __name__ == "__main__":
    # 输入文件夹列表
    input_directories = [
        "/root/code/wangjiaju/zql_workspace/Datasets/claude_0128_2573",
    ]

    # 基础输出目录
    base_output_directory = (
        "/root/code/wangjiaju/zql_workspace/LLaMA-Factory/data/cua_data/cua_json_data"
    )

    # 遍历列表进行处理
    for folder in input_directories:
        # 清理路径中的双斜杠
        folder = os.path.normpath(folder)

        # 自动获取文件夹名 (例如 claude_0121_807)
        folder_name = os.path.basename(folder)

        # 拼接新的输出子目录，防止不同数据集文件同名覆盖
        target_output_dir = os.path.join(base_output_directory, folder_name)

        print(f"\n>>> 开始处理任务组: {folder_name}")
        print(f"    输入: {folder}")
        print(f"    输出: {target_output_dir}")

        process_json_files(folder, target_output_dir)
        print(f"<<< 任务组 {folder_name} 处理结束\n")
