import io
import os
import json
import base64
import textwrap
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

pdfmetrics.registerFont(TTFont("SimSun", "/root/code/wangjiaju/agent-lightning/SimSun.ttf"))

def decode_base64_image(b64_str):
    """解码 Base64 图片"""
    if not b64_str:
        return None
    if "," in b64_str and "base64" in b64_str.split(",", 1)[0]:
        b64_str = b64_str.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(b64_str)
        return io.BytesIO(img_bytes)
    except Exception as e:
        print(f"Image decode error: {e}")
        return None

def organize_steps(trajectory):
    """
    将扁平的轨迹列表组织成 [Step0, Step1, Step2...] 的结构
    每个 Step 包含: {action, outputs, screenshot, step_id}
    """
    steps = []
    
    # 1. 寻找初始状态 (Step 0)
    # 通常是第一个包含 screenshot 但没有 tool_calls/outputs 的事件
    init_event = None
    for ev in trajectory:
        if "screenshot" in ev and ev["screenshot"] and "tool_calls" not in ev and "tool_outputs" not in ev:
            init_event = ev
            break
            
    if init_event:
        steps.append({
            "type": "init",
            "step_id": 0,
            "screenshot": init_event["screenshot"],
            "instruction": "Initial State (Before Start)"
        })

    # 2. 聚合 Action -> Result 循环
    # 逻辑：遇到 tool_calls 开启新 Step，后续的 outputs/screenshot 归入该 Step
    current_step = None
    step_count = 1

    for ev in trajectory:
        # 忽略已经作为初始状态处理过的事件
        if ev == init_event:
            continue

        # Case A: 新的动作 (Action)
        if "tool_calls" in ev:
            # 如果之前有未保存的 step（虽然理论上不太可能，除非数据缺失），先保存
            if current_step:
                steps.append(current_step)
            
            current_step = {
                "type": "action",
                "step_id": step_count,
                "action": ev,          # 包含 tool_calls, summary, thought
                "outputs": None,       # 待填充
                "screenshot": None     # 待填充
            }
            step_count += 1
            
        # Case B: 工具反馈 (Result - Outputs)
        elif "tool_outputs" in ev:
            if current_step:
                current_step["outputs"] = ev["tool_outputs"]
                # 如果这个事件里同时也包含 screenshot (你的某些版本代码可能混合了)
                if "screenshot" in ev and ev["screenshot"]:
                    current_step["screenshot"] = ev["screenshot"]

        # Case C: 视觉反馈 (Result - Screenshot)
        # 处理单独的 screenshot 事件
        elif "screenshot" in ev and ev["screenshot"]:
            if current_step and current_step["screenshot"] is None:
                current_step["screenshot"] = ev["screenshot"]

    # 循环结束后，别忘了把最后一个 step 加入
    if current_step:
        steps.append(current_step)

    return steps

def draw_text_lines(c, text_lines, x, start_y, line_height, min_y):
    """
    辅助函数：绘制多行文本，如果超出 min_y 则停止绘制并返回 False (表示空间不足)
    """
    y = start_y
    for line in text_lines:
        if y < min_y:
            c.drawString(x, y, "... (text truncated for space)")
            return y, False # 空间不足
        c.drawString(x, y, line)
        y -= line_height
    return y, True

def render_action_screenshot_pairs_to_pdf(trajectory, output_pdf_path):
    c = canvas.Canvas(output_pdf_path, pagesize=A4)
    width, height = A4
    margin = 40
    line_height = 12
    
    # 页面布局配置
    # 上半部分用于文本 (Action + Output)，下半部分用于截图
    # text_bottom_limit 限制文本不能写到太下面，给图片留位置
    text_bottom_limit = height * 0.45 
    
    steps = organize_steps(trajectory)

    if not steps:
        c.drawString(margin, height - margin, "No valid steps found in trajectory.")
        c.save()
        return

    c.setFont("SimSun", 10) # 设置默认字体

    for step in steps:
        # === 每一页的通用处理 ===
        y = height - margin
        
        # 1. 标题
        header = f"Step {step['step_id']}: {step['type'].upper()}"
        if step['type'] == 'init':
            header += " - Initial State"
        
        c.setFont("SimSun", 14)
        c.drawString(margin, y, header)
        c.setFont("SimSun", 10) # 恢复正文字体
        y -= 25

        # === 文本区域处理 ===
        
        # 准备要打印的文本内容
        content_lines = []
        
        if step['type'] == 'init':
            content_lines.append("Instruction: User provided initial state.")
        else:
            # Action Info
            action_data = step.get('action', {})
            summary = action_data.get('summary', 'No summary')
            content_lines.append(f"Thought/Summary: {summary}")
            
            tool_calls_str = json.dumps(action_data.get('tool_calls', []), ensure_ascii=False, indent=2)
            content_lines.append(f"Tool Calls:")
            # 将 JSON 拆行处理
            for line in tool_calls_str.split('\n'):
                content_lines.append("  " + line)

            # Outputs Info
            outputs = step.get('outputs')
            if outputs:
                content_lines.append("-" * 80) # 分隔线
                content_lines.append("Tool Outputs:")
                outputs_str = json.dumps(outputs, ensure_ascii=False, indent=2)
                # 限制 Output 长度，防止 log 太多把图片挤没了
                output_lines = outputs_str.split('\n')
                if len(output_lines) > 20:
                    output_lines = output_lines[:20] + ["... (outputs truncated)"]
                for line in output_lines:
                    content_lines.append("  " + line)

        # 渲染文本行
        for raw_line in content_lines:
            # 自动换行 (Wrap)
            wrapped_lines = textwrap.wrap(raw_line, width=95) 
            current_y, has_space = draw_text_lines(c, wrapped_lines, margin, y, line_height, text_bottom_limit)
            y = current_y
            if not has_space:
                break # 空间不够了，停止渲染文本

        # === 图片区域处理 ===
        b64_img = step.get('screenshot')
        if b64_img:
            img_stream = decode_base64_image(b64_img)
            if img_stream:
                try:
                    img = ImageReader(img_stream)
                    img_w, img_h = img.getSize()

                    # 计算可用空间：页面底部到 text_bottom_limit 之间
                    # 为了美观，我们在 text_bottom_limit 下方一点开始画
                    avail_width = width - 2 * margin
                    avail_height = text_bottom_limit - margin 

                    # 计算缩放比例 (保持长宽比 fit)
                    scale = min(avail_width / img_w, avail_height / img_h)
                    draw_w = img_w * scale
                    draw_h = img_h * scale

                    # 居中放置在页面底部区域
                    draw_x = (width - draw_w) / 2
                    # 图片底部对齐 margin
                    draw_y = margin + (avail_height - draw_h) / 2 

                    c.drawImage(img, draw_x, draw_y, width=draw_w, height=draw_h)
                except Exception as e:
                    c.drawString(margin, text_bottom_limit - 20, f"[Image Render Error: {e}]")
        else:
            c.drawString(margin, text_bottom_limit - 20, "[No Screenshot Available for this step]")

        # 结束当前页
        c.showPage()

    c.save()

if __name__ == "__main__":
    json_dir = "/root/code/wangjiaju/agent-lightning/examples/cua/trace/0115/normal"
    for json_file in os.listdir(json_dir):
        if json_file.endswith(".json"):
            json_path = os.path.join(json_dir, json_file)
            with open(json_path, "r", encoding="utf-8") as f:
                trajectory = json.load(f)
            render_action_screenshot_pairs_to_pdf(trajectory, f"{json_path.replace('.json', '')}_pairs.pdf")
            print(f"生成完成：{json_file.replace('.json', '')}_pairs.pdf")
