import base64
import io
from textwrap import wrap
import os
import json
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

pdfmetrics.registerFont(TTFont("SimSun", "/root/code/wangjiaju/agent-lightning/SimSun.ttf"))


def decode_base64_image(b64_str):
    """把 base64 字符串解码为 ImageReader 可用的二进制流."""
    # 兼容 data:image/png;base64,xxx 这种前缀
    if "," in b64_str and "base64" in b64_str.split(",", 1)[0]:
        b64_str = b64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_str)
    return io.BytesIO(img_bytes)


def render_action_screenshot_pairs_to_pdf(trajectory, output_pdf_path):
    """
    将轨迹按 “action + screenshot” 两两一组渲染到 PDF，一页一组。

    trajectory: list[dict]，执行轨迹
    output_pdf_path: 输出 pdf 路径
    """
    c = canvas.Canvas(output_pdf_path, pagesize=A4)
    width, height = A4

    margin_x = 40
    margin_y = 40
    line_height = 14

    # 先把 action / screenshot 配对
    pairs = []
    pending_action = None

    for item in trajectory:
        has_action = "action" in item and item["action"]
        has_screenshot = "screenshot" in item and item["screenshot"]

        if has_action:
            pending_action = item

        if has_screenshot and pending_action is not None:
            pairs.append((pending_action, item))
            pending_action = None  # 已经用掉，等待下一对

    # 如果没有任何配对，直接保存空 PDF
    if not pairs:
        c.drawString(margin_x, height - margin_y, "没有找到可用的 action + screenshot 配对。")
        c.save()
        return

    # 每一对生成一页
    for idx, (action_item, screenshot_item) in enumerate(pairs, start=1):
        c.setFont("SimSun", 11)

        # 新页
        if idx > 1:
            c.showPage()
            c.setFont("SimSun", 11)

        y = height - margin_y

        # 页眉：编号
        c.drawString(margin_x, y, f"步骤 {idx}")
        y -= 2 * line_height

        # --------- 打印 action 相关文本 ----------
        # 这里可以按需增加/减少字段
        text_fields = [
            ("action", action_item.get("action")),
            ("summary", action_item.get("summary")),
            ("task_id", action_item.get("task_id")),
            ("total_tokens", action_item.get("total_tokens")),
        ]

        for label, value in text_fields:
            if value not in (None, ""):
                line = f"{label}: {value}"
                for wl in wrap(line, width=90):
                    if y < height / 2 + margin_y:  # 给下方截图预留空间
                        break
                    c.drawString(margin_x, y, wl)
                    y -= line_height

        # --------- 渲染截图 ----------
        b64_str = screenshot_item["screenshot"]
        img_stream = decode_base64_image(b64_str)
        img = ImageReader(img_stream)

        img_width, img_height = img.getSize()

        # 截图区域：页面下半部分
        max_width = width - 2 * margin_x
        max_height = (height / 2) - 2 * margin_y

        scale = min(max_width / img_width, max_height / img_height)
        draw_w = img_width * scale
        draw_h = img_height * scale

        img_x = (width - draw_w) / 2
        img_y = margin_y  # 靠下居中

        c.drawImage(img, img_x, img_y, width=draw_w, height=draw_h)

    c.save()


if __name__ == "__main__":
    json_dir = "/root/code/wangjiaju/agent-lightning/examples/cua/trace/0114/normal"
    for json_file in os.listdir(json_dir):
        if json_file.endswith(".json"):
            json_path = os.path.join(json_dir, json_file)
            with open(json_path, "r", encoding="utf-8") as f:
                trajectory = json.load(f)
            render_action_screenshot_pairs_to_pdf(trajectory, f"{json_path.replace('.json', '')}_pairs.pdf")
            print(f"生成完成：{json_file.replace('.json', '')}_pairs.pdf")
