#!/usr/bin/env python3
"""Generate the Section 6.4 mobility-aware LLM offloading workflow."""

from pathlib import Path
import shutil

import pypdfium2 as pdfium
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "figures" / "ch6_mobility_offloading_workflow.pdf"
OUTPUT_PATH = ROOT.parent / "output" / "pdf" / "ch6_mobility_offloading_workflow.pdf"
PREVIEW_PATH = ROOT.parent / "tmp" / "pdfs" / "ch6_mobility_offloading_workflow.png"

WIDTH = 1400
HEIGHT = 860

NAVY = "#14336F"
BLUE = "#3568D4"
CYAN = "#168E9C"
GREEN = "#43885A"
ORANGE = "#D87A24"
PURPLE = "#7359BA"
RED = "#D94848"
TEXT = "#17213A"
MUTED = "#526078"
GRID = "#B8C7E5"
PALE_BLUE = "#EDF3FF"
PALE_CYAN = "#EAF8F8"
PALE_GREEN = "#EDF7EF"
PALE_ORANGE = "#FFF3E6"
PALE_PURPLE = "#F1EDFA"
PALE_RED = "#FCEEEE"
WHITE = "#FFFFFF"

FONT = "ArialUnicode"
FONT_PATH = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")


def register_fonts():
    pdfmetrics.registerFont(TTFont(FONT, str(FONT_PATH)))


def text(c, x, y, value, size=16, color=TEXT, align="left"):
    c.setFillColor(color)
    c.setFont(FONT, size)
    if align == "center":
        c.drawCentredString(x, y, value)
    elif align == "right":
        c.drawRightString(x, y, value)
    else:
        c.drawString(x, y, value)


def rounded_box(c, x, y, w, h, fill, stroke=GRID, radius=15, line_width=2, dashed=False):
    c.saveState()
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(line_width)
    if dashed:
        c.setDash(8, 6)
    c.roundRect(x, y, w, h, radius, fill=1, stroke=1)
    c.restoreState()


def box(c, x, y, w, h, title, lines, fill=WHITE, accent=BLUE, dashed=False):
    rounded_box(c, x, y, w, h, fill=fill, stroke=accent, dashed=dashed)
    text(c, x + w / 2, y + h - 29, title, size=20, color=accent, align="center")
    for idx, line in enumerate(lines):
        text(c, x + w / 2, y + h - 58 - idx * 23, line, size=14, color=TEXT, align="center")


def stage_header(c, x, y, w, title, number, accent):
    rounded_box(c, x, y, w, 43, fill=accent, stroke=accent, radius=12)
    text(c, x + 22, y + 13, number, size=15, color=WHITE)
    text(c, x + w / 2 + 10, y + 13, title, size=17, color=WHITE, align="center")


def arrow(c, x1, y1, x2, y2, color=RED, width=3, dashed=False):
    c.saveState()
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(width)
    if dashed:
        c.setDash(8, 6)
    c.line(x1, y1, x2, y2)
    if abs(x2 - x1) >= abs(y2 - y1):
        direction = 1 if x2 >= x1 else -1
        pts = [(x2, y2), (x2 - direction * 13, y2 + 7), (x2 - direction * 13, y2 - 7)]
    else:
        direction = 1 if y2 >= y1 else -1
        pts = [(x2, y2), (x2 - 7, y2 - direction * 13), (x2 + 7, y2 - direction * 13)]
    path = c.beginPath()
    path.moveTo(*pts[0])
    path.lineTo(*pts[1])
    path.lineTo(*pts[2])
    path.close()
    c.drawPath(path, fill=1, stroke=0)
    c.restoreState()


def polyline_arrow(c, points, color=RED, width=3, dashed=False):
    c.saveState()
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(width)
    if dashed:
        c.setDash(8, 6)
    path = c.beginPath()
    path.moveTo(*points[0])
    for point in points[1:]:
        path.lineTo(*point)
    c.drawPath(path, fill=0, stroke=1)
    x1, y1 = points[-2]
    x2, y2 = points[-1]
    if abs(x2 - x1) >= abs(y2 - y1):
        direction = 1 if x2 >= x1 else -1
        pts = [(x2, y2), (x2 - direction * 13, y2 + 7), (x2 - direction * 13, y2 - 7)]
    else:
        direction = 1 if y2 >= y1 else -1
        pts = [(x2, y2), (x2 - 7, y2 - direction * 13), (x2 + 7, y2 - direction * 13)]
    head = c.beginPath()
    head.moveTo(*pts[0])
    head.lineTo(*pts[1])
    head.lineTo(*pts[2])
    head.close()
    c.drawPath(head, fill=1, stroke=0)
    c.restoreState()


def build():
    register_fonts()
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(PDF_PATH), pagesize=(WIDTH, HEIGHT))
    c.setTitle("Mobility-aware long-horizon LLM request offloading workflow")
    c.setFillColor(WHITE)
    c.rect(0, 0, WIDTH, HEIGHT, fill=1, stroke=0)

    text(c, WIDTH / 2, 817, "移动感知的长期 LLM 请求卸载与 KV 管理流程", 28, NAVY, "center")
    text(
        c,
        WIDTH / 2,
        788,
        "上层联合决定计算节点与 KV 获取方式，底层执行版本化多副本和混合恢复",
        15,
        MUTED,
        "center",
    )

    xs = [70, 520, 970, 970, 520, 70]
    ws = [360, 360, 360, 360, 360, 360]
    header_ys = [720, 720, 720, 440, 440, 440]
    headers = [
        ("请求与观测", "01", NAVY),
        ("预测与状态", "02", CYAN),
        ("成本与约束", "03", ORANGE),
        ("长期路由", "04", PURPLE),
        ("执行请求", "05", GREEN),
        ("在线反馈", "06", RED),
    ]
    for x, y, w, (title, number, accent) in zip(xs, header_ys, ws, headers):
        stage_header(c, x, y, w, title, number, accent)

    box(
        c,
        xs[0],
        555,
        ws[0],
        140,
        "请求到达",
        ["用户位置与会话标识", "提示词与完整上下文", "节点负载、显存与带宽"],
        PALE_BLUE,
        NAVY,
    )
    box(
        c,
        xs[1],
        555,
        ws[1],
        140,
        "构造系统状态",
        ["预测未来接入位置", "估计请求长度与资源需求", "汇总版本化 KV 副本"],
        PALE_CYAN,
        CYAN,
    )
    box(
        c,
        xs[2],
        555,
        ws[2],
        140,
        "成本与可行性",
        ["估计通信与推理时延", "计算当前期望成本", "按时延、显存和版本筛选"],
        PALE_ORANGE,
        ORANGE,
    )
    box(
        c,
        xs[3],
        275,
        ws[3],
        140,
        "Double DQN",
        ["评估可行动作的长期成本", "结合未来移动概率", "输出计算节点与 KV 方式"],
        PALE_PURPLE,
        PURPLE,
    )
    box(
        c,
        xs[4],
        275,
        ws[4],
        140,
        "执行请求",
        ["请求转发至计算节点", "复用、同步或重算 KV", "输入处理、生成与返回"],
        PALE_GREEN,
        GREEN,
    )
    box(
        c,
        xs[5],
        275,
        ws[5],
        140,
        "更新系统状态",
        ["记录实际端到端时延", "更新成本模型与预测器", "更新 KV 状态并处理下轮"],
        PALE_RED,
        RED,
    )

    arrow(c, xs[0] + ws[0] + 8, 625, xs[1] - 8, 625)
    arrow(c, xs[1] + ws[1] + 8, 625, xs[2] - 8, 625)
    arrow(c, xs[2] + ws[2] / 2, 547, xs[3] + ws[3] / 2, 491)
    arrow(c, xs[3] - 8, 345, xs[4] + ws[4] + 8, 345)
    arrow(c, xs[4] - 8, 345, xs[5] + ws[5] + 8, 345)

    rounded_box(c, 280, 25, 840, 205, fill="#F8FAFE", stroke=GRID, radius=18, line_width=2)
    text(c, 700, 198, "分布式 KV Manager", 20, NAVY, "center")
    text(c, 700, 173, "并列提供以下三类 KV 管理能力", 14, MUTED, "center")
    box(
        c,
        310,
        55,
        230,
        100,
        "KV 块目录",
        ["记录位置、版本和前缀哈希", "查询目标节点缺失的 KV 块"],
        PALE_BLUE,
        BLUE,
    )
    box(
        c,
        585,
        55,
        230,
        100,
        "副本与缓存策略",
        ["源副本按策略保留", "结合期限、显存和复用概率"],
        PALE_CYAN,
        CYAN,
    )
    box(
        c,
        860,
        55,
        230,
        100,
        "混合 KV 恢复",
        ["部分迁移与部分重算", "用计算空隙覆盖通信等待"],
        PALE_ORANGE,
        ORANGE,
    )
    c.showPage()
    c.save()

    shutil.copy2(PDF_PATH, OUTPUT_PATH)
    pdf = pdfium.PdfDocument(str(PDF_PATH))
    page = pdf[0]
    bitmap = page.render(scale=1.6)
    bitmap.to_pil().save(PREVIEW_PATH)
    pdf.close()


if __name__ == "__main__":
    build()
