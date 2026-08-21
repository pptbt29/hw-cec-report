#!/usr/bin/env python3
"""Generate the detailed Section 6.4 request offloading workflow figure."""

from pathlib import Path
import shutil

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "figures" / "ch6_mobility_offloading_workflow.pdf"
OUTPUT_PATH = ROOT.parent / "output" / "pdf" / "ch6_mobility_offloading_workflow.pdf"

WIDTH = 1800
HEIGHT = 1080

NAVY = "#12306B"
BLUE = "#2E63D9"
CYAN = "#0796A6"
GREEN = "#3D965B"
ORANGE = "#E48322"
PURPLE = "#7057C8"
RED = "#D94848"
TEXT = "#18233A"
MUTED = "#5B6A82"
GRID = "#B9C7E6"
PALE_BLUE = "#EEF4FF"
PALE_CYAN = "#ECFAFA"
PALE_GREEN = "#EEF8F0"
PALE_ORANGE = "#FFF4E8"
PALE_PURPLE = "#F2EEFC"
PALE_RED = "#FCEEEE"
PALE_GRAY = "#F8FAFE"
WHITE = "#FFFFFF"

FONT = "ArialUnicode"
FONT_PATH = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")


def register_fonts():
    global FONT
    if FONT_PATH.exists():
        pdfmetrics.registerFont(TTFont(FONT, str(FONT_PATH)))
    else:
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont

        FONT = "STSong-Light"
        pdfmetrics.registerFont(UnicodeCIDFont(FONT))


def text(c, x, y, value, size=16, color=TEXT, align="left"):
    c.setFont(FONT, size)
    c.setFillColor(color)
    if align == "center":
        c.drawCentredString(x, y, value)
    elif align == "right":
        c.drawRightString(x, y, value)
    else:
        c.drawString(x, y, value)


def rounded_box(c, x, y, w, h, fill=WHITE, stroke=GRID, radius=18, line_width=2, dashed=False):
    c.saveState()
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(line_width)
    if dashed:
        c.setDash(8, 6)
    c.roundRect(x, y, w, h, radius, fill=1, stroke=1)
    c.restoreState()


def card(c, x, y, w, h, title, lines, accent=BLUE, fill=WHITE, title_size=18, body_size=13.2):
    rounded_box(c, x, y, w, h, fill=fill, stroke=accent, radius=18, line_width=2)
    text(c, x + 18, y + h - 29, title, title_size, accent)
    y0 = y + h - 58
    for idx, line in enumerate(lines):
        text(c, x + 20, y0 - idx * 21, line, body_size, TEXT)


def label_box(c, x, y, w, h, label, accent=BLUE, fill=WHITE, size=12.5):
    rounded_box(c, x, y, w, h, fill=fill, stroke=accent, radius=15, line_width=1.7)
    text(c, x + w / 2, y + h / 2 - 5, label, size, accent, "center")


def arrow(c, x1, y1, x2, y2, color=RED, width=3, dashed=False, head=12):
    c.saveState()
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(width)
    if dashed:
        c.setDash(8, 6)
    c.line(x1, y1, x2, y2)
    if abs(x2 - x1) >= abs(y2 - y1):
        d = 1 if x2 >= x1 else -1
        pts = [(x2, y2), (x2 - d * head, y2 + head * 0.55), (x2 - d * head, y2 - head * 0.55)]
    else:
        d = 1 if y2 >= y1 else -1
        pts = [(x2, y2), (x2 - head * 0.55, y2 - d * head), (x2 + head * 0.55, y2 - d * head)]
    p = c.beginPath()
    p.moveTo(*pts[0])
    p.lineTo(*pts[1])
    p.lineTo(*pts[2])
    p.close()
    c.drawPath(p, fill=1, stroke=0)
    c.restoreState()


def polyline(c, pts, color=RED, width=3, dashed=False):
    c.saveState()
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(width)
    if dashed:
        c.setDash(8, 6)
    path = c.beginPath()
    path.moveTo(*pts[0])
    for pt in pts[1:]:
        path.lineTo(*pt)
    c.drawPath(path, fill=0, stroke=1)
    x1, y1 = pts[-2]
    x2, y2 = pts[-1]
    if abs(x2 - x1) >= abs(y2 - y1):
        d = 1 if x2 >= x1 else -1
        head = [(x2, y2), (x2 - d * 12, y2 + 7), (x2 - d * 12, y2 - 7)]
    else:
        d = 1 if y2 >= y1 else -1
        head = [(x2, y2), (x2 - 7, y2 - d * 12), (x2 + 7, y2 - d * 12)]
    p = c.beginPath()
    p.moveTo(*head[0])
    p.lineTo(*head[1])
    p.lineTo(*head[2])
    p.close()
    c.drawPath(p, fill=1, stroke=0)
    c.restoreState()


def cylinder(c, x, y, w, h, title, lines, accent=GREEN, fill=PALE_GREEN):
    c.saveState()
    c.setFillColor(fill)
    c.setStrokeColor(accent)
    c.setLineWidth(2)
    c.rect(x, y + 18, w, h - 36, fill=1, stroke=0)
    c.ellipse(x, y + h - 36, x + w, y + h, fill=1, stroke=1)
    c.line(x, y + 18, x, y + h - 18)
    c.line(x + w, y + 18, x + w, y + h - 18)
    c.ellipse(x, y, x + w, y + 36, fill=1, stroke=1)
    c.restoreState()
    text(c, x + w / 2, y + h - 49, title, 16, accent, "center")
    for i, line in enumerate(lines):
        text(c, x + w / 2, y + h - 76 - i * 20, line, 12, TEXT, "center")


def build():
    register_fonts()
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(PDF_PATH), pagesize=(WIDTH, HEIGHT))
    c.setTitle("Detailed request-level mobility-aware LLM offloading workflow")
    c.setFillColor(WHITE)
    c.rect(0, 0, WIDTH, HEIGHT, fill=1, stroke=0)

    text(c, WIDTH / 2, 1030, "单个 Request 到达后的长期成本感知卸载决策流程", 30, NAVY, "center")
    text(c, WIDTH / 2, 998, "先预测与过滤，再让 Double DQN 在可行动作中选择长期价值最优动作", 15, MUTED, "center")

    # Top-left: request decomposition.
    card(c, 70, 875, 250, 86, "Request", ["用户信息、prompt 信息、SLA", "当前入口与 session 标识"], NAVY, PALE_BLUE)
    card(c, 60, 715, 220, 104, "User Info", ["当前入口节点", "历史移动轨迹", "会话连续性"], NAVY, PALE_BLUE)
    card(c, 310, 715, 220, 104, "Prompt Info", ["输入长度与上下文", "任务类型或 embedding", "历史对话摘要"], CYAN, PALE_CYAN)
    card(c, 560, 715, 220, 104, "SLA", ["E2E latency 上限", "业务优先级", "是否允许降级"], ORANGE, PALE_ORANGE)
    arrow(c, 195, 875, 170, 819, NAVY, 2.5)
    arrow(c, 195, 875, 420, 819, NAVY, 2.5)
    arrow(c, 195, 875, 670, 819, NAVY, 2.5)

    # Predictors.
    card(c, 60, 560, 220, 108, "Mobility Predictor", ["输出未来入口分布", "刻画用户移动不确定性"], BLUE, PALE_BLUE)
    card(c, 310, 560, 220, 108, "Output Length Predictor", ["输出 decode 长度估计", "用于当前时延与显存估计"], CYAN, PALE_CYAN)
    arrow(c, 170, 715, 170, 668, BLUE, 2.6)
    arrow(c, 420, 715, 420, 668, CYAN, 2.6)

    # System-side information.
    cylinder(c, 60, 360, 220, 112, "CEC System State", ["节点负载与剩余显存", "链路有效带宽", "模型实例可用性"], GREEN, PALE_GREEN)
    cylinder(c, 310, 360, 220, 112, "KV Context Manager", ["KV block 位置与版本", "prefix hash 与副本策略", "同步/重算来源"], GREEN, PALE_GREEN)

    # Center: state builder.
    rounded_box(c, 650, 490, 420, 250, fill=PALE_GRAY, stroke=GRID, radius=22, line_width=2)
    text(c, 680, 705, "State Builder / Feature Bundle", 20, NAVY)
    text(c, 680, 675, "将预测结果、系统遥测和 KV 状态组合成决策输入", 13, MUTED)
    label_box(c, 690, 625, 165, 36, "mobility distribution", BLUE, PALE_BLUE)
    label_box(c, 875, 625, 160, 36, "predicted length", CYAN, PALE_CYAN)
    label_box(c, 690, 575, 165, 36, "CEC system state", GREEN, PALE_GREEN)
    label_box(c, 875, 575, 160, 36, "KV cache state", GREEN, PALE_GREEN)
    label_box(c, 775, 525, 175, 36, "request SLA", ORANGE, PALE_ORANGE)
    arrow(c, 280, 615, 650, 645, BLUE, 2.2)
    arrow(c, 530, 615, 650, 645, CYAN, 2.2)
    arrow(c, 280, 415, 650, 595, GREEN, 2.2)
    arrow(c, 530, 415, 650, 595, GREEN, 2.2)
    polyline(c, [(670, 715), (670, 543), (775, 543)], ORANGE, 2.2)

    # Action generation and per-action estimation.
    card(c, 1125, 760, 270, 110, "Action Enumerator", ["枚举计算节点", "枚举 KV 获取方式", "reuse / sync / recompute"], PURPLE, PALE_PURPLE)
    card(c, 1125, 555, 300, 135, "Per-action Estimator", ["逐动作估计 request 传输", "逐动作估计 KV 同步 / 重算", "逐动作估计 prefill、decode 与显存"], ORANGE, PALE_ORANGE)
    arrow(c, 780, 767, 1125, 815, PURPLE, 2.5)
    arrow(c, 1260, 760, 1260, 690, PURPLE, 2.8)
    arrow(c, 1070, 615, 1125, 620, ORANGE, 2.8)

    # Filters.
    card(c, 1500, 745, 245, 95, "E2E SLA Filter", ["剔除预计超时动作", "使用 SLA 与长度预测"], RED, PALE_RED)
    card(c, 1500, 590, 245, 95, "Memory / KV Filter", ["剔除显存不足动作", "检查 KV 版本与节点可用性"], RED, PALE_RED)
    card(c, 1500, 425, 245, 95, "Feasible Actions", ["满足硬约束的动作", "作为 DQN action mask"], GREEN, PALE_GREEN)
    arrow(c, 1425, 620, 1500, 790, RED, 2.8)
    arrow(c, 1623, 745, 1623, 685, RED, 2.8)
    arrow(c, 1623, 590, 1623, 520, RED, 2.8)

    # DQN router.
    card(c, 920, 255, 360, 135, "Double DQN Router", ["输入 state bundle 与 feasible actions", "评估每个动作的长期成本", "选择长期成本最低动作"], PURPLE, PALE_PURPLE, 20, 13.5)
    arrow(c, 860, 490, 1010, 390, PURPLE, 2.6)
    polyline(c, [(1500, 470), (1310, 470), (1310, 330), (1280, 330)], GREEN, 2.8)

    rounded_box(c, 1495, 315, 285, 68, fill=WHITE, stroke=GRID, radius=17, line_width=1.6)
    text(c, 1517, 358, "action examples", 13, MUTED)
    label_box(c, 1645, 345, 90, 27, "A + reuse", GREEN, PALE_GREEN, 11)
    label_box(c, 1517, 318, 85, 27, "B + sync", PURPLE, PALE_PURPLE, 11)
    label_box(c, 1612, 318, 120, 27, "C + recompute", ORANGE, PALE_ORANGE, 11)

    # Final action and execution.
    card(c, 1495, 190, 260, 100, "Final Action", ["计算节点 + KV 获取方式", "node + reuse / sync / recompute"], RED, PALE_RED)
    arrow(c, 1280, 322, 1495, 240, RED, 3)
    rounded_box(c, 1330, 55, 390, 82, fill="#5368B3", stroke="#5368B3", radius=41, line_width=2)
    text(c, 1525, 88, "Environment Execution", 18, WHITE, "center")
    arrow(c, 1600, 190, 1540, 137, RED, 3)

    # Feedback and training loop.
    card(c, 650, 55, 270, 120, "Observed Feedback", ["实际 E2E 与输出长度", "KV 增量与传输量", "显存变化与执行结果"], RED, PALE_RED)
    cylinder(c, 340, 55, 250, 120, "Replay Buffer", ["state, action, cost", "next state 与观测反馈"], ORANGE, PALE_ORANGE)
    arrow(c, 1330, 90, 920, 105, RED, 2.6)
    arrow(c, 650, 115, 590, 115, RED, 2.6)
    polyline(c, [(465, 175), (465, 300), (920, 300)], PURPLE, 2.5, dashed=True)
    text(c, 495, 285, "update Double DQN", 12.5, PURPLE)
    polyline(c, [(465, 175), (465, 235), (35, 235), (35, 615), (60, 615)], CYAN, 2.2, dashed=True)
    text(c, 70, 222, "update predictors / estimator", 12.5, CYAN)

    # Small principle note.
    rounded_box(c, 60, 90, 235, 105, fill=WHITE, stroke=GRID, radius=18, line_width=1.5)
    text(c, 85, 160, "核心区别", 15, NAVY)
    text(c, 85, 135, "filter 处理硬约束，", 12.5, TEXT)
    text(c, 85, 113, "Double DQN 比较长期价值。", 12.5, TEXT)

    c.showPage()
    c.save()
    shutil.copy2(PDF_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    build()
