#!/usr/bin/env python3
"""Generate the Layer-Scope taxonomy figure used in Section 2.3."""

from pathlib import Path
import gc
import shutil
import subprocess
import time

import pypdfium2 as pdfium
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "figures" / "ch2_layer_scope_taxonomy.pdf"
PNG_PATH = ROOT / "figures" / "ch2_layer_scope_taxonomy.png"

WIDTH = 1200
HEIGHT = 820

NAVY = "#102A72"
BLUE = "#3768E8"
CYAN = "#1598A5"
RED = "#EF3E36"
GREEN = "#4B9662"
ORANGE = "#E58A2B"
PURPLE = "#7760C8"
TEXT = "#17213A"
MUTED = "#53627B"
GRID = "#B8C8EA"
PALE_BLUE = "#EEF4FF"
PALE_CYAN = "#EAF8F8"
PALE_GREEN = "#EDF7EF"
PALE_ORANGE = "#FFF4E7"
PALE_PURPLE = "#F2EEFC"
WHITE = "#FFFFFF"


def rounded_box(c, x, y, w, h, stroke=BLUE, fill=WHITE, radius=14, width=2):
    c.setLineWidth(width)
    c.setStrokeColor(stroke)
    c.setFillColor(fill)
    c.roundRect(x, y, w, h, radius, fill=1, stroke=1)


def label(c, x, y, value, size=18, color=TEXT, bold=False, align="left"):
    c.setFillColor(color)
    c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    if align == "center":
        c.drawCentredString(x, y, value)
    elif align == "right":
        c.drawRightString(x, y, value)
    else:
        c.drawString(x, y, value)


def bullet_lines(c, x, y, lines, size=16, leading=27, color=TEXT):
    for index, line in enumerate(lines):
        yy = y - index * leading
        c.setFillColor(CYAN if index % 2 == 0 else BLUE)
        c.circle(x, yy + 5, 4, fill=1, stroke=0)
        label(c, x + 14, yy, line, size=size, color=color)


def arrow(c, x1, y1, x2, y2, color=RED, width=4, dashed=False):
    c.saveState()
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(width)
    if dashed:
        c.setDash(10, 8)
    c.line(x1, y1, x2, y2)
    if abs(x2 - x1) >= abs(y2 - y1):
        direction = 1 if x2 >= x1 else -1
        points = [
            (x2, y2),
            (x2 - direction * 14, y2 + 8),
            (x2 - direction * 14, y2 - 8),
        ]
    else:
        direction = 1 if y2 >= y1 else -1
        points = [
            (x2, y2),
            (x2 - 8, y2 - direction * 14),
            (x2 + 8, y2 - direction * 14),
        ]
    path = c.beginPath()
    path.moveTo(*points[0])
    path.lineTo(*points[1])
    path.lineTo(*points[2])
    path.close()
    c.drawPath(path, fill=1, stroke=0)
    c.restoreState()


def draw_model_icon(c, cx, cy):
    c.setStrokeColor(GREEN)
    c.setFillColor(PALE_GREEN)
    c.setLineWidth(3)
    for x1, y1, x2, y2 in [
        (cx - 38, cy, cx, cy + 28),
        (cx - 38, cy, cx, cy - 28),
        (cx, cy + 28, cx + 38, cy),
        (cx, cy - 28, cx + 38, cy),
        (cx - 38, cy, cx + 38, cy),
    ]:
        c.line(x1, y1, x2, y2)
    for x, y in [
        (cx - 38, cy),
        (cx, cy + 28),
        (cx, cy - 28),
        (cx + 38, cy),
    ]:
        c.circle(x, y, 9, fill=1, stroke=1)


def draw_chip_icon(c, cx, cy):
    c.setStrokeColor(ORANGE)
    c.setFillColor(PALE_ORANGE)
    c.setLineWidth(3)
    c.roundRect(cx - 34, cy - 34, 68, 68, 8, fill=1, stroke=1)
    c.setFillColor(ORANGE)
    for dx in (-17, 0, 17):
        for dy in (-17, 0, 17):
            c.rect(cx + dx - 5, cy + dy - 5, 10, 10, fill=1, stroke=0)


def draw_runtime_icon(c, cx, cy):
    c.setStrokeColor(BLUE)
    c.setFillColor(PALE_BLUE)
    c.setLineWidth(3)
    c.roundRect(cx - 46, cy - 23, 92, 46, 10, fill=1, stroke=1)
    for dx in (-28, -9, 10, 29):
        c.setFillColor(CYAN if dx < 20 else WHITE)
        c.rect(cx + dx - 6, cy - 6, 12, 12, fill=1, stroke=1)
    arrow(c, cx - 30, cy + 40, cx + 30, cy + 40, color=BLUE, width=3)


def draw_routing_icon(c, cx, cy):
    c.setStrokeColor(PURPLE)
    c.setFillColor(PALE_PURPLE)
    c.setLineWidth(3)
    c.circle(cx - 40, cy, 10, fill=1, stroke=1)
    for offset in (32, 0, -32):
        c.line(cx - 28, cy, cx + 28, cy + offset)
        c.circle(cx + 40, cy + offset, 10, fill=1, stroke=1)


def draw_layer_cell(c, x, y, w, h, title, question, fill, accent, icon):
    rounded_box(c, x, y, w, h, stroke=accent, fill=fill, radius=16, width=2.5)
    icon(c, x + 75, y + h / 2)
    title_lines = title if isinstance(title, list) else [title]
    for index, line in enumerate(title_lines):
        label(
            c,
            x + 140,
            y + h - 42 - index * 23,
            line,
            size=20,
            color=NAVY,
            bold=True,
        )
    question_lines = question if isinstance(question, list) else [question]
    for index, line in enumerate(question_lines):
        label(
            c,
            x + 140,
            y + h - 50 - len(title_lines) * 23 - index * 23,
            line,
            size=15,
            color=MUTED,
        )


def draw_scope_cell(c, x, y, w, h, title, lines, accent=BLUE):
    rounded_box(c, x, y, w, h, stroke=GRID, fill=WHITE, radius=14, width=1.8)
    label(c, x + 22, y + h - 35, title, size=17, color=accent, bold=True)
    bullet_lines(c, x + 28, y + h - 69, lines, size=15, leading=24)


def draw_runtime_scope(c, x, y, w, h, distributed=False):
    rounded_box(c, x, y, w, h, stroke=GRID, fill=WHITE, radius=14, width=1.8)
    entries = (
        [
            ("KV / Memory", "Distributed KV, migration, remote reuse", CYAN),
            ("Batching / Scheduling", "Cross-worker queues and PD pools", BLUE),
            ("Decoding", "Draft-target split and verification", PURPLE),
            ("Execution", "TP / PP / EP and communication overlap", ORANGE),
        ]
        if distributed
        else [
            ("KV / Memory", "Paging, prefix reuse, eviction, offload", CYAN),
            ("Batching / Scheduling", "Continuous batch and chunked prefill", BLUE),
            ("Decoding", "Speculative, tree and parallel decoding", PURPLE),
            ("Execution", "Graph replay and compute-copy overlap", ORANGE),
        ]
    )
    box_h = 47
    gap = 6
    top = y + h - 18
    for index, (title, detail, accent) in enumerate(entries):
        yy = top - (index + 1) * box_h - index * gap
        c.setFillColor("#F8FAFE")
        c.setStrokeColor(accent)
        c.setLineWidth(1.5)
        c.roundRect(x + 16, yy, w - 32, box_h, 9, fill=1, stroke=1)
        label(c, x + 28, yy + 27, title, size=16, color=accent, bold=True)
        label(c, x + 28, yy + 9, detail, size=14, color=TEXT)


def draw_axis_cell(c, x, y, w, h, title, question, fill, accent, tags=None):
    rounded_box(c, x, y, w, h, stroke=accent, fill=fill, radius=14, width=2.2)
    title_lines = title if isinstance(title, list) else [title]
    for index, line in enumerate(title_lines):
        label(c, x + 22, y + h - 34 - index * 21, line, size=19, color=NAVY, bold=True)
    for index, line in enumerate(question):
        label(
            c,
            x + 22,
            y + h - 66 - (len(title_lines) - 1) * 18 - index * 23,
            line,
            size=15,
            color=MUTED,
        )
    if tags:
        tag_w = (w - 54) / 2
        for index, (tag, color) in enumerate(tags):
            column = index % 2
            row = index // 2
            xx = x + 22 + column * (tag_w + 10)
            yy = y + 23 + (1 - row) * 38
            c.setFillColor(WHITE)
            c.setStrokeColor(color)
            c.setLineWidth(1.5)
            c.roundRect(xx, yy, tag_w, 29, 7, fill=1, stroke=1)
            label(c, xx + tag_w / 2, yy + 9, tag, size=12.5, color=color, bold=True, align="center")


def build():
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(PDF_PATH), pagesize=(WIDTH, HEIGHT))
    c.setTitle("LLM Inference Optimization: Layer-Scope Taxonomy")
    c.setFillColor(WHITE)
    c.rect(0, 0, WIDTH, HEIGHT, fill=1, stroke=0)

    axis_x, axis_w = 62, 275
    local_x, local_w = 352, 397
    dist_x, dist_w = 764, 407

    rounded_box(c, axis_x, 750, axis_w, 48, stroke=NAVY, fill=PALE_BLUE)
    rounded_box(c, local_x, 750, local_w, 48, stroke=BLUE, fill=PALE_CYAN)
    rounded_box(c, dist_x, 750, dist_w, 48, stroke=BLUE, fill=PALE_PURPLE)
    label(c, axis_x + axis_w / 2, 775, "PRIMARY AXIS", 17, NAVY, True, "center")
    label(c, axis_x + axis_w / 2, 758, "choose by optimization object", 12, MUTED, False, "center")
    label(c, local_x + local_w / 2, 775, "LOCAL", 19, NAVY, True, "center")
    label(c, local_x + local_w / 2, 757, "one serving instance / control domain", 12, MUTED, False, "center")
    label(c, dist_x + dist_w / 2, 775, "DISTRIBUTED", 19, NAVY, True, "center")
    label(c, dist_x + dist_w / 2, 757, "across devices, workers, replicas or nodes", 12, MUTED, False, "center")

    arrow(c, local_x + 70, 730, dist_x + dist_w - 70, 730, color=RED, width=3)
    label(c, (local_x + dist_x + dist_w) / 2, 736, "execution scope expands across entities", 13, RED, True, "center")

    rows = {
        "serving": (585, 130),
        "runtime": (335, 235),
        "kernel": (185, 135),
        "model": (35, 135),
    }

    spine_x = 42
    arrow(c, spine_x, 96, spine_x, 690, color=RED, width=4)
    c.saveState()
    c.translate(18, 390)
    c.rotate(90)
    label(c, 0, 0, "higher-level control decisions", 12, RED, True, "center")
    c.restoreState()

    row_meta = [
        ("model", "1", GREEN),
        ("kernel", "2", ORANGE),
        ("runtime", "3", BLUE),
        ("serving", "4", PURPLE),
    ]
    for key, number, color in row_meta:
        y, h = rows[key]
        cy = y + h / 2
        c.setFillColor(WHITE)
        c.setStrokeColor(color)
        c.setLineWidth(3)
        c.circle(spine_x, cy, 15, fill=1, stroke=1)
        label(c, spine_x, cy - 6, number, 14, color, True, "center")

    y, h = rows["serving"]
    draw_axis_cell(
        c, axis_x, y, axis_w, h,
        ["SERVING POLICY", "& ROUTING"],
        ["Decision object:", "which model, worker or service path?"],
        PALE_PURPLE, PURPLE,
    )
    draw_scope_cell(
        c, local_x, y, local_w, h, "Single instance / local control domain",
        ["Model / adapter / drafter selection", "Local cascade, fallback and path policy"],
        PURPLE,
    )
    draw_scope_cell(
        c, dist_x, y, dist_w, h, "Across replicas, pools, services or regions",
        ["Replica / pool / region routing", "KV-aware routing and service orchestration"],
        PURPLE,
    )

    y, h = rows["runtime"]
    draw_axis_cell(
        c, axis_x, y, axis_w, h,
        "RUNTIME",
        ["Decision object:", "online request and token execution"],
        PALE_BLUE, BLUE,
        tags=[
            ("KV / Memory", CYAN),
            ("Batch / Schedule", BLUE),
            ("Decoding", PURPLE),
            ("Execution", ORANGE),
        ],
    )
    draw_runtime_scope(c, local_x, y, local_w, h, distributed=False)
    draw_runtime_scope(c, dist_x, y, dist_w, h, distributed=True)

    y, h = rows["kernel"]
    draw_axis_cell(
        c, axis_x, y, axis_w, h,
        "KERNEL",
        ["Decision object:", "operator / communication primitive"],
        PALE_ORANGE, ORANGE,
    )
    draw_scope_cell(
        c, local_x, y, local_w, h, "Within one device / host",
        ["FlashAttention, fused GEMM / RMSNorm", "Low-bit kernels, tiling, memory layout"],
        ORANGE,
    )
    draw_scope_cell(
        c, dist_x, y, dist_w, h, "Across devices",
        ["All-reduce / all-to-all / P2P", "RDMA / NVLink paths, fused collectives"],
        ORANGE,
    )

    y, h = rows["model"]
    draw_axis_cell(
        c, axis_x, y, axis_w, h,
        "MODEL",
        ["Decision object:", "parameters, structure and semantics"],
        PALE_GREEN, GREEN,
    )
    draw_scope_cell(
        c, local_x, y, local_w, h, "Model remains locally executable",
        ["Quantization, pruning, distillation", "GQA / MQA / MLA, sparse attention"],
        GREEN,
    )
    draw_scope_cell(
        c, dist_x, y, dist_w, h, "Architecture supports partitioned execution",
        ["MoE / expert-friendly structure", "Partitionable attention, KV, model family"],
        GREEN,
    )

    c.showPage()
    c.save()
    del c
    gc.collect()
    time.sleep(0.2)

    PNG_PATH.unlink(missing_ok=True)
    if shutil.which("sips"):
        subprocess.run(
            ["sips", "-s", "format", "png", str(PDF_PATH), "--out", str(PNG_PATH)],
            check=True,
            capture_output=True,
        )
    else:
        document = pdfium.PdfDocument(str(PDF_PATH))
        document[0].render(scale=2.0).to_pil().save(PNG_PATH)


if __name__ == "__main__":
    build()
