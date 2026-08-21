#!/usr/bin/env python3
"""
Normalize page sizes of an already-generated PDF.

This script works only at the PDF level. It rasterizes each page, optionally
auto-crops visual white/blank margins, then places the page image onto a fixed
canvas. It is designed for PDFs where some pages are extremely wide and others
are narrow, causing inconsistent visual scale when viewed continuously.

Output: <input_stem>_normalized.pdf

Examples:
  python scripts/normalize_pdf_pages.py input.pdf
  python scripts/normalize_pdf_pages.py input.pdf --paper a4 --orientation portrait
  python scripts/normalize_pdf_pages.py input.pdf --paper 16:9 --orientation landscape
  python scripts/normalize_pdf_pages.py input.pdf --crop-mode none
"""

from __future__ import annotations

import argparse
import sys
from io import BytesIO
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageChops, ImageOps
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


POINTS_PER_INCH = 72


PAPER_SIZES_PT = {
    "a4": (595.276, 841.89),
    "letter": (612.0, 792.0),
    "16:9": (960.0, 540.0),
    "4:3": (960.0, 720.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize visual page sizes of a PDF by cropping and padding pages to a fixed canvas."
    )
    parser.add_argument("input_pdf", type=Path, help="Input PDF file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PDF file. Default: <input_stem>_normalized.pdf.",
    )
    parser.add_argument(
        "--paper",
        choices=sorted(PAPER_SIZES_PT),
        default="a4",
        help="Target page canvas. Default: a4.",
    )
    parser.add_argument(
        "--orientation",
        choices=["portrait", "landscape"],
        default="portrait",
        help="Target page orientation. Default: portrait.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Rasterization DPI. Higher is sharper but larger. Default: 180.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=36.0,
        help="Target page margin in PDF points. 36 pt = 0.5 inch. Default: 36.",
    )
    parser.add_argument(
        "--crop-mode",
        choices=["auto", "none"],
        default="auto",
        help="auto: crop blank visual borders; none: use full original page image. Default: auto.",
    )
    parser.add_argument(
        "--background-threshold",
        type=int,
        default=245,
        help="Pixels brighter than this are treated as blank in auto-crop. Default: 245.",
    )
    parser.add_argument(
        "--min-crop-area",
        type=float,
        default=0.03,
        help="If detected content is smaller than this page area, keep full page to avoid over-cropping. Default: 0.03.",
    )
    parser.add_argument(
        "--max-upscale",
        type=float,
        default=1.4,
        help="Maximum enlargement of a cropped page image. Use 1.0 to forbid upscaling. Default: 1.4.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=88,
        help="JPEG quality for embedded page images. Default: 88.",
    )
    return parser.parse_args()


def target_size(paper: str, orientation: str) -> tuple[float, float]:
    width, height = PAPER_SIZES_PT[paper]
    if orientation == "landscape" and height > width:
        width, height = height, width
    if orientation == "portrait" and width > height:
        width, height = height, width
    return width, height


def render_page(page: pdfium.PdfPage, dpi: int) -> Image.Image:
    scale = dpi / POINTS_PER_INCH
    bitmap = page.render(scale=scale)
    image = bitmap.to_pil()
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def content_bbox(image: Image.Image, threshold: int, min_area_ratio: float) -> tuple[int, int, int, int]:
    """
    Detect non-blank visual content.

    This intentionally uses a simple brightness threshold because it is robust
    for scans, screenshots, vector pages rendered to white, and mixed PDFs.
    If detection becomes too small, keep the full page to avoid exploding a tiny
    text region into a huge page.
    """
    gray = ImageOps.grayscale(image)
    blank = Image.new("L", gray.size, 255)
    diff = ImageChops.difference(gray, blank)
    mask = diff.point(lambda p: 255 if p > (255 - threshold) else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return (0, 0, image.width, image.height)

    left, top, right, bottom = bbox
    detected_area = max(1, right - left) * max(1, bottom - top)
    full_area = image.width * image.height
    if detected_area / full_area < min_area_ratio:
        return (0, 0, image.width, image.height)
    return bbox


def page_to_jpeg(image: Image.Image, quality: int) -> ImageReader:
    """
    ReportLab can embed PIL images directly, but converting to RGB first avoids
    alpha and palette surprises.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    buffer.seek(0)
    return ImageReader(buffer)


def normalize_pdf(args: argparse.Namespace) -> Path:
    input_pdf = args.input_pdf.expanduser().resolve()
    if not input_pdf.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_pdf}")
    if input_pdf.suffix.lower() != ".pdf":
        raise ValueError(f"Input file must be a PDF: {input_pdf}")

    output_pdf = args.output
    if output_pdf is None:
        output_pdf = input_pdf.with_name(f"{input_pdf.stem}_normalized.pdf")
    else:
        output_pdf = output_pdf.expanduser().resolve()

    page_width, page_height = target_size(args.paper, args.orientation)
    usable_width = page_width - 2 * args.margin
    usable_height = page_height - 2 * args.margin
    if usable_width <= 0 or usable_height <= 0:
        raise ValueError("Margin is too large for the chosen paper size.")

    pdf = pdfium.PdfDocument(str(input_pdf))
    out = canvas.Canvas(str(output_pdf), pagesize=(page_width, page_height))

    for index in range(len(pdf)):
        page = pdf[index]
        image = render_page(page, args.dpi)
        original_width, original_height = image.size

        if args.crop_mode == "auto":
            bbox = content_bbox(image, args.background_threshold, args.min_crop_area)
            image = image.crop(bbox)

        image_width, image_height = image.size
        fit_scale = min(usable_width / image_width, usable_height / image_height)
        scale = min(fit_scale, args.max_upscale)

        draw_width = image_width * scale
        draw_height = image_height * scale
        x = (page_width - draw_width) / 2
        y = (page_height - draw_height) / 2

        out.setFillColorRGB(1, 1, 1)
        out.rect(0, 0, page_width, page_height, fill=1, stroke=0)
        out.drawImage(
            page_to_jpeg(image, args.quality),
            x,
            y,
            width=draw_width,
            height=draw_height,
            preserveAspectRatio=True,
            mask=None,
        )
        out.showPage()

        print(
            f"page {index + 1}: {original_width}x{original_height}px -> "
            f"{image_width}x{image_height}px -> {page_width:.0f}x{page_height:.0f}pt"
        )

    out.save()
    pdf.close()
    return output_pdf


def main() -> int:
    args = parse_args()
    try:
        output = normalize_pdf(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
