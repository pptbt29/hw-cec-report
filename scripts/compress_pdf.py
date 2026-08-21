#!/usr/bin/env python3
"""Compress a PDF and write <name>_compressed.pdf next to it.

Usage:
    python3 scripts/compress_pdf.py input.pdf
    python3 scripts/compress_pdf.py input.pdf --quality screen
    python3 scripts/compress_pdf.py input.pdf --dpi 150

The script uses Ghostscript (`gs`) and keeps the original PDF unchanged.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


QUALITY_SETTINGS = {
    "screen": "/screen",
    "ebook": "/ebook",
    "printer": "/printer",
    "prepress": "/prepress",
}


def output_path_for(input_pdf: Path) -> Path:
    return input_pdf.with_name(f"{input_pdf.stem}_compressed{input_pdf.suffix}")


def human_size(path: Path) -> str:
    size = path.stat().st_size
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def build_gs_command(input_pdf: Path, output_pdf: Path, quality: str, dpi: int | None) -> list[str]:
    cmd = [
        "gs",
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.5",
        "-dNOPAUSE",
        "-dQUIET",
        "-dBATCH",
        f"-sOutputFile={output_pdf}",
    ]

    if dpi is None:
        cmd.append(f"-dPDFSETTINGS={QUALITY_SETTINGS[quality]}")
    else:
        cmd.extend(
            [
                "-dDownsampleColorImages=true",
                "-dDownsampleGrayImages=true",
                "-dDownsampleMonoImages=true",
                "-dColorImageDownsampleType=/Bicubic",
                "-dGrayImageDownsampleType=/Bicubic",
                "-dMonoImageDownsampleType=/Subsample",
                f"-dColorImageResolution={dpi}",
                f"-dGrayImageResolution={dpi}",
                f"-dMonoImageResolution={max(dpi, 300)}",
            ]
        )

    cmd.append(str(input_pdf))
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Compress a PDF with Ghostscript.")
    parser.add_argument("pdf", help="Input PDF file")
    parser.add_argument(
        "--quality",
        choices=sorted(QUALITY_SETTINGS),
        default="ebook",
        help="Ghostscript quality preset. Default: ebook",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help="Optional image DPI override. Useful for scanned PDFs, e.g. --dpi 150",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing *_compressed.pdf",
    )
    args = parser.parse_args()

    input_pdf = Path(args.pdf).expanduser().resolve()
    if not input_pdf.exists():
        print(f"Error: file not found: {input_pdf}", file=sys.stderr)
        return 1
    if input_pdf.suffix.lower() != ".pdf":
        print(f"Error: input file must be a PDF: {input_pdf}", file=sys.stderr)
        return 1

    gs = shutil.which("gs")
    if gs is None:
        print("Error: Ghostscript `gs` was not found in PATH.", file=sys.stderr)
        print("Install it first, e.g. `brew install ghostscript` on macOS.", file=sys.stderr)
        return 1

    output_pdf = output_path_for(input_pdf)
    if output_pdf.exists() and not args.overwrite:
        print(f"Error: output already exists: {output_pdf}", file=sys.stderr)
        print("Use --overwrite to replace it.", file=sys.stderr)
        return 1

    cmd = build_gs_command(input_pdf, output_pdf, args.quality, args.dpi)
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("Error: Ghostscript compression failed.", file=sys.stderr)
        if result.stderr:
            print(result.stderr.strip(), file=sys.stderr)
        return result.returncode

    print(f"Input : {input_pdf} ({human_size(input_pdf)})")
    print(f"Output: {output_pdf} ({human_size(output_pdf)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
