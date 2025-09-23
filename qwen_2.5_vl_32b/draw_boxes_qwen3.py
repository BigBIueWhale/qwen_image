#!/usr/bin/env python3
"""
Draw bounding boxes and labels from result.json onto output_image.png.
- Scales the image 4x BEFORE drawing (coordinates from JSON are out of 1000).
- Saves to bounding_boxes.png in the current working directory.
- Designed to run on Windows and Linux with only Pillow as a dependency.

If Pillow isn't installed:
    python -m pip install pillow
"""

from __future__ import annotations

import json
import math
import sys
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable, List, Sequence, Tuple, Union, Any, Optional

# Lazy import for Pillow with a clear error if missing.
try:
    from PIL import Image, ImageDraw, ImageFont, ImageColor
except Exception as e:  # pragma: no cover
    sys.stderr.write(
        "Error: This script requires the Pillow library.\n"
        "Install it with:  python -m pip install pillow\n"
        f"Details: {e}\n"
    )
    sys.exit(1)


# ---------- Types ----------
Int = int
Float = float
RGBA = Tuple[Int, Int, Int, Int]
XYXY = Tuple[Int, Int, Int, Int]

@dataclass(frozen=True)
class BBox:
    x1: Int
    y1: Int
    x2: Int
    y2: Int

    @staticmethod
    def from_seq(seq: Sequence[Union[int, float]]) -> "BBox":
        if len(seq) != 4:
            raise ValueError(f"bbox_2d must have 4 elements, got {len(seq)}")
        x1, y1, x2, y2 = (int(seq[0]), int(seq[1]), int(seq[2]), int(seq[3]))
        if x2 < x1 or y2 < y1:
            raise ValueError(f"Invalid bbox with negative size: {(x1, y1, x2, y2)}")
        return BBox(x1, y1, x2, y2)

    def scale(self, factor: Int) -> "BBox":
        return BBox(
            x1=self.x1 * factor,
            y1=self.y1 * factor,
            x2=self.x2 * factor,
            y2=self.y2 * factor,
        )

    def clamp(self, width: Int, height: Int) -> "BBox":
        x1 = max(0, min(self.x1, width - 1))
        y1 = max(0, min(self.y1, height - 1))
        x2 = max(0, min(self.x2, width - 1))
        y2 = max(0, min(self.y2, height - 1))
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1
        return BBox(x1, y1, x2, y2)

    def to_xyxy(self) -> XYXY:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_pixels(self, width: Int, height: Int, base: Int = 1000) -> "BBox":
        return BBox(
            x1=int(round(self.x1 * width / base)),
            y1=int(round(self.y1 * height / base)),
            x2=int(round(self.x2 * width / base)),
            y2=int(round(self.y2 * height / base)),
        )


@dataclass(frozen=True)
class Detection:
    label: str
    bbox: BBox


# ---------- Constants ----------
INPUT_IMAGE: Final[Path] = Path("output_image.png")
INPUT_JSON: Final[Path] = Path("result.json")
OUTPUT_IMAGE: Final[Path] = Path("bounding_boxes.png")
SCALE: Final[Int] = 4


# ---------- Helpers ----------
def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_detections(raw: Any) -> List[Detection]:
    if not isinstance(raw, list):
        raise ValueError("JSON root must be a list.")
    out: List[Detection] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Item {idx} is not an object.")
        label = item.get("label")
        bbox_seq = item.get("bbox_2d")
        if not isinstance(label, str):
            raise ValueError(f"Item {idx} has invalid 'label'.")
        if not isinstance(bbox_seq, Sequence):
            raise ValueError(f"Item {idx} has invalid 'bbox_2d'.")
        bbox = BBox.from_seq(bbox_seq)
        out.append(Detection(label=label, bbox=bbox))
    return out


def try_load_font(point_size: Int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    Try multiple sensible fonts. Fall back to PIL's default if none found.
    """
    # 1) Try DejaVuSans (often bundled with Pillow)
    try:
        return ImageFont.truetype("DejaVuSans.ttf", point_size)
    except Exception:
        pass

    # 2) Try common Windows font
    try:
        return ImageFont.truetype("arial.ttf", point_size)
    except Exception:
        pass

    # 3) Try Noto Sans (common on Linux)
    try:
        return ImageFont.truetype("NotoSans-Regular.ttf", point_size)
    except Exception:
        pass

    # 4) Last resort
    return ImageFont.load_default()


def luminance(rgb: Tuple[Int, Int, Int]) -> Float:
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def text_fill_for_bg(bg_rgb: Tuple[Int, Int, Int]) -> RGBA:
    # Choose white or black text for readability against bg color.
    return (0, 0, 0, 255) if luminance(bg_rgb) > 160.0 else (255, 255, 255, 255)


def color_for_label(label: str) -> Tuple[Int, Int, Int]:
    """
    Deterministic, bright-ish color based on label string.
    """
    h = hashlib.sha256(label.encode("utf-8")).hexdigest()
    # Map to RGB with a minimum channel value to avoid too-dark colors.
    r = 100 + (int(h[0:2], 16) % 156)
    g = 100 + (int(h[2:4], 16) % 156)
    b = 100 + (int(h[4:6], 16) % 156)
    return (r, g, b)


def draw_labelled_box(
    draw: ImageDraw.ImageDraw,
    box: BBox,
    label: str,
    font: ImageFont.ImageFont,
    stroke: Int,
) -> None:
    color_rgb = color_for_label(label)
    # Rectangle
    draw.rectangle(box.to_xyxy(), outline=color_rgb, width=stroke)

    # Text background and text
    padding: Int = max(2, stroke)
    # Compute text size
    # Using textbbox for accurate sizing (Pillow >= 8.0), fallback to textsize.
    try:
        tb = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = (tb[2] - tb[0], tb[3] - tb[1])
    except Exception:
        text_w, text_h = draw.textsize(label, font=font)

    # Try to place above the box; if not enough space, place below.
    bg_x1: Int = box.x1
    bg_y1: Int = box.y1 - text_h - 2 * padding
    if bg_y1 < 0:
        bg_y1 = box.y1 + 1  # below the top edge
    bg_x2: Int = bg_x1 + text_w + 2 * padding
    bg_y2: Int = bg_y1 + text_h + 2 * padding

    # Ensure background stays within image bounds
    img_w, img_h = draw.im.size  # type: ignore[attr-defined]
    if bg_x2 > img_w:
        shift: Int = bg_x2 - img_w
        bg_x1 -= shift
        bg_x2 -= shift
    if bg_y2 > img_h:
        shift = bg_y2 - img_h
        bg_y1 -= shift
        bg_y2 -= shift

    # Draw background
    text_bg: RGBA = (*color_rgb, 180)
    draw.rectangle([(bg_x1, bg_y1), (bg_x2, bg_y2)], fill=text_bg, outline=None)

    # Draw text
    text_color: RGBA = text_fill_for_bg(color_rgb)
    draw.text((bg_x1 + padding, bg_y1 + padding), label, fill=text_color, font=font)


def main() -> int:
    # Validate inputs
    if not INPUT_IMAGE.exists():
        sys.stderr.write(f"Error: {INPUT_IMAGE} not found in {Path.cwd()}\n")
        return 2
    if not INPUT_JSON.exists():
        sys.stderr.write(f"Error: {INPUT_JSON} not found in {Path.cwd()}\n")
        return 3

    # Load data
    try:
        detections_raw: Any = load_json(INPUT_JSON)
        detections: List[Detection] = parse_detections(detections_raw)
    except Exception as e:
        sys.stderr.write(f"Failed to read/parse {INPUT_JSON}: {e}\n")
        return 4

    try:
        with Image.open(INPUT_IMAGE) as im_orig:
            im_orig = im_orig.convert("RGBA")
            w0, h0 = im_orig.size
            # Scale up BEFORE drawing so text renders nicely.
            w1, h1 = (w0 * SCALE, h0 * SCALE)
            im_scaled = im_orig.resize((w1, h1), resample=Image.Resampling.LANCZOS)
    except Exception as e:
        sys.stderr.write(f"Failed to open/scale {INPUT_IMAGE}: {e}\n")
        return 5

    # Prepare drawing context
    draw = ImageDraw.Draw(im_scaled, mode="RGBA")
    stroke: Int = max(2, SCALE)  # line thickness
    # Font size relative to scale and image size
    # Aim for readable labels without overwhelming the image.
    nominal: Int = max(12, int(0.018 * max(w1, h1)))
    font: ImageFont.ImageFont = try_load_font(nominal)

    # Draw each detection
    for det in detections:
        box_scaled: BBox = det.bbox.to_pixels(w1, h1, base=1000).clamp(w1, h1)
        draw_labelled_box(draw, box_scaled, det.label, font, stroke)

    # Save output
    try:
        im_scaled.save(OUTPUT_IMAGE)
        print(f"Saved: {OUTPUT_IMAGE.resolve()}")
    except Exception as e:
        sys.stderr.write(f"Failed to save {OUTPUT_IMAGE}: {e}\n")
        return 6

    return 0


if __name__ == "__main__":
    sys.exit(main())
