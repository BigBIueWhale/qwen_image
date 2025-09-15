#!/usr/bin/env python3
"""
Pre-normalize an image for Qwen2.5-VL via Ollama so Ollama's SmartResize is a no-op.

Reads:  input_image.png
Writes: output_image.png

Behavior (matches Ollama + Qwen guidance):
- Preserve aspect ratio.
- Snap width & height to multiples of 28 (>= 28 px).
- Cap total pixels to Ollama's observed default: 1,003,520 px (≈ 1280 * 28 * 28).
- Never upscale beyond what's needed to meet the 28-px minimum side.
- Emit a verbose, citation-backed audit of the decisions taken.

Citations echoed in logs:
- Qwen HF card: set min/max pixels; "values will be rounded to the nearest multiple of 28".
- Qwen GitHub (grounding): "height and width is 28*n"; absolute pixel coordinates.
- Ollama issue log: default vision cap "qwen25vl.vision.max_pixels default=1003520".
- Ollama user report: "multiple of 28" and "< 1M pixels" final image.

Dependencies: Pillow (PIL). Install with: pip install pillow
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from PIL import Image


# ---- Constants (kept explicit & documented) ---------------------------------

PIXEL_MULTIPLE: int = 28
MIN_SIDE: int = 28

# Ollama's observed default for Qwen2.5-VL images:
#   qwen25vl.vision.max_pixels default=1003520
# Ref: https://github.com/ollama/ollama/issues/11220 (log line)
OLLAMA_DEFAULT_MAX_PIXELS: int = 1_003_520

# Qwen guidance commonly shown in HF cards (examples):
#   min_pixels = 256 * 28 * 28
#   max_pixels = 1280 * 28 * 28
# We only need max-like behavior here to match Ollama's cap.
# Ref: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
#      https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct
DEFAULT_INPUT: Path = Path("input_image.png")
DEFAULT_OUTPUT: Path = Path("output_image.png")


# ---- Data model --------------------------------------------------------------

@dataclass(frozen=True)
class ImageSize:
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_tuple(self) -> Tuple[int, int]:
        return (self.width, self.height)


@dataclass(frozen=True)
class ResizeDecision:
    original: ImageSize
    tentative_scaled: ImageSize
    rounded: ImageSize
    final: ImageSize
    scale_x: float
    scale_y: float
    reason: str


# ---- Core logic --------------------------------------------------------------

def floor_to_multiple(value: float, multiple: int) -> int:
    """Floor a positive float to the nearest lower multiple of `multiple`."""
    if value < 0:
        raise ValueError("Expected non-negative value.")
    return max(multiple, (int(value) // multiple) * multiple)


def compute_target_size(
    orig: ImageSize,
    pixel_multiple: int = PIXEL_MULTIPLE,
    min_side: int = MIN_SIDE,
    max_pixels: int = OLLAMA_DEFAULT_MAX_PIXELS,
) -> ResizeDecision:
    """
    Reproduce Ollama/Qwen SmartResize constraints:

    - Keep aspect ratio (single uniform scale factor).
    - Enforce min side >= 28 (Qwen/Ollama will panic below this).
    - Enforce total pixels <= max_pixels (Ollama default ~1,003,520).
    - Snap both sides to multiples of 28 (Qwen docs: 28*n).
    - Prefer flooring on rounding so we never exceed the cap.

    Returns a detailed ResizeDecision including the rationale.
    """
    W0, H0 = orig.width, orig.height

    if W0 <= 0 or H0 <= 0:
        raise ValueError("Input image has non-positive dimensions.")

    reason_lines = []

    # 1) Enforce minimum side >= 28.
    # If either side < 28, we must upscale to satisfy Qwen/Ollama constraint.
    # Qwen GH (grounding): "resize … so height and width is 28*n"
    # https://github.com/QwenLM/Qwen2.5-VL/issues/721
    scale_for_min_side = 1.0
    if min(W0, H0) < min_side:
        scale_for_min_side = min_side / min(W0, H0)
        reason_lines.append(
            f"- Enforcing minimum side ≥ {min_side}px (Qwen: dimensions are 28*n)."
        )

    # 2) Enforce max pixels (Ollama default cap ≈ 1,003,520 px).
    # Ollama log: "qwen25vl.vision.max_pixels default=1003520"
    # https://github.com/ollama/ollama/issues/11220
    area_after_min = (W0 * scale_for_min_side) * (H0 * scale_for_min_side)
    scale_for_cap = 1.0
    if area_after_min > max_pixels:
        scale_for_cap = math.sqrt(max_pixels / area_after_min)
        reason_lines.append(
            f"- Capping total pixels ≤ {max_pixels:,} (Ollama default)."
        )

    # Use the *product* of both constraints (they are both uniform scaling).
    scale = scale_for_min_side * scale_for_cap

    # Avoid accidental upscaling if not needed (except to meet min_side).
    # (scale_for_cap ≤ 1.0 always; scale_for_min_side may be > 1.0)
    tentative_w = W0 * scale
    tentative_h = H0 * scale

    tentative = ImageSize(
        width=max(1, int(round(tentative_w))),
        height=max(1, int(round(tentative_h))),
    )

    # 3) Snap down to multiples of 28 to avoid exceeding the cap after rounding.
    # Qwen HF cards: "values will be rounded to the nearest multiple of 28".
    # We prefer floor to guarantee not exceeding max_pixels.
    rounded = ImageSize(
        width=floor_to_multiple(tentative.width, pixel_multiple),
        height=floor_to_multiple(tentative.height, pixel_multiple),
    )

    # Ensure we didn't round below the hard minimum.
    if rounded.width < min_side or rounded.height < min_side:
        # If floor pushed us below minimum, round up minimally to min_side (then to multiple).
        need_w = max(min_side, rounded.width)
        need_h = max(min_side, rounded.height)
        rounded = ImageSize(
            width=floor_to_multiple(max(need_w, min_side), pixel_multiple),
            height=floor_to_multiple(max(need_h, min_side), pixel_multiple),
        )

    # If—after rounding—we somehow exceed max_pixels, iteratively drop one step.
    final_w, final_h = rounded.width, rounded.height
    while final_w * final_h > max_pixels and (final_w >= pixel_multiple * 2 or final_h >= pixel_multiple * 2):
        # Reduce the longer side first by one 28-step.
        if final_w >= final_h and final_w > pixel_multiple:
            final_w -= pixel_multiple
        elif final_h > pixel_multiple:
            final_h -= pixel_multiple
        else:
            break

    final = ImageSize(width=final_w, height=final_h)

    # 4) Compute exact scales that map original -> final (what Ollama would apply).
    sx = final.width / W0
    sy = final.height / H0

    # Build rationale.
    reason_lines[:0] = [
        "SmartResize rationale:",
        f"- Orig: {W0}×{H0} ({W0*H0:,} px)",
        f"- Tentative (after min/cap): {tentative.width}×{tentative.height} (~{tentative.area:,} px)",
        f"- Rounded to 28-multiples: {rounded.width}×{rounded.height}",
        f"- Final: {final.width}×{final.height} ({final.area:,} px)",
        f"- Scale factors: sx={sx:.8f}, sy={sy:.8f} (aspect preserved).",
        "Doc snippets:",
        '  • Qwen HF: "Define min_pixels and max_pixels … values will be rounded to the nearest multiple of 28."',
        '  • Qwen GH: "resize the image so that the height and width is 28*n … absolute coordinates on the resized image."',
        '  • Ollama log: qwen25vl.vision.max_pixels default=1003520 (fallback cap).',
        '  • Ollama user: "resizes … multiple of 28 … final image has less than 1M pixels."',
    ]

    return ResizeDecision(
        original=orig,
        tentative_scaled=tentative,
        rounded=rounded,
        final=final,
        scale_x=sx,
        scale_y=sy,
        reason="\n".join(reason_lines),
    )


def main() -> int:
    in_path = DEFAULT_INPUT
    out_path = DEFAULT_OUTPUT

    # 0) Basic existence & format checks
    if not in_path.exists():
        print(f"[ERROR] Input file not found: {in_path.resolve()}", file=sys.stderr)
        return 2

    try:
        with Image.open(in_path) as im:
            im.load()  # force read to catch truncated files early
            if im.mode not in ("RGB", "RGBA", "L"):
                # Pillow will convert later; we warn but continue.
                print(f"[WARN] Input mode is {im.mode}; converting to RGB for processing.")

            orig_size = ImageSize(width=im.width, height=im.height)

            # Guardrails & flags
            too_small = (orig_size.width < MIN_SIDE) or (orig_size.height < MIN_SIDE)
            if too_small:
                print(f"[FLAG] Image has a side < {MIN_SIDE}px; Ollama/Qwen would error. "
                      f"We will upscale minimally to meet the {MIN_SIDE}px rule.")

            # 1) Compute target dims using our SmartResize replica.
            decision = compute_target_size(orig_size)

            # 2) Report
            print(decision.reason)

            # 3) Decide whether we need to resample
            if decision.final.width == orig_size.width and decision.final.height == orig_size.height:
                print("[OK] Image already satisfies Qwen/Ollama constraints. No resize performed.")
                # Still write a fresh file to match requested behavior.
                out_img = im.copy()
            else:
                # 4) Resize with high-quality filter.
                print(f"[DO] Resizing to {decision.final.width}×{decision.final.height} "
                      f"(LANCZOS; preserves aspect, 28-multiples, ≤ cap).")
                if im.mode not in ("RGB", "RGBA"):
                    im = im.convert("RGB")
                out_img = im.resize(decision.final.as_tuple(), resample=Image.Resampling.LANCZOS)

            # 5) Save deterministically
            out_img.save(out_path, format="PNG", optimize=True)
            print(f"[WRITE] {out_path} ({decision.final.width}×{decision.final.height})")
            print("[NOTE] Est. visual tokens ≈ (W*H)/(28*28) "
                  f"= {decision.final.area/(PIXEL_MULTIPLE*PIXEL_MULTIPLE):.2f}")

            # 6) Mapping hints (for your boxes)
            print("[MAP] To map model [x1,y1,x2,y2] back to original: divide x by sx and y by sy.")
            print(f"      sx = {decision.scale_x:.8f}  sy = {decision.scale_y:.8f}")
            print("      (Qwen uses absolute pixel coords on the resized image.)")

    except Exception as e:
        print(f"[ERROR] Failed to process image: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
