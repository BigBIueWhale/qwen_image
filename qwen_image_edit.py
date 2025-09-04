# https://huggingface.co/DFloat11/Qwen-Image-Edit-DF11/blob/4f8f85749619ef78ae4c56dc912480e559573813/README.md
# Qwen-Image-Edit with DFloat11 compression
#
# This script mirrors the overall logic of qwen_image.py:
#  - outputs sequential files to ./output_edit (000001.png, 000002.png, ...)
#  - loops until CTRL+C to generate more edited variants
#  - supports CPU offloading options (DFloat11 + pipeline)
#
# Example:
#   python qwen_image_edit.py "Add a hat to the cat." \
#       --image https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png
#
#   python qwen_image_edit.py "Turn the sky pink" --cpu_offload --cpu_offload_blocks 30 --no_pin_memory \
#       --image ./my_photo.jpg

import argparse
import os
import re
from pathlib import Path

import torch
from diffusers.utils import load_image
from diffusers import QwenImageTransformer2DModel, QwenImageEditPipeline
from transformers.modeling_utils import no_init_weights
from dfloat11 import DFloat11Model


# ------------------------
# Arg parsing (prompt is positional like qwen_image.py)
# ------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Edit images with Qwen-Image-Edit (loops until CTRL+C)"
    )
    # Offload & memory controls
    parser.add_argument("--cpu_offload", action="store_true", help="Enable CPU offloading")
    parser.add_argument(
        "--cpu_offload_blocks",
        type=int,
        default=None,
        help="Number of transformer blocks to offload to CPU (higher = less GPU VRAM, more CPU RAM)",
    )
    parser.add_argument("--no_pin_memory", action="store_true", help="Disable memory pinning")

    # Editing controls
    parser.add_argument(
        "image",
        type=str,
        nargs="?",
        default="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png",
        help="Path or URL of the input image",
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="Text prompt describing the edit (e.g., 'Add a hat to the cat.')",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=" ",
        help="Negative prompt to steer away from undesired content",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--true_cfg_scale",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (each image increments by +1)",
    )

    return parser.parse_args()


args = parse_args()

MODEL_ID = "Qwen/Qwen-Image-Edit"
OUTPUT_DIR = Path.cwd() / "output_edit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FNAME_RE = re.compile(r"^(\d{6})\.png$")


# ------------------------
# Helpers for sequential filenames
# ------------------------

def _next_index() -> int:
    max_idx = 0
    for name in os.listdir(OUTPUT_DIR):
        m = FNAME_RE.match(name)
        if m:
            try:
                idx = int(m.group(1))
                if idx > max_idx:
                    max_idx = idx
            except ValueError:
                pass
    return max_idx + 1 if max_idx >= 1 else 1


# ------------------------
# Load DFloat11-compressed transformer and pipeline
# ------------------------
with no_init_weights():
    transformer = QwenImageTransformer2DModel.from_config(
        QwenImageTransformer2DModel.load_config(
            MODEL_ID,
            subfolder="transformer",
        ),
    ).to(torch.bfloat16)

DFloat11Model.from_pretrained(
    "DFloat11/Qwen-Image-Edit-DF11",
    device="cpu",
    cpu_offload=args.cpu_offload,
    cpu_offload_blocks=args.cpu_offload_blocks,
    pin_memory=not args.no_pin_memory,
    bfloat16_model=transformer,
)

pipe = QwenImageEditPipeline.from_pretrained(
    MODEL_ID,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
pipe.set_progress_bar_config(disable=None)

# Pre-load the conditioning image once (URL or local path)
source_image = load_image(args.image)

# ------------------------
# Continuous editing loop (CTRL+C to stop)
# ------------------------
idx = _next_index()
images_made = 0

gen_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Run] Output directory: {OUTPUT_DIR}")
print(f"[Run] Starting filename index: {idx:06d}")
print(f"[Run] CUDA available: {torch.cuda.is_available()}")
print("[Run] Press CTRL+C to stop.")

try:
    while True:
        current_seed = args.seed + images_made
        generator = torch.Generator(device=gen_device).manual_seed(current_seed)

        with torch.inference_mode():
            result = pipe(
                image=source_image,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.num_inference_steps,
                true_cfg_scale=args.true_cfg_scale,
                generator=generator,
            )
            out_img = result.images[0]

        fname = f"{idx:06d}.png"
        fpath = OUTPUT_DIR / fname
        out_img.save(fpath)

        if torch.cuda.is_available():
            max_mem = torch.cuda.max_memory_allocated()
            print(
                f"[OK] Saved {fname} | seed={current_seed} | Max memory: {max_mem / (1000 ** 3):.2f} GB"
            )
        else:
            print(f"[OK] Saved {fname} | seed={current_seed}")

        idx += 1
        images_made += 1

except KeyboardInterrupt:
    print("\n[Stop] CTRL+C detected. Cleaning up...")

finally:
    try:
        del pipe
        del transformer
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    print(f"[Done] Generated {images_made} edited image(s).")
