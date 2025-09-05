# https://huggingface.co/DFloat11/Qwen-Image-DF11/blob/e78d713015b713a3d5d6965c3399f74fadba0822/README.md
# Example Python code.

from diffusers import DiffusionPipeline, QwenImageTransformer2DModel
import torch
from transformers.modeling_utils import no_init_weights
from dfloat11 import DFloat11Model
import argparse
from typing import Tuple

# --- added imports for looping + file management ---
import os
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Generate images using Qwen-Image model (loops until CTRL+C)')
    parser.add_argument('--cpu_offload', action='store_true', help='Enable CPU offloading')
    parser.add_argument('--cpu_offload_blocks', type=int, default=None, help='Number of transformer blocks to offload to CPU')
    parser.add_argument('--no_pin_memory', action='store_true', help='Disable memory pinning')

    # prompt is now REQUIRED (positional)
    parser.add_argument('prompt', type=str, help='Text prompt for image generation')

    parser.add_argument('--negative_prompt', type=str, default=' ',
                        help='Negative prompt for image generation')
    parser.add_argument('--aspect_ratio', type=str, default='16:9', choices=['1:1', '16:9', '9:16', '4:3', '3:4'],
                        help='Aspect ratio of generated image')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                        help='Number of denoising steps')
    parser.add_argument('--true_cfg_scale', type=float, default=4.0,
                        help='Classifier free guidance scale')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed for generation (each image increments by +1)')

    # --- dedicated MCNL / NSFW LoRA switches (opinionated defaults) ---
    # Download from https://civitai.com/models/1851673/mcnl-multi-concept-nsfw-lora-qwen-image (need to login)
    # And place the LoRA in "models/loras/qwen_MCNL_v1.0.safetensors"
    parser.add_argument('--nsfw', '--mcnl', dest='nsfw', action='store_true',
                        help='Enable MCNL (Multi-Concept NSFW) LoRA for Qwen-Image')
    parser.add_argument('--mcnl-path', type=str,
                        default=str(Path.cwd() / "models" / "loras" / "qwen_MCNL_v1.0.safetensors"),
                        help='Path to MCNL LoRA .safetensors (default: ./models/loras/qwen_MCNL_v1.0.safetensors)')
    parser.add_argument('--mcnl-scale', type=float, default=0.8,
                        help='Strength of MCNL LoRA (default: 0.8)')
    parser.add_argument('--fuse_lora', action='store_true',
                        help='Permanently fuse LoRA after loading (slight speedup; fixed scale during run)')

    return parser.parse_args()


args = parse_args()

model_name = "Qwen/Qwen-Image"

# --- helpers for sequential filenames in ./output ---
OUTPUT_DIR = Path.cwd() / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FNAME_RE = re.compile(r"^(\d{6})\.png$")


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


# --- MCNL/NSFW LoRA helper (keeps structure; called before CPU offload) ---
def _maybe_enable_mcnl(pipe: DiffusionPipeline, lora_path: str, scale: float, fuse: bool) -> Tuple[bool, str]:
    p = Path(lora_path)
    if not p.is_file():
        print(
            "[LoRA] MCNL requested but file not found:\n"
            f"       {p}\n"
            "       Place the LoRA at this path (recommended) or pass --mcnl-path.\n"
            "       (Expected filename from community listing: qwen_MCNL_v1.0.safetensors)"
        )
        return False, ""
    try:
        adapter_name = "mcnl"
        pipe.load_lora_weights(str(p), adapter_name=adapter_name)
        pipe.set_adapters([adapter_name], adapter_weights=[float(scale)])
        if fuse:
            pipe.fuse_lora()
            print(f"[LoRA] Loaded & fused MCNL @ scale={scale}")
        else:
            print(f"[LoRA] Loaded MCNL @ scale={scale}")
        return True, adapter_name
    except Exception as e:
        print(f"[LoRA] Failed to load MCNL from {p}: {e}")
        return False, ""


with no_init_weights():
    transformer = QwenImageTransformer2DModel.from_config(
        QwenImageTransformer2DModel.load_config(
            model_name, subfolder="transformer",
        ),
    ).to(torch.bfloat16)

DFloat11Model.from_pretrained(
    "DFloat11/Qwen-Image-DF11",
    device="cpu",
    cpu_offload=args.cpu_offload,
    cpu_offload_blocks=args.cpu_offload_blocks,
    pin_memory=not args.no_pin_memory,
    bfloat16_model=transformer,
)

pipe = DiffusionPipeline.from_pretrained(
    model_name,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)

# --- Attach MCNL BEFORE enabling CPU offload ---
if args.nsfw:
    _maybe_enable_mcnl(pipe, args.mcnl_path, args.mcnl_scale, args.fuse_lora)

pipe.enable_model_cpu_offload()

# Generate with different aspect ratios
aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
}

width, height = aspect_ratios[args.aspect_ratio]

# --- continuous generation loop, graceful CTRL+C ---
idx = _next_index()
images_made = 0
gen_device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[Run] Output directory: {OUTPUT_DIR}")
print(f"[Run] Starting filename index: {idx:06d}")
print("[Run] Press CTRL+C to stop.")

try:
    while True:
        current_seed = args.seed + images_made
        generator = torch.Generator(device=gen_device).manual_seed(current_seed)

        image = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            width=width,
            height=height,
            num_inference_steps=args.num_inference_steps,
            true_cfg_scale=args.true_cfg_scale,
            generator=generator
        ).images[0]

        fname = f"{idx:06d}.png"
        fpath = OUTPUT_DIR / fname
        image.save(fpath)

        if torch.cuda.is_available():
            max_memory = torch.cuda.max_memory_allocated()
            print(f"[OK] Saved {fname} | seed={current_seed} | Max memory: {max_memory / (1000 ** 3):.2f} GB")
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
    print(f"[Done] Generated {images_made} image(s).")
