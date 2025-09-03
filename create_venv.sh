#!/usr/bin/env bash
# Robust venv setup for Qwen-Image on RTX 5090 (Ubuntu 24.04.x)
# - Creates ./.venv
# - Installs PyTorch nightly cu130 + pinned dependencies
# - Verifies CUDA + Diffusers imports

set -Eeuo pipefail

# ---------------- helpers ----------------
err() {
  local exit_code=$?
  echo ""
  echo "============================================================"
  echo "❌ Setup failed"
  echo "Command: ${BASH_COMMAND}"
  echo "Line   : ${BASH_LINENO[0]}"
  echo "Exit   : ${exit_code}"
  echo "Tip    : Scroll up for the full traceback. See common fixes:"
  echo "         • Ensure 'python3-venv' is installed (Ubuntu: sudo apt install -y python3-venv)"
  echo "         • Internet access to PyTorch nightly index"
  echo "         • Sufficient disk space and permissions"
  echo "============================================================"
  exit "${exit_code}"
}
trap err ERR

log() { echo -e "➡️  $*"; }

# ---------------- config ----------------
VENV_DIR=".venv"
PYTHON_BIN="python3"
TORCH_INDEX_URL="https://download.pytorch.org/whl/nightly/cu130"

# ---------------- checks ----------------
log "Checking Python..."
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python3 not found on PATH."
  echo "Install Python 3.10+ and re-run."
  exit 1
fi

log "Checking venv module..."
if ! "${PYTHON_BIN}" -m venv --help >/dev/null 2>&1; then
  echo "'python3-venv' module missing."
  echo "Install it (Ubuntu): sudo apt update && sudo apt install -y python3-venv"
  exit 1
fi

# ---------------- venv ----------------
if [[ -d "${VENV_DIR}" ]]; then
  log "Existing venv found at ${VENV_DIR} (reusing)."
else
  log "Creating virtual environment at ${VENV_DIR}..."
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
log "Using Python: $(python -V)"
log "Upgrading build tooling..."
python -m pip install --upgrade pip setuptools wheel

# ---------------- installs ----------------
log "Installing PyTorch (nightly, CUDA 13.0 - cu130) for RTX 5090..."
pip install --pre torch torchvision --index-url "${TORCH_INDEX_URL}"

log "Installing pinned Hugging Face + Diffusers stack..."
pip install \
  "transformers==4.56.0" \
  "accelerate==1.10.1" \
  "safetensors==0.6.2" \
  "huggingface-hub==0.34.4" \
  "pillow==11.3.0" \
  "cupy-cuda13x==13.6.0"

pip install -U dfloat11[cuda13]

# Diffusers latest from GitHub
pip install "git+https://github.com/huggingface/diffusers"

# ---------------- validation ----------------
log "Validating PyTorch CUDA availability..."
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in PyTorch. Check driver/CUDA/Torch wheel.")
print(f"✔ CUDA device: {torch.cuda.get_device_name(0)}")
print(f"✔ Torch: {torch.__version__} | CUDA toolkit: {torch.version.cuda}")
print(f"✔ Supported archs: {torch.cuda.get_arch_list()}")
torch.randn(1, device="cuda").sin_()
PY

log "Validating Diffusers import..."
python - <<'PY'
from diffusers import DiffusionPipeline
print("✔ Diffusers import OK:", DiffusionPipeline)
PY

# ---------------- summary ----------------
echo ""
echo "============================================================"
echo "✅ Environment ready!"
echo "Location : ${VENV_DIR}"
echo "Activate : source ${VENV_DIR}/bin/activate"
echo "Test gen : python - <<'PY'\n"
echo "from diffusers import DiffusionPipeline; import torch"
echo 'pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=torch.bfloat16).to("cuda")'
echo 'img = pipe(prompt="A retro-futuristic city street at dusk", negative_prompt=" ", num_inference_steps=50, true_cfg_scale=4.0).images[0]'
echo 'img.save("qwen_image.png"); print("Saved qwen_image.png")'
echo "PY"
echo "============================================================"
