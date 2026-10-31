#!/usr/bin/env bash
# Sketch2Shape AI — environment, InstantMesh (pinned), and weight download.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

INSTANTMESH_COMMIT="8079d9136d738e2c016f4ba4a6bc916d52147761"

pick_python() {
  for candidate in python3.11 python3.10 python3 \
      /opt/homebrew/opt/python@3.11/bin/python3.11 \
      /opt/homebrew/bin/python3.11; do
    if command -v "$candidate" >/dev/null 2>&1; then
      ver="$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
      major="${ver%%.*}"
      minor="${ver#*.}"
      if [[ "$major" -gt 3 ]] || { [[ "$major" -eq 3 ]] && [[ "$minor" -ge 10 ]]; }; then
        echo "$candidate"
        return 0
      fi
    fi
  done
  echo "Need Python 3.10 or 3.11. On macOS: brew install python@3.11" >&2
  exit 1
}

PYTHON_BIN="$(pick_python)"
echo "Using $($PYTHON_BIN --version)"

HAS_CUDA=0
if command -v nvidia-smi >/dev/null 2>&1; then
  HAS_CUDA=1
fi

create_venv() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
    "$PYTHON_BIN" -m venv "$dir"
  fi
}

install_torch() {
  local pip="$1"
  "$pip" install --upgrade pip wheel setuptools
  if [[ "$HAS_CUDA" -eq 1 ]]; then
    echo ">>> Installing PyTorch 2.1.0 + CUDA 12.1"
    "$pip" install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
    "$pip" install xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu121 || \
      echo "xformers skipped (optional)"
  else
    echo ">>> No NVIDIA GPU detected — installing CPU/MPS PyTorch (InstantMesh will not run here)"
    "$pip" install torch torchvision torchaudio
  fi
}

echo "=== Sketch2Shape AI setup ==="

create_venv ".venv"
install_torch ".venv/bin/pip"
.venv/bin/pip install -r "$PROJECT_ROOT/requirements.txt"

create_venv ".venv-instantmesh"
install_torch ".venv-instantmesh/bin/pip"

mkdir -p third_party
if [[ ! -d third_party/InstantMesh/.git ]]; then
  echo ">>> Cloning InstantMesh"
  git clone https://github.com/TencentARC/InstantMesh.git third_party/InstantMesh
fi
git -C third_party/InstantMesh fetch --all --tags
git -C third_party/InstantMesh checkout "$INSTANTMESH_COMMIT"

echo ">>> Installing InstantMesh Python deps into .venv-instantmesh"
REQ_TMP="$(mktemp)"
# nvdiffrast and bitsandbytes need CUDA; skip them on CPU/macOS.
grep -v -E 'nvdiffrast|bitsandbytes' third_party/InstantMesh/requirements.txt > "$REQ_TMP" || true
# InstantMesh pins an old Gradio; we only need its inference stack in this venv.
.venv-instantmesh/bin/pip install -r "$REQ_TMP" || true
rm -f "$REQ_TMP"

if [[ "$HAS_CUDA" -eq 1 ]]; then
  echo ">>> Installing nvdiffrast (CUDA)"
  .venv-instantmesh/bin/pip install "git+https://github.com/NVlabs/nvdiffrast.git" || \
    echo "nvdiffrast failed — install CUDA toolkit + build-essential and retry"
else
  echo ">>> Skipping nvdiffrast (no CUDA). Stage 4 will fail on this machine."
fi

export INSTANTMESH_PYTHON="$PROJECT_ROOT/.venv-instantmesh/bin/python"
export INSTANTMESH_DIR="$PROJECT_ROOT/third_party/InstantMesh"

echo ">>> Pre-caching Hugging Face weights (long download, ~15 GB)"
.venv/bin/python scripts/download_weights.py || echo "Weight download incomplete — re-run: .venv/bin/python scripts/download_weights.py"

echo
echo "=== Setup finished ==="
echo "Activate:        source .venv/bin/activate"
echo "Web app:         python app.py"
echo "CLI:             python run_pipeline.py --input examples/inputs/chair.png --prompt 'wooden chair'"
if [[ "$HAS_CUDA" -eq 0 ]]; then
  echo
  echo "WARNING: This machine has no NVIDIA GPU. Preprocess/UI can run, but"
  echo "SD+ControlNet is slow on CPU and InstantMesh requires CUDA 12.1."
  echo "Copy this repo to a Linux box with a 12–24 GB NVIDIA GPU and re-run setup.sh"
fi
