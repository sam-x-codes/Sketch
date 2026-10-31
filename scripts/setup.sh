#!/usr/bin/env bash
set -euo pipefail

# Sketch2Shape AI — one-shot environment + InstantMesh + weights setup
# TechZephyr 4.0 · IIT Bhubaneswar · 24-Hour Live Hackathon

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=== Sketch2Shape AI Setup ==="
echo

# 1) Create & activate venv
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

# 2) Install PyTorch (cpu-compatible version)
echo ">>> Installing PyTorch..."
pip install --upgrade pip
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0

# 3) Install xformers (cpu-compatible)
echo ">>> Installing xformers..."
pip install xformers==0.0.22.post7

# 4) Clone InstantMesh if not present
if [ ! -d "third_party/InstantMesh" ]; then
  echo ">>> Cloning InstantMesh..."
  git clone https://github.com/TencentARC/InstantMesh.git third_party/InstantMesh
fi
cd third_party/InstantMesh
git checkout main 2>/dev/null || true
pip install -r requirements.txt
cd "$PROJECT_ROOT"

# 5) Install remaining Python deps
echo ">>> Installing Python dependencies..."
pip install -r requirements.txt

# 6) Install controlnet-annotator (scribble utility; control map generated in preprocess.py)
echo ">>> Installing controlnet-annotator (may fail on CPU/Python 3.15; scribble map generated in preprocess.py)..."
pip install controlnet-annotator[scribble] 2>/dev/null || echo "Note: controlnet-annotator skipped - scribble map generated in preprocess.py"

# 7) Download model weights
echo ">>> Downloading model weights..."
python3 scripts/download_weights.py

echo
echo "=== Setup complete ==="
echo "Activate venv: source .venv/bin/activate"
echo "Launch app:   python3 app.py"