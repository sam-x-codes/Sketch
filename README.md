# Sketch2Shape AI: Running on Kaggle (Complete Setup & Reproducibility Guide)

This guide explains, step by step, how to set up and run Sketch2Shape AI on a free **Kaggle GPU notebook**,
without owning an NVIDIA GPU. It documents the exact versions that work, the reason for every pin, the measured
runtime, and a troubleshooting table covering every error encountered while building this setup.

Everything here was tested on Kaggle with a **Tesla T4 (14.56 GB VRAM)**, Python 3.11, PyTorch 2.1.0 + CUDA 12.1.

---

## 1. Why Kaggle, and what to expect

InstantMesh (the image-to-3D stage) needs an NVIDIA GPU, CUDA and a compiled `nvdiffrast` extension. A Windows
laptop without a GPU cannot run it, so we use Kaggle's free GPU.

| Item | Value |
|---|---|
| Platform | Kaggle Notebook (interactive session) |
| Accelerator | GPU T4 x2 (the pipeline uses one GPU) or GPU P100 |
| OS / Python | Ubuntu (Kaggle image), Python 3.11 installed via `uv` |
| First-time setup | about 45 to 60 minutes (mostly downloads of ~15 GB of weights and two PyTorch installs) |
| One generation | about 4 minutes on a T4 (first run, see Section 8) |
| Free GPU quota | about 30 hours per week, counted only while a GPU session is running |

**Important:** a Kaggle session is temporary. When it ends or is reset, everything in `/tmp` is deleted
(environments, weights, outputs). Always download results before stopping the session (Section 9).

---

## 2. Before you start

1. A Kaggle account with **phone verification** (required to enable Internet and GPU).
2. Create a new notebook and open its settings sidebar:
   - **Accelerator:** GPU T4 x2 (or P100)
   - **Internet:** On
3. Disk layout. `/kaggle/working` is limited to about 20 GB, which is **not enough** for two virtual
   environments plus ~15 GB of model weights. The root filesystem (`/tmp`) has far more space, so **everything is
   built in `/tmp/Sketch`**.

---

## 3. Cell-by-cell setup

Run each block below as its own notebook cell, in order.

> **Rule for every `%%bash` cell:** `%%bash` must be on the very first line of the cell, with nothing above it
> (not even a blank line or a comment). Otherwise Kaggle runs the cell as Python and fails with a `SyntaxError`.

### Step 1: Clone the project into `/tmp`

```bash
%%bash
cd /tmp
git clone https://github.com/rajivdey2/YOUR_REPO_NAME.git Sketch
ls Sketch
```

`ls` should show `app.py`, `scripts`, `requirements.txt`, `sketch2shape`, and so on.
If the repository is private, upload the project as a Kaggle Dataset and copy it instead:
`cp -r /kaggle/input/YOUR_DATASET_NAME /tmp/Sketch`.

### Step 2: Install Python 3.11 and run the main setup (about 25 to 40 minutes)

Kaggle ships a newer Python than the one this project supports (PyTorch 2.1.0 has no wheels for Python 3.12 or
3.13). We install Python 3.11 with `uv` and put it first on `PATH` so `setup.sh` picks it up for both virtual
environments.

```bash
%%bash
cd /tmp/Sketch
pip install -q uv
uv python install 3.11
PY=$(uv python find 3.11)
export PATH="$(dirname "$PY"):$PATH"
rm -rf .venv .venv-instantmesh
bash scripts/setup.sh 2>&1 | tee setup.log
```

This creates two environments (`.venv` for the app, `.venv-instantmesh` for InstantMesh), installs PyTorch
2.1.0 (CUDA 12.1), clones InstantMesh at the pinned commit `8079d9136d738e2c016f4ba4a6bc916d52147761`, and
downloads the model weights.

Expected messages at the end of this step (all harmless):
- `nvdiffrast failed ...`: it is rebuilt properly in Step 5.
- `zero123++ skipped ...`: InstantMesh downloads Zero123++ itself on the first run.

### Step 3: Install the CUDA 12.1 toolkit

Kaggle provides CUDA 13 (`nvcc 13.0`), but our PyTorch is built for CUDA 12.1. PyTorch refuses to compile
extensions when the major versions differ, so the matching toolkit must be installed alongside.

```bash
%%bash
if [ -x /usr/local/cuda-12.1/bin/nvcc ]; then echo "already installed"; else
cd /tmp
wget -q https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
sh cuda_12.1.1_530.30.02_linux.run --silent --toolkit --toolkitpath=/usr/local/cuda-12.1 --no-drm --no-man-page --override
rm -f cuda_12.1.1_530.30.02_linux.run
fi
/usr/local/cuda-12.1/bin/nvcc --version | tail -2
```

The last line must show `release 12.1`.

### Step 4: Apply the pinned package versions

`setup.sh` installs the newest allowed versions of several packages, and these newer versions are incompatible
with PyTorch 2.1.0 and the Gradio app. This step pins the versions that are known to work together.

```bash
%%bash
cd /tmp/Sketch

# Main environment (.venv): app, Stable Diffusion, ControlNet
.venv/bin/pip uninstall -y -q opencv-python opencv-python-headless
.venv/bin/pip install -q "numpy<2" "opencv-python-headless==4.10.0.84"
.venv/bin/pip install -q --no-deps "pydantic==2.10.6" "pydantic-core==2.27.2" "starlette==0.37.2" "fastapi==0.110.3"

# InstantMesh environment (.venv-instantmesh)
.venv-instantmesh/bin/pip install -q "setuptools==69.5.1" wheel ninja "numpy<2" "onnxruntime==1.18.1" "huggingface_hub==0.23.5"
```

`pip` may print "dependency resolver" warnings about `tokenizers` or `opencv`. These are metadata
warnings only. The imports are verified in Step 7.

### Step 5: Build nvdiffrast

```bash
%%bash
cd /tmp/Sketch
apt-get install -y -qq gcc-12 g++-12 > /dev/null
export CC=gcc-12 CXX=g++-12 CUDAHOSTCXX=g++-12
export CUDA_HOME=/usr/local/cuda-12.1 PATH=/usr/local/cuda-12.1/bin:$PATH
export TORCH_CUDA_ARCH_LIST="6.0;7.5" MAX_JOBS=2
rm -rf third_party/nvdiffrast
git clone -q https://github.com/NVlabs/nvdiffrast.git third_party/nvdiffrast
.venv-instantmesh/bin/pip install --no-build-isolation third_party/nvdiffrast > build.log 2>&1
echo "nvdiffrast exit: $?"
tail -3 build.log
```

It must print `nvdiffrast exit: 0`. The build takes a few minutes.

- `--no-build-isolation` lets the build see the installed PyTorch.
- `gcc-12` is used because CUDA 12.1 rejects newer default compilers.
- `TORCH_CUDA_ARCH_LIST="6.0;7.5"` covers the P100 (6.0) and T4 (7.5).

### Step 6: Configuration changes

```bash
%%bash
cd /tmp/Sketch
sed -i -E 's/grid_res: [0-9]+/grid_res: 96/; s/texture_resolution: [0-9]+/texture_resolution: 2048/' third_party/InstantMesh/configs/instant-mesh-large.yaml
sed -i 's/launch(server_name="0.0.0.0", server_port=7860)/launch(server_name="0.0.0.0", server_port=7860, share=True)/' app.py
.venv/bin/python scripts/make_example_sketches.py
grep -n -E "grid_res|texture_resolution" third_party/InstantMesh/configs/instant-mesh-large.yaml
```

| Change | Reason |
|---|---|
| `grid_res: 96` | The default (128) ran out of memory during mesh extraction on a 15 GB GPU. 96 fits and gives a smoother mesh than 64. |
| `texture_resolution: 2048` | Higher-resolution baked texture (default is lower). |
| `share=True` in `app.py` | Kaggle has no public port, so Gradio creates a temporary public `gradio.live` link. |

### Step 7: Verify the installation

```bash
%%bash
cd /tmp/Sketch
.venv/bin/python -c "import cv2, numpy, torch, pydantic, gradio; print('main env OK', cv2.__version__, numpy.__version__, torch.cuda.is_available(), pydantic.VERSION)"
.venv-instantmesh/bin/python -c "import rembg, accelerate, diffusers, nvdiffrast.torch; print('instantmesh env OK')"
ls third_party/InstantMesh/ckpts
```

Expected: `main env OK 4.10.0 1.26.4 True 2.10.6`, then `instantmesh env OK`, and the checkpoint
folder listing `diffusion_pytorch_model.bin`, `instant_mesh_base.ckpt` and `instant_mesh_large.ckpt`.

### Step 8: Launch the web app

This is a plain **Python** cell (no `%%bash`). It sets the environment, starts the app in the background, and
prints the public link.

```python
import os, subprocess, time
W = "/tmp/Sketch"
os.environ.update({
  "MPLBACKEND": "Agg",
  "CUDA_HOME": "/usr/local/cuda-12.1",
  "PATH": "/usr/local/cuda-12.1/bin:" + os.environ["PATH"],
  "TORCH_CUDA_ARCH_LIST": "6.0;7.5",
  "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
  "INSTANTMESH_PYTHON": f"{W}/.venv-instantmesh/bin/python",
  "INSTANTMESH_DIR": f"{W}/third_party/InstantMesh",
})
subprocess.run('pkill -f "[a]pp.py"', shell=True); time.sleep(3)
subprocess.Popen(f"cd {W} && .venv/bin/python app.py > app.log 2>&1", shell=True)
time.sleep(40)
print(subprocess.run("grep gradio.live /tmp/Sketch/app.log", shell=True, capture_output=True, text=True).stdout
      or "No link yet. Wait 30 s, then run: !tail -5 /tmp/Sketch/app.log")
```

Open the printed `https://....gradio.live` link. Draw on the canvas or upload a sketch, optionally enter a
prompt, and click **Generate 3D**. The link works only while the session is running and expires after 72 hours.

Environment variables used by the app:

| Variable | Purpose |
|---|---|
| `MPLBACKEND=Agg` | Notebooks set an inline Matplotlib backend that the app's environment does not know; `Agg` avoids the crash. |
| `CUDA_HOME`, `PATH` | Point to the CUDA 12.1 toolkit so nvdiffrast can compile at runtime. |
| `TORCH_CUDA_ARCH_LIST` | GPU architectures to compile for (P100 and T4). |
| `PYTORCH_CUDA_ALLOC_CONF` | Reduces memory fragmentation and out-of-memory errors. |
| `INSTANTMESH_PYTHON`, `INSTANTMESH_DIR` | Tell the app where the InstantMesh interpreter and checkout live. |

### Using the web app and what it runs

`app.py` is the Gradio front end. When you click **Generate 3D**, it calls `run_pipeline` in
`sketch2shape/pipeline.py`, which runs the five stages below. The terminal log (`/tmp/Sketch/app.log`) prints
`1/5` to `5/5` as each stage starts.

| In the app | What happens |
|---|---|
| Draw on the canvas (brush, eraser, clear) **or** upload an image | Uploaded images take priority over the canvas |
| Optional prompt, for example `wooden lamp, brass base` | Never required; a neutral default is used if empty |
| Sketch adherence slider | Controls how strongly ControlNet follows your lines (`control_scale`) |
| Seed | Same seed and inputs give a repeatable result |
| **Generate 3D** | Runs stages 1/5 to 5/5 (about 4 minutes on a T4) |
| Result panel | Shows the intermediate images, an interactive 3D viewer, and download buttons for `.glb` and `.obj` |

| Stage | Module | What it does |
|---|---|---|
| 1/5 Preprocess | `sketch2shape/preprocess.py` | Cleans the sketch and builds the scribble control map |
| 2/5 Stable Diffusion + ControlNet | `sketch2shape/generate.py` | Generates a product-style image guided by the sketch and prompt |
| 3/5 Background removal | `sketch2shape/prepare.py` | Isolates and centers the object (rembg) |
| 4/5 InstantMesh reconstruction | `sketch2shape/reconstruct.py` | Generates multi-view images and reconstructs the textured mesh |
| 5/5 Validate and export | `sketch2shape/export.py` | Checks the mesh, writes `.glb` and `.obj`, and writes `report.json` |

If the app shows an error after you click Generate, the stage that failed is the last `N/5` line in
`/tmp/Sketch/app.log` (`!tail -40 /tmp/Sketch/app.log`).

### Step 8b (alternative): Run from the command line

```bash
%%bash
cd /tmp/Sketch
export MPLBACKEND=Agg CUDA_HOME=/usr/local/cuda-12.1 PATH=/usr/local/cuda-12.1/bin:$PATH TORCH_CUDA_ARCH_LIST="6.0;7.5"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export INSTANTMESH_PYTHON=/tmp/Sketch/.venv-instantmesh/bin/python INSTANTMESH_DIR=/tmp/Sketch/third_party/InstantMesh
.venv/bin/python run_pipeline.py --input examples/inputs/chair.png --prompt "wooden chair" --seed 42
```

---

## 9. Saving results and stopping the session

Do this **before** stopping, because `/tmp` is erased.

```bash
%%bash
cd /tmp/Sketch
pkill -f "[a]pp.py" || true
zip -rq /kaggle/working/results.zip outputs
ls -lh /kaggle/working/results.zip
```

Download `results.zip` from the notebook's **Output** tab (right sidebar). Then **stop the session**
(Session options, then Stop Session). Closing the browser tab does **not** stop it and the GPU quota keeps counting.
You can confirm nothing is running under **Active Events** in your Kaggle profile menu.

Each run folder in `outputs/<timestamp>-<id>/` contains:

| File | Content |
|---|---|
| `01_sketch_clean.png` | Preprocessed sketch |
| `02_control_map.png` | ControlNet scribble map |
| `03_generated.png` | Sketch-guided generated image |
| `04_prepared.png`, `04_prepared_rgba.png` | Background-removed image sent to the 3D stage |
| `instantmesh/.../images/instantmesh_input.png` | Multi-view image produced by Zero123++ |
| `model.glb` | Final textured 3D model |
| `model.obj`, `material.mtl`, `material_0.png` | Same model as OBJ with texture |
| `report.json` | Prompt, seed, hardware, per-stage timings, mesh statistics, validation checks |

---

## 10. Measured results (Tesla T4)

Taken from `report.json` of a real run (prompt: "lamp", seed 42):

| Item | Value |
|---|---|
| GPU / VRAM | Tesla T4, 14.56 GB |
| PyTorch | 2.1.0+cu121 |
| Preprocess | 0.04 s |
| Stable Diffusion + ControlNet (25 steps) | 25.5 s |
| Background removal | 2.6 s |
| InstantMesh reconstruction | 200.6 s |
| Validate + export | 2.7 s |
| **Total** | **231.5 s** |
| Mesh | 22,491 vertices, 35,420 faces, textured (2048 x 2048) |
| Validation | non-empty, finite bounds, real depth, not a flat plane, GLB re-opens (round-trip valid) |

Notes on these numbers:
- This was a **first run in a fresh session**. The reconstruction time includes loading the Zero123++ and
  InstantMesh models and one-time downloads of Zero123++ and DINO weights. Later runs in the same session skip the
  downloads and are faster.
- InstantMesh runs as a separate process, so its models are reloaded for each generation.
- Warm-run total (second generation in the same session): **_fill in from a second `report.json`_**.

---

## 11. Why each version is pinned

| Package | Pinned version | Reason |
|---|---|---|
| Python | 3.11 (via `uv`) | PyTorch 2.1.0 has no wheels for 3.12/3.13, and `venv` fails on Kaggle's default Python. |
| PyTorch | 2.1.0 + cu121 | Version required by the pinned InstantMesh commit. |
| CUDA toolkit | 12.1 | Must match the PyTorch build to compile nvdiffrast. |
| gcc / g++ | 12 | CUDA 12.1 rejects newer host compilers. |
| numpy | < 2 | PyTorch 2.1.0 is built against NumPy 1.x. |
| opencv-python-headless | 4.10.0.84 | OpenCV 5.x requires NumPy 2. |
| pydantic / pydantic-core | 2.10.6 / 2.27.2 | Newer pydantic breaks Gradio's API-info generation (`TypeError: argument of type 'bool' is not iterable`). |
| starlette / fastapi | 0.37.2 / 0.110.3 | Newer releases are incompatible with the Gradio version used. |
| setuptools | 69.5.1 | Newer setuptools breaks PyTorch 2.1.0's extension builder. |
| onnxruntime | 1.18.1 | Required by `rembg`; not installed by InstantMesh's requirements. |
| huggingface_hub | 0.23.5 | `accelerate` needs a newer hub than the one pulled in with `diffusers==0.20.2`. |

---

## 12. Troubleshooting (errors we actually hit)

| Symptom | Cause | Fix |
|---|---|---|
| `SyntaxError` on `pip install ...` in a cell | `%%bash` was not the first line | Put `%%bash` alone on line 1 |
| `ensurepip ... returned non-zero exit status 1` | Default Python too new for `venv` | Use Python 3.11 via `uv` (Step 2) |
| No matching distribution for `torch==2.1.0` | Python 3.12 or 3.13 | Use Python 3.11 (Step 2) |
| `cd: /kaggle/working/Sketch: No such file or directory` | A command still uses the old path | Use `/tmp/Sketch` in every cell |
| Disk nearly full (`/kaggle/working` 20 GB) | Environments and weights are too large | Build in `/tmp` (Section 2) |
| nvdiffrast: "Cannot compile nvdiffrast CUDA extension ... --no-build-isolation" | Build cannot see PyTorch, or setuptools too new | `setuptools==69.5.1` and `--no-build-isolation` (Steps 4, 5) |
| nvdiffrast build fails with no output | CUDA 13 `nvcc` vs CUDA 12.1 PyTorch, or incompatible gcc | Install CUDA 12.1 toolkit and use gcc-12 (Steps 3, 5) |
| `Failed to initialize NumPy` or `opencv requires numpy>=2` | NumPy 2 / OpenCV 5 installed | `numpy<2` and `opencv-python-headless==4.10.0.84` (Step 4) |
| `No module named 'onnxruntime'` (InstantMesh stage) | Missing dependency of `rembg` | `onnxruntime==1.18.1` in `.venv-instantmesh` (Step 4) |
| `cannot import name 'split_torch_state_dict_into_shards'` | `huggingface_hub` too old for `accelerate` | `huggingface_hub==0.23.5` (Step 4) |
| `'module://matplotlib_inline.backend_inline' is not a valid value for backend` | Notebook sets an inline Matplotlib backend | `MPLBACKEND=Agg` (Step 8) |
| `CUDA out of memory. Tried to allocate 15.00 GiB` at mesh extraction | `grid_res: 128` too large for 15 GB | `grid_res: 96` (Step 6); use 80 if it still fails |
| Gradio page crashes: `TypeError: argument of type 'bool' is not iterable` | Newer pydantic incompatible with Gradio | `pydantic==2.10.6`, `pydantic-core==2.27.2` (Step 4) |
| Gradio page crashes in `TemplateResponse` | Newer starlette/fastapi | `starlette==0.37.2`, `fastapi==0.110.3` (Step 4) |
| `OSError: Cannot find empty port in range: 7860-7860` | An old copy of the app is still running | `pkill -f "[a]pp.py"` then relaunch (Step 8) |
| `pip` runs for 10+ minutes when changing a version | Dependency resolver backtracking | Use `pip install --no-deps` with exact versions |
| "Pipelines loaded with dtype=torch.float16 cannot run with cpu" | Library warning only; the GPU is used (SD ran 25 steps in about 5 s) | Ignore |
| "pip's dependency resolver ... tokenizers requires huggingface_hub<0.18" | Metadata warning; imports verified in Step 7 | Ignore |
| `/tmp/Sketch/app.py: No such file` after reopening the notebook | The session was reset and `/tmp` was erased | Repeat Steps 1 to 6 |

---

## 13. Limitations

- Requires an NVIDIA GPU (CUDA 12.1). Kaggle T4/P100 works; CPU-only machines and Apple Silicon cannot run the
  3D reconstruction stage.
- Mesh resolution is capped at `grid_res: 96` on a 15 GB GPU. Higher values ran out of memory on the T4.
- A session reset requires repeating the setup (about 45 to 60 minutes of GPU time). To avoid this, the
  environments and weights can be saved as a Kaggle Dataset.
- Very thin (1 px) line drawings can lose strokes during preprocessing; bold, dark strokes on a light background
  give the best sketch adherence.
- The exported mesh is not watertight (marching-cubes output) and is not intended for 3D printing or animation.
- Hidden sides of the object are inferred from a single view, so the back is plausible rather than exact.
- The free Kaggle quota is about 30 GPU hours per week; sessions end when idle or after the maximum session length.

---

## 14. Reproducibility checklist (Kaggle)

- [ ] Notebook accelerator set to GPU, Internet On
- [ ] Project cloned into `/tmp/Sketch`
- [ ] Steps 1 to 7 completed; Step 7 prints both "OK" lines and `nvdiffrast exit: 0` was seen in Step 5
- [ ] App launched (Step 8) and the `gradio.live` link opens
- [ ] One generation completed; `model.glb` opens (for example in https://gltf-viewer.donmccurdy.com or Blender)
- [ ] `results.zip` downloaded and the session stopped
