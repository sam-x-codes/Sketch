"""Stage 4: InstantMesh in an isolated subprocess (pinned CUDA env)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from .config import get_settings


class ReconstructError(RuntimeError):
    """InstantMesh failed or produced no mesh."""


def reconstruct_mesh(
    prepared_image: np.ndarray,
    *,
    output_dir: str | Path,
    seed: int = 42,
    config_path: Optional[str] = None,
    timeout_s: int = 1200,
) -> Dict[str, Any]:
    settings = get_settings()
    im_dir = Path(settings["instantmesh_dir"]).resolve()
    if not (im_dir / "run.py").is_file():
        raise ReconstructError(
            f"InstantMesh not found at {im_dir}. Run: bash scripts/setup.sh"
        )

    python = Path(settings["instantmesh_python"])
    if not python.is_file():
        python = Path(sys.executable)

    rel_config = config_path or settings["instantmesh_config"]
    cfg = Path(rel_config)
    if not cfg.is_absolute():
        cfg = im_dir / rel_config
    if not cfg.is_file():
        raise ReconstructError(f"InstantMesh config missing: {cfg}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    input_path = out / "instantmesh_input.png"
    bgr = prepared_image[:, :, ::-1] if prepared_image.ndim == 3 else prepared_image
    cv2.imwrite(str(input_path), bgr)

    im_out = out / "instantmesh"
    im_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(python),
        "run.py",
        str(cfg),
        str(input_path.resolve()),
        "--output_path",
        str(im_out.resolve()),
        "--export_texmap",
        "--seed",
        str(int(seed) if seed is not None and int(seed) >= 0 else 42),
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    print("Running InstantMesh:", " ".join(cmd), flush=True)
    try:
        result = subprocess.run(
            cmd,
            cwd=str(im_dir),
            env=env,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise ReconstructError(
            f"InstantMesh timed out after {timeout_s}s. Try instant-mesh-base.yaml or S2S_LOW_VRAM=1."
        ) from exc

    if result.returncode != 0:
        raise ReconstructError(
            f"InstantMesh exited with code {result.returncode}. "
            "Need an NVIDIA GPU with CUDA 12.1. See README troubleshooting."
        )

    meshes = sorted((im_out).rglob("*.obj"))
    meshes = [p for p in meshes if "meshes" in p.parts] or meshes
    if not meshes:
        raise ReconstructError(f"InstantMesh produced no .obj under {im_out}")

    mesh_path = meshes[0]
    texture = mesh_path.with_suffix(".png")
    if not texture.is_file():
        pngs = list(mesh_path.parent.glob("*.png"))
        texture = pngs[0] if pngs else None

    return {
        "mesh_path": str(mesh_path),
        "texture_path": str(texture) if texture else None,
        "instantmesh_dir": str(im_out),
        "command": cmd,
    }
