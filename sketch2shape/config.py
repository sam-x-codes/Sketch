"""Paths, model ids, and environment-overridable settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_EXAMPLES_INPUTS = PROJECT_ROOT / "examples" / "inputs"
DEFAULT_EXAMPLES_OUTPUTS = PROJECT_ROOT / "examples" / "outputs"

INSTANTMESH_DIR_DEFAULT = PROJECT_ROOT / "third_party" / "InstantMesh"
INSTANTMESH_COMMIT = "8079d9136d738e2c016f4ba4a6bc916d52147761"
INSTANTMESH_CONFIG_RELATIVE = "configs/instant-mesh-large.yaml"

SD_MODEL_NAME = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET_SCRIBBLE_NAME = "lllyasviel/sd-controlnet-scribble"

DEFAULT_PROMPT = "a single everyday object, studio product photo"
PROMPT_SUFFIX = ", single object, white background, studio render, centered, photorealistic"

INPUT_TARGET_SIZE = 512
FOREGROUND_RATIO = 0.85
SD_STEPS = 25
SD_GUIDANCE = 7.5

STAGES = ("preprocess", "generate", "prepare", "reconstruct", "export")


def env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def get_settings() -> Dict[str, Any]:
    im_dir = Path(os.getenv("INSTANTMESH_DIR", str(INSTANTMESH_DIR_DEFAULT))).expanduser()
    config = os.getenv("INSTANTMESH_CONFIG", INSTANTMESH_CONFIG_RELATIVE)
    python = os.getenv("INSTANTMESH_PYTHON", str(PROJECT_ROOT / ".venv-instantmesh" / "bin" / "python"))
    output = Path(os.getenv("S2S_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))).expanduser()
    hf_home: Optional[str] = os.getenv("HF_HOME")
    return {
        "instantmesh_dir": im_dir,
        "instantmesh_config": config,
        "instantmesh_python": Path(python),
        "low_vram": env_flag("S2S_LOW_VRAM"),
        "output_dir": output,
        "hf_home": hf_home,
    }
