"""Configuration for Sketch2Shape AI pipelines."""

from __future__ import annotations

import os
from typing import Dict, List

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
DEFAULT_EXAMPLES_INPUTS = os.path.join(PROJECT_ROOT, "examples", "inputs")
DEFAULT_EXAMPLES_OUTPUTS = os.path.join(PROJECT_ROOT, "examples", "outputs")

# Model paths
INSTANTMESH_DIR_DEFAULT = os.path.join(PROJECT_ROOT, "third_party", "InstantMesh")
INSTANTMESH_CONFIG_DEFAULT = os.path.join(INSTANTMESH_DIR_DEFAULT, "configs", "instant-mesh-large.yaml")

# SD / ControlNet
SD_MODEL_NAME = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET_SKRIBBLE_NAME = "lllyasviel/sd-controlnet-scribble"

# Prompt template
PROMPT_SUFFIX = ", single object, white background, studio render"

# Image processing
INPUT_TARGET_SIZE = 512
FOREGROUND_RATIO = 0.85

# Pipeline stages
STAGES = [
    "preprocess",
    "generate",
    "prepare",
    "reconstruct",
    "export",
]


def get_env_vars() -> Dict[str, str]:
    """Return optional environment variable overrides."""
    return {
        "INSTANTMESH_DIR": os.getenv("INSTANTMESH_DIR", INSTANTMESH_DIR_DEFAULT),
        "INSTANTMESH_CONFIG": os.getenv(
            "INSTANTMESH_CONFIG", INSTANTMESH_CONFIG_DEFAULT
        ),
        "S2S_LOW_VRAM": os.getenv("S2S_LOW_VRAM", "0"),
        "S2S_OUTPUT_DIR": os.getenv("S2S_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
        "HF_HOME": os.getenv("HF_HOME"),
    }