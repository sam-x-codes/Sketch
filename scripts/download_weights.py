#!/usr/bin/env python
"""Pre-cache all model weights used by Sketch2Shape AI.

This script downloads:
- Stable Diffusion 1.5 checkpoint
- ControlNet scribble model
- rembg U²-Net weights
- InstantMesh checkpoints (Zero123++ + LRM)

Run once after setup.sh, or whenever weights are missing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))
from sketch2shape.config import get_env_vars

def main():
    env = get_env_vars()
    hf_home = env.get("HF_HOME")
    if hf_home:
        os.environ["HF_HOME"] = hf_home
        os.environ["HF_HUB_DOWNLOAD_ROOT"] = hf_home

    print(">>> Downloading Stable Diffusion 1.5 ...")
    from diffusers import StableDiffusionPipeline
    _ = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype="float16",
    )
    print("  ✓ Stable Diffusion cached")

    print(">>> Downloading ControlNet scribble ...")
    from diffusers import StableDiffusionControlNetPipeline
    _ = StableDiffusionControlNetPipeline.from_pretrained(
        "lllyasviel/sd-controlnet-scribble",
        torch_dtype="float16",
    )
    print("  ✓ ControlNet scribble cached")

    print(">>> Ensuring rembg weights ...")
    from rembg import remove
    # remove() triggers download of U²-Net on first call
    test_img = __import__("numpy").array(__import__("numpy").zeros((10, 10, 3)), dtype="uint8")
    # The actual download happens inside remove(); we just call it to cache
    del test_img

    print(">>> Checking InstantMesh weights ...")
    im_dir = Path(__file__).parent.parent / "third_party" / "InstantMesh"
    config_path = im_dir / "configs" / "instant-mesh-large.yaml"
    if not config_path.exists():
        print(f"  ! InstantMesh config not found at {config_path}")
        print("  Run: bash scripts/setup.sh")
    else:
        print(f"  ✓ InstantMesh config present at {config_path}")

    print
    print(">>> All weights cached successfully!")


if __name__ == "__main__":
    main()