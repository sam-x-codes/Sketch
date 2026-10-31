#!/usr/bin/env python3
"""Pre-cache SD 1.5, ControlNet scribble, rembg, and InstantMesh checkpoints."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sketch2shape.config import (  # noqa: E402
    CONTROLNET_SCRIBBLE_NAME,
    SD_MODEL_NAME,
    get_settings,
)


def main() -> int:
    settings = get_settings()
    if settings["hf_home"]:
        os.environ["HF_HOME"] = settings["hf_home"]

    import torch
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
    from huggingface_hub import hf_hub_download

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(">>> Stable Diffusion 1.5 + ControlNet scribble")
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_SCRIBBLE_NAME, torch_dtype=dtype)
    _ = StableDiffusionControlNetPipeline.from_pretrained(
        SD_MODEL_NAME,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=dtype,
    )
    print("    cached")

    print(">>> rembg U2-Net")
    try:
        from rembg import new_session

        new_session("u2net")
        print("    cached")
    except Exception as exc:
        print(f"    rembg skipped: {exc}")

    print(">>> InstantMesh checkpoints (TencentARC/InstantMesh)")
    ckpt_dir = Path(settings["instantmesh_dir"]) / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "diffusion_pytorch_model.bin",
        "instant_mesh_large.ckpt",
        "instant_mesh_base.ckpt",
    ):
        try:
            path = hf_hub_download(
                repo_id="TencentARC/InstantMesh",
                filename=filename,
                repo_type="model",
                local_dir=str(ckpt_dir),
            )
            print(f"    {filename} -> {path}")
        except Exception as exc:
            print(f"    {filename} skipped: {exc}")

    print(">>> Zero123++ v1.2")
    try:
        from diffusers import DiffusionPipeline

        DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.2",
            custom_pipeline="zero123plus",
            torch_dtype=dtype,
        )
        print("    cached")
    except Exception as exc:
        print(f"    zero123++ skipped (ok if InstantMesh venv will pull it): {exc}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
