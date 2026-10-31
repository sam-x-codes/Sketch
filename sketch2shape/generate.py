"""Stage 2: Sketch-guided image generation.

Uses Stable Diffusion 1.5 + ControlNet scribble to generate an image
conditioned on the sketch's contours, with an optional text prompt.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Dict, Any

from .config import (
    SD_MODEL_NAME,
    CONTROLNET_SKRIBBLE_NAME,
    PROMPT_SUFFIX,
    STAGES,
    get_env_vars,
)


def _get_pipeline():
    """Load SD + ControlNet pipeline on demand."""
    from diffusers import StableDiffusionControlNetPipeline

    env = get_env_vars()
    low_vram = env.get("S2S_LOW_VRAM", "0") == "1"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        SD_MODEL_NAME,
        controlnet=CONTROLNET_SKRIBBLE_NAME,
        safety_checker=None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    if low_vram:
        pipeline.enable_xformers_memory_efficient_attention()
        pipeline.to("cpu")
        pipeline.enable_sequential_cpu_offload()
    else:
        pipeline.to(device)

    return pipeline


def generate_image(
    sketch_control_map: np.ndarray,
    prompt: str = "",
    *,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    seed: int = -1,
) -> Dict[str, Any]:
    """Generate a sketch-guided image using SD + ControlNet.

    Args:
        sketch_control_map: HxW binary control map (white strokes on black).
        prompt: Optional text prompt. If empty, uses default suffix.
        num_inference_steps: Diffusion steps.
        guidance_scale: Classifier-free guidance scale.
        seed: Random seed (-1 for random).

    Returns:
        dict with keys:
            - "image": generated RGB numpy array (H, W, 3)
            - "control_map": the input control map
            - "prompt": the effective prompt used
    """
    if not prompt:
        prompt = PROMPT_SUFFIX.lstrip(", ")

    effective_prompt = f"masterpiece, best quality, {prompt}{PROMPT_SUFFIX}"

    pipeline = _get_pipeline()

    generator = torch.Generator(device=pipeline.device)
    if seed >= 0:
        generator.manual_seed(seed)

    # ControlNet expects HWC image with 3 channels
    control_hwc = np.stack([sketch_control_map.astype(np.uint8)] * 3, axis=-1)

    with torch.no_grad():
        images = pipeline(
            prompt=effective_prompt,
            image=control_hwc,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            controlnet_conditioning_scale=1.0,
        ).images

    img_array = np.asarray(images[0])
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]

    return {
        "image": img_array,
        "control_map": sketch_control_map,
        "prompt": effective_prompt,
    }