"""Stage 2: Stable Diffusion 1.5 + ControlNet scribble."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from .config import (
    CONTROLNET_SCRIBBLE_NAME,
    DEFAULT_PROMPT,
    PROMPT_SUFFIX,
    SD_GUIDANCE,
    SD_MODEL_NAME,
    SD_STEPS,
    get_settings,
)

_PIPE = None


def _dtype_device():
    import torch

    if torch.cuda.is_available():
        return torch.float16, "cuda"
    return torch.float32, "cpu"


def _load_pipeline():
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    import torch
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        UniPCMultistepScheduler,
    )

    settings = get_settings()
    dtype, device = _dtype_device()
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_SCRIBBLE_NAME, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SD_MODEL_NAME,
        controlnet=controlnet,
        safety_checker=None,
        requires_safety_checker=False,
        torch_dtype=dtype,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if settings["low_vram"] and device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
        if device == "cuda":
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
    _PIPE = pipe
    return pipe


def unload_generator() -> None:
    """Free SD/ControlNet VRAM before InstantMesh starts."""
    global _PIPE
    import torch

    if _PIPE is None:
        return
    try:
        _PIPE.to("cpu")
    except Exception:
        pass
    del _PIPE
    _PIPE = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def generate_image(
    control_map: np.ndarray,
    prompt: str = "",
    *,
    num_inference_steps: int = SD_STEPS,
    guidance_scale: float = SD_GUIDANCE,
    seed: int = -1,
    control_scale: float = 1.0,
) -> Dict[str, Any]:
    user = (prompt or "").strip() or DEFAULT_PROMPT
    effective = f"{user}{PROMPT_SUFFIX}"
    negative = (
        "multiple objects, cropped, out of frame, blurry, watermark, text, "
        "deformed, cluttered background, low quality"
    )

    control = control_map
    if control.ndim == 2:
        control = np.stack([control] * 3, axis=-1)
    control_pil = Image.fromarray(control.astype(np.uint8)).convert("RGB").resize((512, 512))

    import torch

    pipe = _load_pipeline()
    generator: Optional[torch.Generator] = None
    if seed is not None and int(seed) >= 0:
        gen_device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=gen_device).manual_seed(int(seed))

    result = pipe(
        prompt=effective,
        negative_prompt=negative,
        image=control_pil,
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        controlnet_conditioning_scale=float(control_scale),
        generator=generator,
    )
    image = result.images[0].convert("RGB")
    return {
        "image": np.asarray(image, dtype=np.uint8),
        "prompt": effective,
        "seed": int(seed),
        "control_scale": float(control_scale),
    }
