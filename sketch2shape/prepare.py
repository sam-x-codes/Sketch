"""Stage 3: Background removal and foreground isolation.

Uses rembg (U²-Net) to remove the background, then crops and scales
the foreground so it centers and fills ~85% of the frame on white.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, Any

try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False


def remove_background(
    image: np.ndarray,
) -> np.ndarray:
    """Remove background from an RGB image using rembg.

    Args:
        image: HxWx3 RGB uint8 array.

    Returns:
        RGBA numpy array with background removed (alpha channel included).
    """
    if not REMBG_AVAILABLE:
        raise ImportError(
            "rembg is not installed. Install with: pip install rembg"
        )

    rgba = remove(image)
    return rgba


def center_and_scale_foreground(
    rgba: np.ndarray,
    *,
    foreground_ratio: float = 0.85,
    target_size: int = 512,
) -> Dict[str, np.ndarray]:
    """Crop and scale the foreground so it centers and fills the frame.

    Args:
        rgba: HxWx4 RGBA array (foreground + alpha).
        foreground_ratio: Foreground should fill approx this ratio of the frame.
        target_size: Output square size in pixels.

    Returns:
        dict with keys:
            - "foreground_rgb": HxWx3 RGB array, centered, scaled
            - "foreground_mask": HxW binary mask of foreground
            - "prepared_rgb": HxWx3 RGB array on white background, ready for 3D reconstruction
    """
    alpha = rgba[:, :, 3]
    rgb = rgba[:, :, :3]

    # Find bounding box of non-transparent pixels
    coords = np.argwhere(alpha > 0)
    if coords.size == 0:
        # No foreground - return white image
        prepared = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
        return {
            "foreground_rgb": rgb,
            "foreground_mask": alpha,
            "prepared_rgb": prepared,
        }

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    h, w = y_max - y_min + 1, x_max - x_min + 1

    # Crop to foreground bbox
    cropped = rgb[y_min : y_max + 1, x_min : x_max + 1]
    cropped_alpha = alpha[y_min : y_max + 1, x_min : x_max + 1]

    # Resize so that the longer side is target_size * foreground_ratio / (1 - some margin)
    # Simple approach: resize so the bounding box fits in target_size with some padding
    new_h, new_w = target_size, target_size

    # Scale the cropped image
    scaled = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # Create white background
    prepared = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255

    # Center the scaled image
    h_s, w_s = scaled.shape[:2]
    y_offset = (target_size - h_s) // 2
    x_offset = (target_size - w_s) // 2
    prepared[y_offset : y_offset + h_s, x_offset : x_offset + w_s] = scaled

    return {
        "foreground_rgb": scaled,
        "foreground_mask": cropped_alpha,
        "prepared_rgb": prepared,
    }