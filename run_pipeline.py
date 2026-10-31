"""Command-line entry point for Sketch2Shape AI.

Usage:

    python run_pipeline.py --input examples/inputs/chair.png \
        --prompt "wooden chair, oak finish" --seed 42

Results are saved to ``outputs/<timestamp>/``.
"""

from __future__ import annotations

import argparse
import uuid
import time
import json
import sys
from pathlib import Path

import numpy as np

from .pipeline import run_pipeline as pipeline_runner
from .config import DEFAULT_OUTPUT_DIR


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Sketch2Shape AI: from 2D sketch to 3D model"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input sketch image (PNG/JPG/WEBP)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Optional text prompt (appended with default suffix)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed for generation (-1 for random)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output directory (default: S2S_OUTPUT_DIR env or outputs/)",
    )

    args = parser.parse_args(argv)

    # Load input image
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    from PIL import Image
    img = np.array(Image.open(input_path).convert("RGBA"))
    # Ensure HxWx4 or HxWx3
    if img.shape[2] == 4:
        # Alpha-flatten on white
        background = np.ones_like(img, dtype=np.uint8) * 255
        background[:, :, :3] = img[:, :, :3] * (img[:, :, 3:] / 255.0) + 255 * (1.0 - img[:, :, 3:] / 255.0)
        img = background[:, :, :3]

    # Run pipeline
    output_dir = args.output or ""
    output = pipeline_runner(
        img,
        prompt=args.prompt,
        seed=args.seed,
        output_dir=output_dir or None,
    )

    # Print summary
    print(f"\nSketch2Shape Pipeline Complete")
    print(f"Output directory: {output['output_dir']}")
    print(f"Report: {output['report_json_path']}")
    print(f"GLB: {output['glb_path']}")
    print(f"OBJ: {output['obj_path']}")
    print(f"Valid: {output['valid']}")
    print(f"Round-trip valid: {output['roundtrip_valid']}")

    stages = output.get("stages", {})
    total = sum(stages.values())
    print("\nStage timings (seconds):")
    for stage, t in stages.items():
        print(f"  {stage}: {t:.1f}s")
    print(f"  Total: {total:.1f}s")


if __name__ == "__main__":
    main()