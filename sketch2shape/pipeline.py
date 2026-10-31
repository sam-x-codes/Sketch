"""Stage orchestration, timing, and report.json generation.

Ties together all five pipeline stages, records per-stage durations,
and writes a ``report.json`` at the end of every run.
"""

from __future__ import annotations

import time
import json
import os
from pathlib import Path
from typing import Dict, Any

from .config import STAGES, get_env_vars, DEFAULT_OUTPUT_DIR

from .preprocess import preprocess_sketch
from .generate import generate_image
from .prepare import remove_background, center_and_scale_foreground
from .reconstruct import reconstruct_mesh
from .export import run_pipeline as export_pipeline, validate_mesh


class PipelineTimer:
    """Context manager that records elapsed seconds per stage name."""

    def __init__(self, stage: str):
        self.stage = stage
        self.start: float | None = None
        self.elapsed: float | None = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *_):
        self.elapsed = time.time() - self.start


def run_pipeline(
    sketch_image: np.ndarray,
    *,
    prompt: str = "",
    seed: int = -1,
    output_dir: str = "",
) -> Dict[str, Any]:
    """Run the full Sketch2Shape pipeline and return results + report.

    Args:
        sketch_image: HxWx3 or HxWx4 numpy array (canvas drawing or uploaded image).
        prompt: Optional text prompt appended with default suffix.
        seed: Random seed for diffusion (-1 for random).
        output_dir: Output directory for this run. If empty, uses S2S_OUTPUT_DIR
            env var or ``outputs/``.

    Returns:
        dict with:
            - "stages": dict mapping stage name to elapsed seconds
            - "report_json_path": path to the written report.json
            - "output_dir": final output directory
            - all intermediate artifacts (generated image, prepared image, etc.)
    """
    if not output_dir:
        env = get_env_vars()
        output_dir = env.get("S2S_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    timer: Dict[str, float] = {}
    result: Dict[str, Any] = {}

    # ---- Stage 1: Preprocess ----
    with PipelineTimer("preprocess") as timer_pre:
        preprocessed = preprocess_sketch(sketch_image)
    timer["preprocess"] = timer_pre.elapsed
    result["sketch"] = preprocessed["sketch"]
    result["control_map"] = preprocessed["control_map"]

    # ---- Stage 2: Generate ----
    with PipelineTimer("generate") as timer_gen:
        generated = generate_image(
            preprocessed["control_map"],
            prompt=prompt,
            seed=seed,
        )
    timer["generate"] = timer_gen.elapsed
    result["generated_image"] = generated["image"]
    result["generated_prompt"] = generated["prompt"]

    # ---- Stage 3: Prepare (background removal) ----
    with PipelineTimer("prepare") as timer_prep:
        rgba = remove_background(generated["image"])
        prepared = center_and_scale_foreground(rgba)
    timer["prepare"] = timer_prep.elapsed
    result["prepared_image"] = prepared["prepared_rgb"]
    result["foreground_mask"] = prepared["foreground_mask"]

    # ---- Stage 4: Reconstruct ----
    with PipelineTimer("reconstruct") as timer_recon:
        recon = reconstruct_mesh(
            prepared["prepared_rgb"],
            output_dir=str(out / "reconstruct"),
        )
    timer["reconstruct"] = timer_recon.elapsed
    result["reconstruct_report"] = recon.get("report", {})

    if recon.get("mesh_path"):
        result["mesh_path"] = recon["mesh_path"]

    # ---- Stage 5: Export ----
    with PipelineTimer("export") as timer_exp:
        mesh_data = {
            "mesh": trimesh.load(result.get("mesh_path"), file_type="obj")
            if result.get("mesh_path")
            else None,
            "texture_path": recon.get("texture_path"),
        }
        export_result = export_pipeline(mesh_data, output_dir=str(out))
    timer["export"] = timer_exp.elapsed

    result["glb_path"] = export_result["glb_path"]
    result["obj_path"] = export_result["obj_path"]
    result["report_json_path"] = export_result["report_json_path"]
    result["valid"] = export_result["valid"]
    result["roundtrip_valid"] = export_result["roundtrip_valid"]
    result["checks"] = export_result["checks"]

    # Store timings in result
    result["stages"] = timer

    # Write consolidated report.json
    final_report = {
        "stages": timer,
        "prompt": prompt or "none",
        "seed": seed,
        "valid": result["valid"],
        "roundtrip_valid": result["roundtrip_valid"],
        "checks": result["checks"],
        "timings_seconds": timer,
    }

    report_path = out / "report.json"
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2)

    result["report_json_path"] = report_path

    return result