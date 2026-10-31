"""Gradio UI: draw or upload a sketch, preview and download a 3D asset."""

from __future__ import annotations

import json
from typing import Any, Optional

import gradio as gr

from sketch2shape.pipeline import run_pipeline
from sketch2shape.preprocess import SketchError, to_rgb_uint8


def _has_pixels(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return any(value.get(k) is not None for k in ("composite", "image", "value", "background"))
    return True


def generate_3d(
    sketch: Any,
    uploaded: Any,
    prompt: str,
    seed: float,
    control_scale: float,
    progress=gr.Progress(track_tqdm=False),
):
    try:
        if _has_pixels(uploaded):
            image = to_rgb_uint8(uploaded)
        elif _has_pixels(sketch):
            image = to_rgb_uint8(sketch)
        else:
            raise SketchError("Draw a sketch or upload an image first.")
        seed_i = int(seed)
        status_messages = []

        def on_progress(message: str) -> None:
            status_messages.append(message)
            try:
                progress(len(status_messages) / 5.0, desc=message)
            except Exception:
                pass

        result = run_pipeline(
            image,
            prompt=prompt or "",
            seed=seed_i,
            control_scale=float(control_scale),
            progress=on_progress,
        )
        stages = result["stages"]
        total = sum(stages.values())
        hw = result.get("hardware") or {}
        status = (
            "Pipeline complete.\n\n"
            f"- Preprocess: {stages.get('preprocess', 0):.1f}s\n"
            f"- SD + ControlNet: {stages.get('generate', 0):.1f}s\n"
            f"- Background removal: {stages.get('prepare', 0):.1f}s\n"
            f"- InstantMesh: {stages.get('reconstruct', 0):.1f}s\n"
            f"- Validate + export: {stages.get('export', 0):.1f}s\n"
            f"- **Total: {total:.1f}s**\n\n"
            f"GPU: {hw.get('gpu', 'unknown')} · VRAM: {hw.get('vram_gb', 'n/a')} GB\n"
            f"Output: `{result['output_dir']}`"
        )
        report_text = json.dumps(result.get("report", {}), indent=2)
        return (
            status,
            result["sketch_path"],
            result["control_map_path"],
            result["generated_path"],
            result["prepared_path"],
            result["glb_path"],
            result["glb_path"],
            result["obj_path"],
            report_text,
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Sketch2Shape AI", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
# Sketch2Shape AI
Draw a rough 2D sketch **or** upload an image, optionally add a prompt, and generate a textured 3D model (`.glb` + `.obj`).
Upload takes priority over the canvas. A prompt is never required.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                sketch = gr.Sketchpad(
                    label="Draw (brush / eraser / clear)",
                    type="numpy",
                    height=480,
                )
                uploaded = gr.Image(
                    label="Or upload a sketch (PNG / JPG / WEBP)",
                    sources=["upload"],
                    type="numpy",
                    height=240,
                )
                prompt = gr.Textbox(
                    label="Optional prompt",
                    placeholder="red leather armchair, matte finish",
                )
                with gr.Row():
                    seed = gr.Number(label="Seed (-1 = random)", value=42, precision=0)
                    control_scale = gr.Slider(
                        label="Sketch adherence",
                        minimum=0.4,
                        maximum=1.5,
                        value=1.0,
                        step=0.05,
                    )
                generate_btn = gr.Button("Generate 3D", variant="primary")
            with gr.Column(scale=1):
                status = gr.Markdown("Ready.")
                model = gr.Model3D(label="Interactive 3D preview", height=420)
                with gr.Row():
                    glb = gr.File(label="Download .glb")
                    obj = gr.File(label="Download .obj")
        with gr.Row():
            sketch_clean = gr.Image(label="01 sketch clean")
            control_map = gr.Image(label="02 control map")
            generated = gr.Image(label="03 generated")
            prepared = gr.Image(label="04 prepared")
        report = gr.Code(label="report.json", language="json")
        generate_btn.click(
            fn=generate_3d,
            inputs=[sketch, uploaded, prompt, seed, control_scale],
            outputs=[
                status,
                sketch_clean,
                control_map,
                generated,
                prepared,
                model,
                glb,
                obj,
                report,
            ],
        )
        gr.Markdown(
            "TechZephyr 4.0 · local models only after the one-time weight download. "
            "Needs an NVIDIA GPU for InstantMesh."
        )
    return demo


if __name__ == "__main__":
    build_app().launch(server_name="0.0.0.0", server_port=7860)
