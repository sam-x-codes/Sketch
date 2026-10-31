"""Gradio web app for Sketch2Shape AI.

Draw on the canvas or upload a sketch, optionally add a text prompt,
and generate a textured 3D model (.glb + .obj) with an interactive
preview in the browser.
"""

from __future__ import annotations

import os
import uuid
import time
import json
import numpy as np
import gradio as gr

from .pipeline import run_pipeline
from .config import DEFAULT_OUTPUT_DIR


def launch_app():
    """Launch the Gradio web interface."""
    with gr.Blocks(title="Sketch2Shape AI", css=".gradio-container {max-width: 1200px !important}") as app:
        gr.Markdown("# Sketch2Shape AI\nFrom a 2D sketch to a 3D asset")

        with gr.Row():
            with gr.Column(scale=1):
                sketch_input = gr.Sketchpad(
                    label="Draw or erase your sketch",
                    type="numpy",
                    height=512,
                    interactive=True,
                )
                upload_input = gr.Image(
                    label="Or upload a sketch image",
                    sources=["upload"],
                    type="numpy",
                    visible=True,
                )
                prompt_input = gr.Textbox(
                    label="Optional prompt (e.g. 'red leather armchair, matte finish')",
                    placeholder="e.g. red leather armchair, matte finish",
                    value="",
                    scale=4,
                )
                generate_btn = gr.Button("Generate 3D", variant="primary")

            with gr.Column(scale=1):
                status_output = gr.Markdown(label="Status")
                progress = gr.Bar(
                    label="Progress", visible=False, interactive=False
                )
                report_json = gr.Code(label="Report (JSON)", language="json")

        with gr.Row(vISIBLE=False) as model_row:
            model_3d = gr.Model3D(
                label="3D Preview",
                height=500,
                interactive=True,
            )

        with gr.Row():
            with gr.Column():
                glb_output = gr.File(label="Download .glb", visible=False)
                obj_output = gr.File(label="Download .obj", visible=False)

        # Logic
        generate_btn.click(
            fn=_handle_generate,
            inputs=[sketch_input, upload_input, prompt_input],
            outputs=[
                status_output,
                model_3d,
                glb_output,
                obj_output,
                report_json,
            ],
        )

        # When upload changes, hide sketchpad or keep both usable
        upload_input.input(
            fn=lambda: gr.update(visible=False),
            inputs=[],
            outputs=sketch_input,
        )

    return app


def _handle_generate(
    sketch: np.ndarray | None,
    uploaded: np.ndarray | None,
    prompt: str,
) -> tuple:
    """Handle the Generate 3D button click.

    Priority: uploaded image over sketch drawing.
    """
    # Use uploaded image if provided, otherwise sketch drawing
    if uploaded is not None:
        input_image = uploaded
    elif sketch is not None:
        input_image = sketch
    else:
        raise gr.Error("Please draw a sketch or upload an image first.")

    # Run pipeline
    with gr.Spinner("Running pipeline..."):
        output = run_pipeline(
            input_image,
            prompt=prompt or "",
            seed=-1,
            output_dir=str(uuid.uuid4()),
        )

    # Update UI
    status = f"""Pipeline complete ✅
- Preprocess: {output["stages"].get("preprocess", 0):.1f}s
- Generate: {output["stages"].get("generate", 0):.1f}s
- Prepare: {output["stages"].get("prepare", 0):.1f}s
- Reconstruct: {output["stages"].get("reconstruct", 0):.1f}s
- Export: {output["stages"].get("export", 0):.1f}s
- Total: {sum(output["stages"].values()):.1f}s"""
    
    # Load the generated 3D model
    try:
        model_3d = gr.update(
            value=str(output["glb_path"]),
            visible=True,
        )
    except Exception:
        model_3d = gr.update(visible=False)

    return status, model_3d, gr.update(value=str(output["glb_path"]), visible=True), \
           gr.update(value=str(output["obj_path"]), visible=True), \
           gr.update(value=json.dumps(output.get("checks", {}), indent=2), visible=True)


if __name__ == "__main__":
    app = launch_app()
    app.launch(server_name="localhost", server_port=7860)