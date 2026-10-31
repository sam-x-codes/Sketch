"""Stage 4: Image-to-3D with texture.

Uses InstantMesh (Zero123++ → LRM) to reconstruct a textured 3D mesh
from a single prepared RGB image.

This stage runs InstantMesh as an isolated subprocess to pin CUDA deps
and free VRAM after each run.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import cv2
from pathlib import Path
from typing import Dict, Any, Optional


def reconstruct_mesh(
    prepared_image: np.ndarray,
    *,
    output_dir: str,
    config_path: Optional[str] = None,
    instantmesh_dir: str = "",
) -> Dict[str, Any]:
    """Run InstantMesh to reconstruct a textured 3D mesh from an image.

    Args:
        prepared_image: HxWx3 RGB uint8 array, centered on white.
        output_dir: Directory where mesh files will be saved.
        config_path: Path to InstantMesh config YAML.
            Defaults to ``instant-mesh-large.yaml``.
        instantmesh_dir: Path to the InstantMesh checkout.
            If empty, uses ``INSTANTMESH_DIR`` env var or
            ``third_party/InstantMesh`` relative to project root.

    Returns:
        dict with keys:
            - "mesh_path": path to the exported .obj (or .glb) file
            - "texture_path": path to the texture map image
            - "report": mesh statistics (vertices, faces, etc.)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    im_dir = Path(instantmesh_dir).resolve() if instantmesh_dir else Path(
        os.getenv("INSTANTMESH_DIR", "third_party/InstantMesh")
    ).resolve()

    if config_path is None:
        config_path = str(im_dir / "configs" / "instant-mesh-large.yaml")

    # Prepare the input image
    with tempfile.NamedTemporaryFile(
        suffix=".png", delete=False, dir=str(output_path)
    ) as tmp:
        tmp_path = tmp.name
        img_bgr = prepared_image[:, :, ::-1]  # RGB -> BGR for OpenCV
        cv2.imwrite(tmp_path, img_bgr)

    # Run InstantMesh CLI
    # The CLI typically accepts: instant-mesh <input_image> <output_dir> [flags]
    cmd = [
        "python", "-m", "instant_mesh.run",
        "--input_image", tmp_path,
        "--output_dir", str(output_path),
        "--config", config_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout
        )
        # Parse output or report
        report = {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired as e:
        report = {
            "stdout": e.stdout if e.stdout else "",
            "stderr": e.stderr if e.stderr else "",
            "timeout": True,
        }

    # Find the generated mesh file
    mesh_files = list(output_path.glob("*.obj")) + list(output_path.glob("*.glb"))
    texture_files = list(output_path.glob("*.png")) + list(output_path.glob("*.jpg"))

    result = {
        "mesh_path": str(mesh_files[0]) if mesh_files else None,
        "texture_path": str(texture_files[0]) if texture_files else None,
        "report": report,
    }

    return result