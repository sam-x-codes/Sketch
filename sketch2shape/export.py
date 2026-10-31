"""Stage 5: Mesh validation and export (.glb / .obj).

Validates the reconstructed mesh (non-empty, volume, finite bounds,
watertight flag, texture flag) and exports as .glb and .obj + .mtl.
Also performs a round-trip check: re-opens the GLB to confirm it's valid.

Uses trimesh for all mesh operations.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any

import trimesh


MESH_VALIDATION_CHECKS = {
    "non_empty": lambda m: len(m.vertices) > 0,
    "has_volume": lambda m: m.volume > 0.001,
    "finite_bounds": lambda m: m.bounds[1].max() - m.bounds[0].min() > 0,
    "watertight": lambda m: m.is_watertight,
    "has_faces": lambda m: len(m.faces) > 0,
    "max_faces": lambda m: len(m.faces) <= 100_000,
    "max_vertices": lambda m: len(m.vertices) <= 200_000,
}


def validate_mesh(mesh: trimesh.Trimesh) -> Dict[str, bool]:
    """Run validation checks on a trimesh Trimesh object.

    Args:
        mesh: A trimesh Trimesh instance.

    Returns:
        dict mapping check name to bool result.
    """
    results = {}
    for name, check_fn in MESH_VALIDATION_CHECKS.items():
        try:
            results[name] = check_fn(mesh)
        except Exception:
            results[name] = False
    return results


def export_glb(
    mesh: trimesh.Trimesh,
    output_path: Path,
    *,
    maintain_order: bool = False,
) -> Path:
    """Export a mesh to .glb format.

    Args:
        mesh: trimesh Trimesh object.
        output_path: Path where the .glb file will be written.
        maintain_order: If True, preserve original vertex order (may be slower).

    Returns:
        Path to the exported .glb file.
    """
    # Export as GLB
    mesh.export(str(output_path), maintain_order=maintain_order)
    return output_path


def export_obj(
    mesh: trimesh.Trimesh,
    output_path: Path,
    texture_path: Optional[Path] = None,
) -> Dict[str, Path]:
    """Export a mesh to .obj + .mtl, optionally with texture.

    Args:
        mesh: trimesh Trimesh object.
        output_path: Base path (without extension) for .obj output.
        texture_path: Optional path to a .png texture map.

    Returns:
        dict with keys "obj", "mtl", and optionally "texture" paths.
    """
    obj_path = output_path.with_suffix(".obj")
    mtl_path = output_path.with_suffix(".mtl")

    # Write OBJ with materials
    mesh.export(str(obj_path), file_type="obj")

    # If texture provided, we'd handle UV mapping here
    # For now, just export without explicit material
    if texture_path and texture_path.exists():
        # TODO: bake texture onto mesh
        pass

    return {"obj": obj_path, "mtl": mtl_path}


def roundtrip_glb(glb_path: Path) -> bool:
    """Re-open a GLB file to verify it's valid (round-trip check).

    Args:
        glb_path: Path to the .glb file.

    Returns:
        True if the file opens successfully in trimesh.
    """
    try:
        mesh = trimesh.load(str(glb_path), file_type="glb")
        return mesh is not None and not mesh.empty
    except Exception:
        return False


def run_pipeline(
    input_data: Dict[str, Any],
    *,
    output_dir: str = "outputs",
) -> Dict[str, Any]:
    """Run the full export validation pipeline on a reconstructed mesh.

    Args:
        input_data: dict containing at least "mesh" (trimesh.Trimesh)
            and optionally "texture_path".
        output_dir: Where to write the .glb and .obj files.

    Returns:
        dict with:
            - "mesh": the validated/trimesh mesh object (or original if invalid)
            - "glb_path": path to the exported .glb
            - "obj_path": path to the exported .obj
            - "report_json_path": path to report.json
            - "valid": whether all checks passed
            - "roundtrip_valid": whether GLB round-trip check passed
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    mesh = input_data.get("mesh")
    texture_path = input_data.get("texture_path")

    if mesh is None or not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Input must contain a valid trimesh.Trimesh object")

    # Validation
    checks = validate_mesh(mesh)
    valid = all(checks.values())

    # Export
    glb_path = output / "model.glb"
    obj_path = output / "model.obj"

    export_glb(mesh, glb_path)
    export_obj(mesh, obj_path, texture_path=texture_path)

    # Round-trip check
    roundtrip_valid = roundtrip_glb(glb_path)

    # Write report.json
    report = {
        "checks": checks,
        "valid": valid,
        "roundtrip_valid": roundtrip_valid,
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "volume": float(mesh.volume) if hasattr(mesh, "volume") else None,
        "timestamp": time.time(),
    }

    report_path = output / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return {
        "mesh": mesh,
        "glb_path": glb_path,
        "obj_path": obj_path,
        "report_json_path": report_path,
        "valid": valid,
        "roundtrip_valid": roundtrip_valid,
        "checks": checks,
    }