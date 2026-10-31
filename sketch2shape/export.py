"""Stage 5: validate reconstructed mesh and export .glb + .obj."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import trimesh

PathLike = Union[str, Path]


class MeshExportError(ValueError):
    """Mesh failed mandatory geometry checks."""


def load_mesh(path: PathLike) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path), process=False, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise MeshExportError(f"No triangle mesh inside {path}")
        loaded = trimesh.util.concatenate(geoms)
    if not isinstance(loaded, trimesh.Trimesh):
        raise MeshExportError(f"Could not load a triangle mesh from {path}")
    return loaded


def mesh_stats(mesh: trimesh.Trimesh) -> Dict[str, Any]:
    extents = mesh.extents
    min_e = float(extents.min()) if extents is not None else 0.0
    max_e = float(extents.max()) if extents is not None else 0.0
    flatness = min_e / (max_e + 1e-12)
    volume = None
    try:
        if mesh.is_watertight:
            volume = float(mesh.volume)
    except Exception:
        volume = None
    has_texture = bool(
        getattr(getattr(mesh, "visual", None), "uv", None) is not None
        or getattr(getattr(mesh, "visual", None), "material", None) is not None
    )
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "is_watertight": bool(mesh.is_watertight),
        "volume": volume,
        "bounds_min": mesh.bounds[0].tolist() if mesh.bounds is not None else None,
        "bounds_max": mesh.bounds[1].tolist() if mesh.bounds is not None else None,
        "extents": extents.tolist() if extents is not None else None,
        "flatness_ratio": float(flatness),
        "has_texture": has_texture,
        "finite_vertices": bool(mesh.vertices.size and np_all_finite(mesh.vertices)),
    }


def np_all_finite(values) -> bool:
    import numpy as np

    return bool(np.isfinite(values).all())


def validate_mesh(mesh: trimesh.Trimesh) -> Dict[str, Any]:
    stats = mesh_stats(mesh)
    checks = {
        "non_empty": stats["vertices"] > 0 and stats["faces"] > 0,
        "finite_bounds": stats["finite_vertices"] and stats["extents"] is not None,
        "has_volume_or_depth": stats["flatness_ratio"] >= 0.04,
        "not_a_flat_plane": stats["flatness_ratio"] >= 0.04,
    }
    if stats["volume"] is not None:
        checks["positive_volume"] = stats["volume"] > 1e-8
    return {"checks": checks, "stats": stats, "valid": all(checks.values())}


def roundtrip_glb(glb_path: PathLike) -> bool:
    try:
        mesh = load_mesh(glb_path)
        return len(mesh.vertices) > 0 and len(mesh.faces) > 0
    except Exception:
        return False


def export_assets(
    mesh_or_path: Union[trimesh.Trimesh, PathLike],
    output_dir: PathLike,
    *,
    texture_path: Optional[PathLike] = None,
) -> Dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mesh = mesh_or_path if isinstance(mesh_or_path, trimesh.Trimesh) else load_mesh(mesh_or_path)
    report = validate_mesh(mesh)
    if not report["valid"]:
        raise MeshExportError(
            "Mesh rejected as flat or empty. Re-run with another seed; "
            f"checks={report['checks']}"
        )

    glb_path = out / "model.glb"
    obj_path = out / "model.obj"
    mesh.export(str(glb_path))
    mesh.export(str(obj_path))
    if not roundtrip_glb(glb_path):
        raise MeshExportError(f"Exported GLB could not be re-opened: {glb_path}")

    payload = {
        "glb_path": str(glb_path),
        "obj_path": str(obj_path),
        "texture_path": str(texture_path) if texture_path else None,
        "roundtrip_valid": True,
        **report,
    }
    return payload


def main(argv: Optional[list[str]] = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("Usage: python -m sketch2shape.export path/to/model.glb", file=sys.stderr)
        return 2
    path = Path(args[0])
    mesh = load_mesh(path)
    report = validate_mesh(mesh)
    report["roundtrip_valid"] = path.suffix.lower() in {".glb", ".gltf"} and roundtrip_glb(path)
    print(json.dumps(report, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
