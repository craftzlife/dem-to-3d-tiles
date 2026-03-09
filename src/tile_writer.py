"""
Tile writer: saves terrain meshes to OBJ files with metadata.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .mesh_generator import Mesh
from .quadtree import Cell

logger = logging.getLogger(__name__)


def write_obj(mesh: Mesh, path: Path) -> None:
    """Write a mesh to Wavefront OBJ format."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write(f"# Terrain mesh: {path.stem}\n")
        f.write(f"# Vertices: {mesh.vertex_count}, Triangles: {mesh.triangle_count}\n\n")

        # Vertices
        for i in range(mesh.vertex_count):
            v = mesh.vertices[i]
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        f.write("\n")

        # Texture coordinates
        for i in range(mesh.vertex_count):
            uv = mesh.uvs[i]
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

        f.write("\n")

        # Normals
        for i in range(mesh.vertex_count):
            n = mesh.normals[i]
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

        f.write("\n")

        # Faces (1-indexed, with texture coords and normals)
        for i in range(mesh.triangle_count):
            t = mesh.triangles[i]
            i0, i1, i2 = t[0] + 1, t[1] + 1, t[2] + 1
            f.write(f"f {i0}/{i0}/{i0} {i1}/{i1}/{i1} {i2}/{i2}/{i2}\n")


def write_tile(mesh: Mesh, cell: Cell, output_dir: Path) -> Path:
    """Write a mesh tile and return the output file path."""
    rel_path = cell.tile_path + ".obj"
    out_path = output_dir / rel_path
    write_obj(mesh, out_path)
    logger.debug(f"Wrote tile: {rel_path}")
    return out_path


def write_manifest(
    manifest_data: dict,
    output_dir: Path,
) -> None:
    """Write the tile hierarchy manifest as JSON."""
    manifest_path = output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    logger.info(f"Wrote manifest: {manifest_path}")
