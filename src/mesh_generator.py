"""
Terrain mesh generator.

Generates triangle meshes for quadtree cells by sampling DEM elevation data
and projecting vertices onto the cube-sphere surface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ELEVATION_BASE_SCALE, ELEVATION_SCALE, MESH_RESOLUTION, SPHERE_RADIUS
from .cube_sphere import face_uv_to_latlng_batch, face_uv_to_xyz_batch
from .dem_reader import DEMReader
from .quadtree import Cell


@dataclass
class Mesh:
    """A triangle mesh with vertices, normals, and triangle indices."""

    vertices: np.ndarray  # (N, 3) float64
    normals: np.ndarray  # (N, 3) float64
    triangles: np.ndarray  # (M, 3) int32 — vertex indices
    uvs: np.ndarray  # (N, 2) float32 — texture coordinates

    @property
    def vertex_count(self) -> int:
        return self.vertices.shape[0]

    @property
    def triangle_count(self) -> int:
        return self.triangles.shape[0]


def generate_cell_mesh(
    cell: Cell,
    dem_reader: DEMReader,
    resolution: int = MESH_RESOLUTION,
    sphere_radius: float = SPHERE_RADIUS,
    elevation_scale: float = ELEVATION_SCALE,
) -> Mesh:
    """
    Generate a terrain mesh for a quadtree cell.

    1. Creates a grid of UV points within the cell bounds.
    2. Converts UV to lat/lon and samples DEM elevation.
    3. Converts UV + elevation to 3D vertices on the cube-sphere.
    4. Builds triangle indices for the grid.

    Args:
        cell: The quadtree cell to generate a mesh for.
        dem_reader: DEM data source.
        resolution: Number of vertices per cell edge.
        sphere_radius: Radius of the sphere in output units.
        elevation_scale: Elevation exaggeration factor.

    Returns:
        A Mesh object with vertices, normals, triangles, and UVs.
    """
    u_min, u_max = cell.u_range
    v_min, v_max = cell.v_range

    # Create UV grid within cell bounds
    u_vals = np.linspace(u_min, u_max, resolution)
    v_vals = np.linspace(v_min, v_max, resolution)
    u_grid, v_grid = np.meshgrid(u_vals, v_vals)  # (res, res) each

    # Convert UV to lat/lon for DEM sampling
    lats, lons = face_uv_to_latlng_batch(cell.face, u_grid, v_grid)

    # Sample elevation from DEM
    elevations = dem_reader.get_elevations(lats, lons)

    # Convert UV to unit sphere XYZ (direction vectors)
    dir_x, dir_y, dir_z = face_uv_to_xyz_batch(cell.face, u_grid, v_grid)

    # Apply elevation: radius = sphere_radius + elevation * scale
    radii = sphere_radius + elevations * ELEVATION_BASE_SCALE * elevation_scale * sphere_radius

    # Compute 3D vertex positions
    vx = dir_x * radii
    vy = dir_y * radii
    vz = dir_z * radii

    # Flatten to (N, 3) vertex array
    vertices = np.stack([vx.ravel(), vy.ravel(), vz.ravel()], axis=1)

    # Normals: outward-pointing (same as direction for a sphere)
    normals = np.stack([dir_x.ravel(), dir_y.ravel(), dir_z.ravel()], axis=1)

    # Texture coordinates: local UV within cell mapped to [0, 1]
    local_u = np.linspace(0, 1, resolution)
    local_v = np.linspace(0, 1, resolution)
    lu, lv = np.meshgrid(local_u, local_v)
    uvs = np.stack([lu.ravel(), lv.ravel()], axis=1).astype(np.float32)

    # Build triangle indices for the grid
    triangles = _build_grid_triangles(resolution, resolution)

    return Mesh(
        vertices=vertices,
        normals=normals,
        triangles=triangles,
        uvs=uvs,
    )


def _build_grid_triangles(rows: int, cols: int) -> np.ndarray:
    """
    Build triangle indices for a rows×cols vertex grid.

    Each quad (i,j)-(i+1,j)-(i,j+1)-(i+1,j+1) produces 2 triangles.
    Returns (M, 3) int32 array of vertex indices.
    """
    tris = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            # Vertex indices for the quad corners
            tl = r * cols + c
            tr = r * cols + (c + 1)
            bl = (r + 1) * cols + c
            br = (r + 1) * cols + (c + 1)
            # Two triangles per quad
            tris.append([tl, bl, tr])
            tris.append([tr, bl, br])
    return np.array(tris, dtype=np.int32)
