"""
Terrain heightmap generator.

Generates elevation heightmaps for quadtree cells by sampling DEM data.
Mesh reconstruction is deferred to the Unity client.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MESH_RESOLUTION
from .cube_sphere import face_uv_to_latlng_batch
from .dem_reader import DEMReader
from .quadtree import Cell


@dataclass
class Heightmap:
    """A 2D elevation grid for a single quadtree cell."""

    elevations: np.ndarray  # (resolution, resolution) float32 — meters

    @property
    def resolution(self) -> int:
        return self.elevations.shape[0]

    @property
    def elevation_min(self) -> float:
        return float(np.min(self.elevations))

    @property
    def elevation_max(self) -> float:
        return float(np.max(self.elevations))

    @property
    def elevation_mean(self) -> float:
        return float(np.mean(self.elevations))


def generate_cell_heightmap(
    cell: Cell,
    dem_reader: DEMReader,
    resolution: int = MESH_RESOLUTION,
) -> Heightmap:
    """
    Generate a terrain heightmap for a quadtree cell.

    1. Creates a grid of UV points within the cell bounds.
    2. Converts UV to lat/lon via cube-sphere projection.
    3. Samples DEM elevation at each grid point.

    Args:
        cell: The quadtree cell to generate a heightmap for.
        dem_reader: DEM data source.
        resolution: Number of pixels per cell edge.

    Returns:
        A Heightmap with elevation values in meters.
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
    elevations = dem_reader.get_elevations(lats, lons).astype(np.float32)

    return Heightmap(elevations=elevations)
