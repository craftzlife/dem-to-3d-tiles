"""
Main processing pipeline.

Orchestrates: DEM indexing → cell enumeration → mesh generation → tile writing.
Only generates tiles for cells that overlap with available DEM data.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from tqdm import tqdm

from .config import (
    ELEVATION_SCALE,
    MAX_LOD,
    MESH_RESOLUTION,
    MIN_LOD,
    NUM_FACES,
    SPHERE_RADIUS,
)
from .cube_sphere import face_uv_to_latlng
from .dem_reader import DEMReader
from .mesh_generator import generate_cell_mesh
from .quadtree import Cell, iter_cells_at_level
from .tile_writer import write_manifest, write_tile

logger = logging.getLogger(__name__)


def _cell_latlng_bounds(cell: Cell) -> tuple[float, float, float, float]:
    """
    Get approximate lat/lon bounds for a cell by sampling its corners and edges.

    Returns (lat_min, lat_max, lon_min, lon_max).
    """
    u_min, u_max = cell.u_range
    v_min, v_max = cell.v_range

    # Sample corners and edge midpoints for better bounds estimation
    sample_points = [
        (u_min, v_min),
        (u_max, v_min),
        (u_min, v_max),
        (u_max, v_max),
        ((u_min + u_max) / 2, v_min),
        ((u_min + u_max) / 2, v_max),
        (u_min, (v_min + v_max) / 2),
        (u_max, (v_min + v_max) / 2),
        ((u_min + u_max) / 2, (v_min + v_max) / 2),
    ]

    lats = []
    lons = []
    for u, v in sample_points:
        lat, lon = face_uv_to_latlng(cell.face, u, v)
        lats.append(lat)
        lons.append(lon)

    return min(lats), max(lats), min(lons), max(lons)


def run_pipeline(
    input_dir: str | Path,
    output_dir: str | Path,
    min_lod: int = MIN_LOD,
    max_lod: int = MAX_LOD,
    resolution: int = MESH_RESOLUTION,
    sphere_radius: float = SPHERE_RADIUS,
    elevation_scale: float = ELEVATION_SCALE,
    faces: list[int] | None = None,
) -> dict:
    """
    Run the full DEM-to-tiles pipeline.

    Args:
        input_dir: Directory containing DEM GeoTIFF files.
        output_dir: Directory for output mesh tiles.
        min_lod: Minimum LOD level to generate.
        max_lod: Maximum LOD level to generate.
        resolution: Mesh resolution (vertices per cell edge).
        sphere_radius: Output sphere radius.
        elevation_scale: Elevation exaggeration factor.
        faces: List of face indices to process (default: all 6).

    Returns:
        Manifest dictionary with tile hierarchy metadata.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if faces is None:
        faces = list(range(NUM_FACES))

    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"LOD range: {min_lod}-{max_lod}, Resolution: {resolution}")
    logger.info(f"Faces: {faces}")

    start_time = time.time()

    with DEMReader(input_dir) as dem_reader:
        dem_bounds = dem_reader.bounds
        if dem_bounds is None:
            logger.error("No DEM data found!")
            return {"tiles": [], "error": "No DEM data found"}

        logger.info(
            f"DEM bounds: lat [{dem_bounds[0]:.2f}, {dem_bounds[1]:.2f}], "
            f"lon [{dem_bounds[2]:.2f}, {dem_bounds[3]:.2f}]"
        )

        manifest_tiles = []
        total_generated = 0
        total_skipped = 0

        for level in range(min_lod, max_lod + 1):
            level_generated = 0
            cells = []
            for face in faces:
                cells.extend(iter_cells_at_level(face, level))

            desc = f"LOD {level} ({len(cells)} cells)"
            for cell in tqdm(cells, desc=desc, leave=True):
                # Check if cell overlaps with DEM data
                lat_min, lat_max, lon_min, lon_max = _cell_latlng_bounds(cell)
                if not dem_reader.has_data_in_region(
                    lat_min, lat_max, lon_min, lon_max
                ):
                    total_skipped += 1
                    continue

                # Generate and write mesh
                mesh = generate_cell_mesh(
                    cell,
                    dem_reader,
                    resolution=resolution,
                    sphere_radius=sphere_radius,
                    elevation_scale=elevation_scale,
                )
                tile_path = write_tile(mesh, cell, output_dir)

                manifest_tiles.append(
                    {
                        "face": cell.face,
                        "level": cell.level,
                        "ix": cell.ix,
                        "iy": cell.iy,
                        "path": str(tile_path.relative_to(output_dir)),
                        "vertices": mesh.vertex_count,
                        "triangles": mesh.triangle_count,
                    }
                )
                level_generated += 1
                total_generated += 1

            logger.info(f"LOD {level}: generated {level_generated} tiles")

    elapsed = time.time() - start_time
    logger.info(
        f"Pipeline complete: {total_generated} tiles generated, "
        f"{total_skipped} skipped, {elapsed:.1f}s elapsed"
    )

    manifest = {
        "version": "1.0",
        "sphere_radius": sphere_radius,
        "elevation_scale": elevation_scale,
        "resolution": resolution,
        "lod_range": [min_lod, max_lod],
        "tile_count": total_generated,
        "tiles": manifest_tiles,
    }

    write_manifest(manifest, output_dir)
    return manifest
