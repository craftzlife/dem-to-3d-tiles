"""
Main processing pipeline.

Orchestrates: DEM indexing -> cell enumeration -> heightmap generation -> tile writing.
Only generates tiles for cells that overlap with available DEM data.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .config import (
    ELEVATION_BASE_SCALE,
    MAX_LOD,
    MESH_RESOLUTION,
    MIN_LOD,
    NUM_FACES,
    SPHERE_RADIUS,
)
from .cube_sphere import face_uv_to_latlng_batch
from .dem_reader import DEMReader
from .heightmap_generator import generate_cell_heightmap
from .quadtree import Cell, iter_cells_at_level
from .tile_writer import write_manifest, write_tile

logger = logging.getLogger(__name__)

# Process-local DEMReader for workers (populated by _init_worker)
_worker_dem_reader: DEMReader | None = None


def _init_worker(input_dir: str) -> None:
    """Pool initializer: create a process-local DEMReader (no shared state)."""
    global _worker_dem_reader
    _worker_dem_reader = DEMReader(input_dir)
    # Note: no atexit needed — ProcessPoolExecutor workers exit via os._exit(0),
    # which bypasses Python cleanup (and stuck GDAL background threads).


def _process_cell(args: tuple[Any, ...]) -> dict | None:
    """
    Module-level, picklable worker function.

    Processes one cell end-to-end: bounds check → heightmap → tile write.
    Returns a manifest entry dict, or None if the cell was skipped.
    """
    cell, output_dir_str, resolution = args
    output_dir = Path(output_dir_str)

    dem_reader = _worker_dem_reader
    assert dem_reader is not None, "Worker DEMReader not initialized"

    # Bounds check (may return 6-tuple for antimeridian-crossing cells)
    bounds = _cell_latlng_bounds(cell)
    if len(bounds) == 6:
        lat_min, lat_max, lon_min1, lon_max1, lon_min2, lon_max2 = bounds
        if not (
            dem_reader.has_data_in_region(lat_min, lat_max, lon_min1, lon_max1)
            or dem_reader.has_data_in_region(lat_min, lat_max, lon_min2, lon_max2)
        ):
            return None
    else:
        lat_min, lat_max, lon_min, lon_max = bounds
        if not dem_reader.has_data_in_region(lat_min, lat_max, lon_min, lon_max):
            return None

    # Generate and write heightmap
    heightmap = generate_cell_heightmap(cell, dem_reader, resolution=resolution)
    tile_path = write_tile(heightmap, cell, output_dir)

    return {
        "face": cell.face,
        "level": cell.level,
        "ix": cell.ix,
        "iy": cell.iy,
        "path": str(tile_path.relative_to(output_dir)),
        "elevation_min": round(heightmap.elevation_min, 2),
        "elevation_max": round(heightmap.elevation_max, 2),
        "elevation_mean": round(heightmap.elevation_mean, 2),
    }


def _cell_latlng_bounds(
    cell: Cell,
) -> tuple[float, float, float, float] | tuple[float, float, float, float, float, float]:
    """
    Get approximate lat/lon bounds for a cell by sampling its corners and edges.

    Uses vectorized face_uv_to_latlng_batch for a single batch call instead of
    9 individual scalar calls.

    Returns (lat_min, lat_max, lon_min, lon_max) normally, or
    (lat_min, lat_max, lon_min1, lon_max1, lon_min2, lon_max2) when the cell
    crosses the antimeridian (split into two longitude ranges).
    """
    u_min, u_max = cell.u_range
    v_min, v_max = cell.v_range
    u_mid = (u_min + u_max) / 2
    v_mid = (v_min + v_max) / 2

    us = np.array([u_min, u_max, u_min, u_max, u_mid, u_mid, u_min, u_max, u_mid])
    vs = np.array([v_min, v_min, v_max, v_max, v_min, v_max, v_mid, v_mid, v_mid])

    lats, lons = face_uv_to_latlng_batch(cell.face, us, vs)

    lat_min, lat_max = float(lats.min()), float(lats.max())

    # Detect antimeridian crossing: if any pair of sampled longitudes differs
    # by more than 180°, the cell straddles the ±180° boundary.
    lon_span = float(lons.max()) - float(lons.min())
    if lon_span > 180.0:
        # Split into two ranges: [min_positive, 180] and [-180, max_negative]
        positive = lons[lons >= 0]
        negative = lons[lons < 0]
        return (
            lat_min, lat_max,
            float(positive.min()), 180.0,
            -180.0, float(negative.max()),
        )

    return lat_min, lat_max, float(lons.min()), float(lons.max())


def run_pipeline(
    input_dir: str | Path,
    output_dir: str | Path,
    min_lod: int = MIN_LOD,
    max_lod: int = MAX_LOD,
    resolution: int = MESH_RESOLUTION,
    sphere_radius: float = SPHERE_RADIUS,
    faces: list[int] | None = None,
    num_workers: int | None = None,
) -> dict:
    """
    Run the full DEM-to-tiles pipeline.

    Args:
        input_dir: Directory containing DEM GeoTIFF files.
        output_dir: Directory for output heightmap tiles.
        min_lod: Minimum LOD level to generate.
        max_lod: Maximum LOD level to generate.
        resolution: Heightmap resolution (pixels per cell edge).
        sphere_radius: Output sphere radius (stored in manifest for Unity).
        faces: List of face indices to process (default: all 6).
        num_workers: Number of parallel worker processes. Defaults to
            min(os.cpu_count(), 8). Use 1 for sequential/debug mode.

    Returns:
        Manifest dictionary with tile hierarchy metadata.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if faces is None:
        faces = list(range(NUM_FACES))

    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 8)

    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"LOD range: {min_lod}-{max_lod}, Resolution: {resolution}")
    logger.info(f"Faces: {faces}")
    logger.info(f"Workers: {num_workers}")

    start_time = time.time()

    # Build full work list, sorted by face+level for spatial locality
    all_cells: list[Cell] = []
    for level in range(min_lod, max_lod + 1):
        for face in faces:
            all_cells.extend(iter_cells_at_level(face, level))
    all_cells.sort(key=lambda c: (c.face, c.level, c.iy, c.ix))

    output_dir_str = str(output_dir)
    work_args = [(cell, output_dir_str, resolution) for cell in all_cells]

    manifest_tiles: list[dict] = []
    total_generated = 0
    total_skipped = 0

    if num_workers == 1:
        # Sequential mode for debugging: use a single in-process DEMReader
        with DEMReader(input_dir) as dem_reader:
            # Temporarily patch the global so _process_cell works correctly
            global _worker_dem_reader
            _worker_dem_reader = dem_reader
            for args in tqdm(work_args, desc="Processing cells"):
                result = _process_cell(args)
                if result is None:
                    total_skipped += 1
                else:
                    manifest_tiles.append(result)
                    total_generated += 1
            _worker_dem_reader = None
    else:
        ctx = multiprocessing.get_context("spawn")
        # chunksize batches work per IPC round-trip; 64 balances latency vs memory.
        # executor.map() is lazy — unlike submit(), it does NOT pre-allocate
        # millions of Future objects, so tqdm starts updating immediately.
        chunksize = max(1, len(work_args) // (num_workers * 16))
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(str(input_dir),),
        ) as executor:
            for result in tqdm(
                executor.map(_process_cell, work_args, chunksize=chunksize),
                total=len(work_args),
                desc="Processing cells",
            ):
                if result is None:
                    total_skipped += 1
                else:
                    manifest_tiles.append(result)
                    total_generated += 1
        # ProcessPoolExecutor.__exit__ calls shutdown(wait=True) which sends a
        # sentinel to each worker causing them to call os._exit(0), cleanly
        # bypassing any stuck GDAL C-level background threads.

    elapsed = time.time() - start_time
    logger.info(
        f"Pipeline complete: {total_generated} tiles generated, "
        f"{total_skipped} skipped, {elapsed:.1f}s elapsed"
    )

    manifest = {
        "version": "2.0",
        "format": "exr",
        "elevation_unit": "meters",
        "sphere_radius": sphere_radius,
        "elevation_base_scale": ELEVATION_BASE_SCALE,
        "resolution": resolution,
        "lod_range": [min_lod, max_lod],
        "tile_count": total_generated,
        "tiles": manifest_tiles,
    }

    write_manifest(manifest, output_dir)
    return manifest
