"""
Chunk-based pipeline orchestrator.

Downloads and processes DEM data in geographic chunks (e.g., 15x15 degrees),
iterating across the globe. Supports progress tracking and resume.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from .s3_downloader import cleanup_chunk, download_chunk
from .pipeline import run_pipeline

logger = logging.getLogger(__name__)


def generate_chunk_bboxes(
    chunk_size: float = 15.0,
    global_bbox: tuple[float, float, float, float] | None = None,
) -> list[tuple[float, float, float, float]]:
    """
    Generate non-overlapping BBOXes covering the specified area.

    Args:
        chunk_size: Degrees per chunk edge.
        global_bbox: (lat_min, lon_min, lat_max, lon_max). Defaults to full Earth.

    Returns:
        List of (lat_min, lon_min, lat_max, lon_max) tuples.
    """
    if global_bbox is None:
        global_bbox = (-90.0, -180.0, 90.0, 180.0)

    lat_min, lon_min, lat_max, lon_max = global_bbox
    bboxes: list[tuple[float, float, float, float]] = []

    lat = lat_min
    while lat < lat_max:
        lat_top = min(lat + chunk_size, lat_max)
        lon = lon_min
        while lon < lon_max:
            lon_right = min(lon + chunk_size, lon_max)
            bboxes.append((lat, lon, lat_top, lon_right))
            lon = lon_right
        lat = lat_top

    return bboxes


def _bbox_key(bbox: tuple[float, float, float, float]) -> str:
    """Convert a BBOX to a string key for progress tracking."""
    return ",".join(str(v) for v in bbox)


def _bbox_folder(bbox: tuple[float, float, float, float]) -> str:
    """Convert a BBOX to a folder name like 'S90_00_W180_00-S75_00_W165_00-DEM'."""
    def _coord(lat: float, lon: float) -> str:
        lat_p = "N" if lat >= 0 else "S"
        lon_p = "E" if lon >= 0 else "W"
        return f"{lat_p}{abs(int(lat)):02d}_00_{lon_p}{abs(int(lon)):03d}_00"
    lat_min, lon_min, lat_max, lon_max = bbox
    return f"{_coord(lat_min, lon_min)}-{_coord(lat_max, lon_max)}-DEM"


def _load_progress(output_dir: Path) -> set[str]:
    """Load set of completed bbox keys from progress.json."""
    progress_path = output_dir / "progress.json"
    if not progress_path.exists():
        return set()
    with open(progress_path) as f:
        data = json.load(f)
    return set(data.get("completed", []))


def _save_progress(output_dir: Path, completed: set[str]) -> None:
    """Save completed bbox keys to progress.json."""
    progress_path = output_dir / "progress.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(progress_path, "w") as f:
        json.dump({"completed": sorted(completed)}, f, indent=2)


def process_chunk(
    bbox: tuple[float, float, float, float],
    download_dir: str | Path,
    output_dir: str | Path,
    min_lod: int = 0,
    max_lod: int = 14,
    resolution: int = 256,
    sphere_radius: float = 100.0,
    faces: list[int] | None = None,
    num_workers: int | None = None,
    cleanup: bool = True,
) -> dict | None:
    """
    Download DEM tiles for a BBOX, process them, and optionally clean up.

    Args:
        bbox: (lat_min, lon_min, lat_max, lon_max) in degrees.
        download_dir: Directory for temporary DEM downloads.
        output_dir: Output directory for tiles.
        min_lod: Minimum LOD level.
        max_lod: Maximum LOD level.
        resolution: Heightmap pixels per cell edge.
        sphere_radius: Output sphere radius.
        faces: Face indices to process.
        num_workers: Number of parallel worker processes.
        cleanup: Remove downloaded .tif files after processing.

    Returns:
        Manifest dict, or None if no tiles were downloaded.
    """
    download_dir = Path(download_dir)
    output_dir = Path(output_dir)

    # Each chunk gets its own subfolder for easy backup/reuse
    chunk_download_dir = download_dir / _bbox_folder(bbox)

    logger.info(f"Processing chunk: bbox={bbox}")

    # Download DEM tiles for this BBOX
    downloaded = download_chunk(bbox, chunk_download_dir)
    if not downloaded:
        logger.info(f"No DEM tiles found for bbox={bbox}, skipping")
        return None

    # Run pipeline with BBOX filtering
    manifest = run_pipeline(
        input_dir=chunk_download_dir,
        output_dir=output_dir,
        min_lod=min_lod,
        max_lod=max_lod,
        resolution=resolution,
        sphere_radius=sphere_radius,
        faces=faces,
        num_workers=num_workers,
        bbox=bbox,
    )

    # Clean up downloaded files
    if cleanup:
        cleanup_chunk(chunk_download_dir)

    return manifest


def process_global(
    download_dir: str | Path,
    output_dir: str | Path,
    chunk_size: float = 15.0,
    global_bbox: tuple[float, float, float, float] | None = None,
    min_lod: int = 0,
    max_lod: int = 14,
    resolution: int = 256,
    sphere_radius: float = 100.0,
    faces: list[int] | None = None,
    num_workers: int | None = None,
    cleanup: bool = True,
    force: bool = False,
) -> None:
    """
    Process the entire globe (or a sub-region) in chunks.

    Downloads, processes, and cleans up one chunk at a time to minimize
    disk usage. Tracks progress so processing can be resumed.

    Args:
        download_dir: Directory for temporary DEM downloads.
        output_dir: Output directory for tiles.
        chunk_size: Degrees per chunk edge.
        global_bbox: Area to process. Defaults to full Earth.
        min_lod: Minimum LOD level.
        max_lod: Maximum LOD level.
        resolution: Heightmap pixels per cell edge.
        sphere_radius: Output sphere radius.
        faces: Face indices to process.
        num_workers: Number of parallel worker processes.
        cleanup: Remove downloaded .tif files after each chunk.
        force: Reprocess already-completed chunks.
    """
    output_dir = Path(output_dir)
    download_dir = Path(download_dir)

    bboxes = generate_chunk_bboxes(chunk_size, global_bbox)
    total = len(bboxes)
    logger.info(f"Global processing: {total} chunks of {chunk_size}x{chunk_size} degrees")

    completed = _load_progress(output_dir)

    for i, bbox in enumerate(bboxes, 1):
        key = _bbox_key(bbox)
        if key in completed and not force:
            logger.info(
                f"Chunk {i}/{total}: bbox={bbox} already completed, skipping"
            )
            continue

        logger.info(
            f"Chunk {i}/{total}: bbox={bbox}"
        )

        process_chunk(
            bbox=bbox,
            download_dir=download_dir,
            output_dir=output_dir,
            min_lod=min_lod,
            max_lod=max_lod,
            resolution=resolution,
            sphere_radius=sphere_radius,
            faces=faces,
            num_workers=num_workers,
            cleanup=cleanup,
        )

        completed.add(key)
        _save_progress(output_dir, completed)

    logger.info(f"Global processing complete: {total} chunks")
