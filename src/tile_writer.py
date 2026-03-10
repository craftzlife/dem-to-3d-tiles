"""
Tile writer: saves terrain heightmaps to EXR files with metadata.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import OpenEXR

from .heightmap_generator import Heightmap
from .quadtree import Cell

logger = logging.getLogger(__name__)


def write_exr(heightmap: Heightmap, path: Path) -> None:
    """Write a heightmap to a single-channel float32 EXR file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    resolution = heightmap.resolution
    elevations = heightmap.elevations.astype(np.float32)

    header = {
        "compression": OpenEXR.ZIP_COMPRESSION,
        "type": OpenEXR.scanlineimage,
    }
    channels = {"Y": elevations}
    exr = OpenEXR.File(header, channels)
    exr.write(str(path))


def write_tile(heightmap: Heightmap, cell: Cell, output_dir: Path) -> Path:
    """Write a heightmap tile and return the output file path."""
    rel_path = cell.tile_path + ".exr"
    out_path = output_dir / rel_path
    write_exr(heightmap, out_path)
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
