"""
DEM (Digital Elevation Model) reader for Copernicus GeoTIFF files.

Indexes all .tif files in the input directory by their geographic bounds
and provides elevation lookup by lat/lon coordinates.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio
from rasterio.transform import rowcol

from .config import DEM_NODATA

logger = logging.getLogger(__name__)


@dataclass
class TileInfo:
    """Metadata for a single DEM tile."""

    path: Path
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    transform: rasterio.transform.Affine
    width: int
    height: int


class DEMReader:
    """Reads elevation data from a directory of Copernicus DEM GeoTIFF tiles."""

    def __init__(self, input_dir: str | Path):
        self.input_dir = Path(input_dir)
        self.tiles: list[TileInfo] = []
        self._open_datasets: dict[Path, rasterio.DatasetReader] = {}
        # Band cache: stores the full elevation array for each opened tile path
        self._band_cache: dict[Path, np.ndarray] = {}
        # Spatial index: (floor_lat, floor_lon) -> list of TileInfo
        self._spatial_index: dict[tuple[int, int], list[TileInfo]] = {}
        self._index_tiles()

    def _index_tiles(self):
        """Scan input directory for GeoTIFF files and index their bounds."""
        tif_files = sorted(self.input_dir.rglob("*.tif"))
        if not tif_files:
            logger.warning(f"No .tif files found in {self.input_dir}")
            return

        for tif_path in tif_files:
            try:
                with rasterio.open(tif_path) as ds:
                    bounds = ds.bounds
                    self.tiles.append(
                        TileInfo(
                            path=tif_path,
                            lat_min=bounds.bottom,
                            lat_max=bounds.top,
                            lon_min=bounds.left,
                            lon_max=bounds.right,
                            transform=ds.transform,
                            width=ds.width,
                            height=ds.height,
                        )
                    )
            except Exception as e:
                logger.warning(f"Failed to read {tif_path}: {e}")

        logger.info(f"Indexed {len(self.tiles)} DEM tiles")
        self._build_spatial_index()

    def _build_spatial_index(self):
        """
        Build a spatial index from integer (floor_lat, floor_lon) keys to tiles.

        Copernicus DEM tiles are typically 1°×1° so each tile is reliably
        addressable by the integer floor of its lat/lon min corner. Tiles
        spanning more than one degree cell are added to all covered cells.
        """
        index: dict[tuple[int, int], list[TileInfo]] = {}
        for tile in self.tiles:
            lat_lo = math.floor(tile.lat_min)
            lat_hi = math.floor(tile.lat_max)
            lon_lo = math.floor(tile.lon_min)
            lon_hi = math.floor(tile.lon_max)
            for lat_cell in range(lat_lo, lat_hi + 1):
                for lon_cell in range(lon_lo, lon_hi + 1):
                    key = (lat_cell, lon_cell)
                    index.setdefault(key, []).append(tile)
        self._spatial_index = index

    @property
    def bounds(self) -> tuple[float, float, float, float] | None:
        """Overall geographic bounds (lat_min, lat_max, lon_min, lon_max)."""
        if not self.tiles:
            return None
        return (
            min(t.lat_min for t in self.tiles),
            max(t.lat_max for t in self.tiles),
            min(t.lon_min for t in self.tiles),
            max(t.lon_max for t in self.tiles),
        )

    def _find_tile(self, lat: float, lon: float) -> TileInfo | None:
        """Find the tile containing the given lat/lon using spatial index (O(1))."""
        key = (math.floor(lat), math.floor(lon))
        candidates = self._spatial_index.get(key)
        if candidates is None:
            return None
        for tile in candidates:
            if (
                tile.lat_min <= lat <= tile.lat_max
                and tile.lon_min <= lon <= tile.lon_max
            ):
                return tile
        return None

    def _get_dataset(self, tile: TileInfo) -> rasterio.DatasetReader:
        """Get or open a rasterio dataset for the tile (cached)."""
        if tile.path not in self._open_datasets:
            self._open_datasets[tile.path] = rasterio.open(tile.path)
        return self._open_datasets[tile.path]

    def _get_band(self, tile: TileInfo) -> np.ndarray:
        """
        Return the full elevation band for a tile, reading and caching on first access.

        Each tile is ~49 MB (3601×3601 float32). Adjacent cells frequently share
        the same tile, so caching avoids repeated full-band reads.
        """
        if tile.path not in self._band_cache:
            ds = self._get_dataset(tile)
            self._band_cache[tile.path] = ds.read(1)
        return self._band_cache[tile.path]

    def get_elevation(self, lat: float, lon: float) -> float:
        """Get elevation at a single lat/lon point. Returns 0.0 if no data."""
        tile = self._find_tile(lat, lon)
        if tile is None:
            return 0.0

        ds = self._get_dataset(tile)
        row, col = rowcol(ds.transform, lon, lat)

        if 0 <= row < ds.height and 0 <= col < ds.width:
            val = ds.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]
            if val == DEM_NODATA or np.isnan(val):
                return 0.0
            return float(val)
        return 0.0

    def get_elevations(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """
        Get elevations for arrays of lat/lon coordinates (vectorized).

        Groups coordinates by source tile, performs a single numpy index into the
        cached band per tile, and eliminates the per-pixel Python loop entirely.

        Returns array of elevations in meters. Missing data returns 0.0.
        """
        result = np.zeros(lats.shape, dtype=np.float64)
        flat_lats = lats.ravel()
        flat_lons = lons.ravel()
        flat_result = result.ravel()

        # Group coordinate indices by their source tile using vectorized floor
        floor_lats = np.floor(flat_lats).astype(np.int32)
        floor_lons = np.floor(flat_lons).astype(np.int32)

        # Unique spatial-index keys present in this batch
        keys = set(zip(floor_lats.tolist(), floor_lons.tolist()))

        for key in keys:
            # Gather candidates from spatial index
            candidates = self._spatial_index.get(key, [])
            if not candidates:
                continue

            # Mask of coordinates falling in this index cell
            cell_mask = (floor_lats == key[0]) & (floor_lons == key[1])
            indices = np.where(cell_mask)[0]

            cell_lats = flat_lats[indices]
            cell_lons = flat_lons[indices]

            # Route each coordinate to its actual tile (usually just one)
            for tile in candidates:
                tile_mask = (
                    (cell_lats >= tile.lat_min) & (cell_lats <= tile.lat_max)
                    & (cell_lons >= tile.lon_min) & (cell_lons <= tile.lon_max)
                )
                if not np.any(tile_mask):
                    continue

                tile_indices = indices[tile_mask]
                t_lats = flat_lats[tile_indices]
                t_lons = flat_lons[tile_indices]

                # Batch rowcol transform
                ds = self._get_dataset(tile)
                rows, cols = rowcol(ds.transform, t_lons, t_lats)
                rows = np.asarray(rows, dtype=np.int32)
                cols = np.asarray(cols, dtype=np.int32)

                # Clamp to valid range
                valid = (rows >= 0) & (rows < tile.height) & (cols >= 0) & (cols < tile.width)

                if np.any(valid):
                    band = self._get_band(tile)
                    v_rows = rows[valid]
                    v_cols = cols[valid]
                    vals = band[v_rows, v_cols].astype(np.float64)

                    # Replace nodata / NaN with 0
                    vals = np.where(
                        (vals == DEM_NODATA) | np.isnan(vals), 0.0, vals
                    )
                    flat_result[tile_indices[valid]] = vals

        return flat_result.reshape(lats.shape)

    def has_data_in_region(
        self, lat_min: float, lat_max: float, lon_min: float, lon_max: float
    ) -> bool:
        """Check if any DEM tile overlaps the given geographic region using spatial index."""
        lat_lo = math.floor(lat_min)
        lat_hi = math.floor(lat_max)
        lon_lo = math.floor(lon_min)
        lon_hi = math.floor(lon_max)

        for lat_cell in range(lat_lo, lat_hi + 1):
            for lon_cell in range(lon_lo, lon_hi + 1):
                candidates = self._spatial_index.get((lat_cell, lon_cell))
                if candidates is None:
                    continue
                for tile in candidates:
                    if (
                        tile.lat_max >= lat_min
                        and tile.lat_min <= lat_max
                        and tile.lon_max >= lon_min
                        and tile.lon_min <= lon_max
                    ):
                        return True
        return False

    def close(self):
        """Close all open datasets and clear caches."""
        for ds in self._open_datasets.values():
            ds.close()
        self._open_datasets.clear()
        self._band_cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
