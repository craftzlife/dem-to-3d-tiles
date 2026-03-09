"""
DEM (Digital Elevation Model) reader for Copernicus GeoTIFF files.

Indexes all .tif files in the input directory by their geographic bounds
and provides elevation lookup by lat/lon coordinates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

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
        """Find the tile containing the given lat/lon."""
        for tile in self.tiles:
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
        Get elevations for arrays of lat/lon coordinates.

        Returns array of elevations in meters. Missing data returns 0.0.
        """
        elevations = np.zeros(lats.shape, dtype=np.float64)
        flat_lats = lats.ravel()
        flat_lons = lons.ravel()
        flat_elevs = elevations.ravel()

        for i in range(len(flat_lats)):
            flat_elevs[i] = self.get_elevation(flat_lats[i], flat_lons[i])

        return flat_elevs.reshape(lats.shape)

    def has_data_in_region(
        self, lat_min: float, lat_max: float, lon_min: float, lon_max: float
    ) -> bool:
        """Check if any DEM tile overlaps the given geographic region."""
        for tile in self.tiles:
            if (
                tile.lat_max >= lat_min
                and tile.lat_min <= lat_max
                and tile.lon_max >= lon_min
                and tile.lon_min <= lon_max
            ):
                return True
        return False

    def close(self):
        """Close all open datasets."""
        for ds in self._open_datasets.values():
            ds.close()
        self._open_datasets.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
