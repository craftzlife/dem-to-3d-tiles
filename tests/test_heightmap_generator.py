"""Tests for heightmap generator."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.heightmap_generator import Heightmap, generate_cell_heightmap
from src.quadtree import Cell


class TestHeightmap:
    def test_resolution(self):
        data = np.zeros((32, 32), dtype=np.float32)
        hm = Heightmap(elevations=data)
        assert hm.resolution == 32

    def test_dtype(self):
        data = np.ones((16, 16), dtype=np.float32) * 100.0
        hm = Heightmap(elevations=data)
        assert hm.elevations.dtype == np.float32

    def test_elevation_stats(self):
        data = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        hm = Heightmap(elevations=data)
        assert hm.elevation_min == 10.0
        assert hm.elevation_max == 40.0
        assert hm.elevation_mean == 25.0


class TestGenerateCellHeightmap:
    def _make_mock_dem(self, constant_elevation: float = 500.0):
        """Create a mock DEMReader that returns a constant elevation."""
        mock = MagicMock()
        mock.get_elevations.side_effect = lambda lats, lons: (
            np.full_like(lats, constant_elevation, dtype=np.float64)
        )
        return mock

    def test_output_shape(self):
        cell = Cell(face=0, level=1, ix=0, iy=0)
        dem = self._make_mock_dem()
        hm = generate_cell_heightmap(cell, dem, resolution=16)
        assert hm.elevations.shape == (16, 16)

    def test_output_dtype(self):
        cell = Cell(face=0, level=1, ix=0, iy=0)
        dem = self._make_mock_dem()
        hm = generate_cell_heightmap(cell, dem, resolution=8)
        assert hm.elevations.dtype == np.float32

    def test_constant_elevation(self):
        cell = Cell(face=2, level=0, ix=0, iy=0)
        dem = self._make_mock_dem(constant_elevation=1234.5)
        hm = generate_cell_heightmap(cell, dem, resolution=4)
        assert hm.elevation_min == pytest.approx(1234.5, abs=0.1)
        assert hm.elevation_max == pytest.approx(1234.5, abs=0.1)
        assert hm.elevation_mean == pytest.approx(1234.5, abs=0.1)

    def test_dem_reader_called(self):
        cell = Cell(face=0, level=2, ix=1, iy=1)
        dem = self._make_mock_dem()
        generate_cell_heightmap(cell, dem, resolution=8)
        dem.get_elevations.assert_called_once()
        args = dem.get_elevations.call_args[0]
        assert args[0].shape == (8, 8)  # lats
        assert args[1].shape == (8, 8)  # lons
