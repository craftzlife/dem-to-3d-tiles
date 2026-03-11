"""Tests for S3 downloader tile key generation."""

import pytest

from src.s3_downloader import copernicus_tile_key


class TestCopernicusTileKey:
    """Test S3 key generation for various lat/lon combinations."""

    def test_north_east(self):
        key = copernicus_tile_key(45, 10)
        assert key == (
            "Copernicus_DSM_COG_10_N45_00_E010_00_DEM/"
            "Copernicus_DSM_COG_10_N45_00_E010_00_DEM.tif"
        )

    def test_south_west(self):
        key = copernicus_tile_key(-30, -120)
        assert key == (
            "Copernicus_DSM_COG_10_S30_00_W120_00_DEM/"
            "Copernicus_DSM_COG_10_S30_00_W120_00_DEM.tif"
        )

    def test_equator_prime_meridian(self):
        key = copernicus_tile_key(0, 0)
        assert key == (
            "Copernicus_DSM_COG_10_N00_00_E000_00_DEM/"
            "Copernicus_DSM_COG_10_N00_00_E000_00_DEM.tif"
        )

    def test_south_zero_lon(self):
        key = copernicus_tile_key(-1, 0)
        assert key == (
            "Copernicus_DSM_COG_10_S01_00_E000_00_DEM/"
            "Copernicus_DSM_COG_10_S01_00_E000_00_DEM.tif"
        )

    def test_antimeridian_east(self):
        key = copernicus_tile_key(10, 179)
        assert key == (
            "Copernicus_DSM_COG_10_N10_00_E179_00_DEM/"
            "Copernicus_DSM_COG_10_N10_00_E179_00_DEM.tif"
        )

    def test_antimeridian_west(self):
        key = copernicus_tile_key(10, -180)
        assert key == (
            "Copernicus_DSM_COG_10_N10_00_W180_00_DEM/"
            "Copernicus_DSM_COG_10_N10_00_W180_00_DEM.tif"
        )

    def test_north_pole(self):
        key = copernicus_tile_key(89, 0)
        assert key == (
            "Copernicus_DSM_COG_10_N89_00_E000_00_DEM/"
            "Copernicus_DSM_COG_10_N89_00_E000_00_DEM.tif"
        )

    def test_south_pole(self):
        key = copernicus_tile_key(-90, 0)
        assert key == (
            "Copernicus_DSM_COG_10_S90_00_E000_00_DEM/"
            "Copernicus_DSM_COG_10_S90_00_E000_00_DEM.tif"
        )

    def test_zero_padding(self):
        """Lat is 2-digit, lon is 3-digit zero-padded."""
        key = copernicus_tile_key(5, 9)
        assert "N05_00_E009_00" in key

    def test_large_lon(self):
        key = copernicus_tile_key(0, -179)
        assert "W179_00" in key
