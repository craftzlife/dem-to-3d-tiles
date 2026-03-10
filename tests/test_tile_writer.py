"""Tests for tile writer (EXR output)."""

import json
from pathlib import Path

import numpy as np
import OpenEXR
import pytest

from src.heightmap_generator import Heightmap
from src.quadtree import Cell
from src.tile_writer import write_exr, write_manifest, write_tile


class TestWriteExr:
    def test_roundtrip(self, tmp_path):
        """Write an EXR and read it back — float32 values should match exactly."""
        data = np.random.default_rng(42).uniform(-100, 8000, (32, 32)).astype(np.float32)
        hm = Heightmap(elevations=data)
        exr_path = tmp_path / "test.exr"
        write_exr(hm, exr_path)

        assert exr_path.exists()

        # Read back
        exr_file = OpenEXR.File(str(exr_path))
        channels = exr_file.channels()
        assert "Y" in channels
        read_back = channels["Y"].pixels
        np.testing.assert_array_equal(read_back, data)

    def test_creates_parent_dirs(self, tmp_path):
        data = np.ones((4, 4), dtype=np.float32)
        hm = Heightmap(elevations=data)
        exr_path = tmp_path / "a" / "b" / "c" / "test.exr"
        write_exr(hm, exr_path)
        assert exr_path.exists()


class TestWriteTile:
    def test_produces_exr_path(self, tmp_path):
        data = np.zeros((4, 4), dtype=np.float32)
        hm = Heightmap(elevations=data)
        cell = Cell(face=0, level=1, ix=0, iy=1)
        path = write_tile(hm, cell, tmp_path)
        assert path.suffix == ".exr"
        assert path.exists()


class TestWriteManifest:
    def test_writes_valid_json(self, tmp_path):
        manifest_data = {
            "version": "2.0",
            "format": "exr",
            "tiles": [{"face": 0, "level": 0}],
        }
        write_manifest(manifest_data, tmp_path)
        manifest_path = tmp_path / "manifest.json"
        assert manifest_path.exists()
        loaded = json.loads(manifest_path.read_text())
        assert loaded["version"] == "2.0"
        assert loaded["format"] == "exr"
        assert len(loaded["tiles"]) == 1
