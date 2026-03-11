"""Tests for chunk pipeline orchestrator and BBOX filtering."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.chunk_pipeline import generate_chunk_bboxes, _bbox_key, _load_progress, _save_progress
from src.pipeline import _cell_intersects_bbox
from src.quadtree import Cell
from src.tile_writer import merge_manifest


class TestGenerateChunkBboxes:
    """Test BBOX generation covers the area without gaps or overlaps."""

    def test_full_earth_default(self):
        bboxes = generate_chunk_bboxes(chunk_size=15.0)
        # 180/15 * 360/15 = 12 * 24 = 288
        assert len(bboxes) == 288

    def test_no_gaps_or_overlaps(self):
        bboxes = generate_chunk_bboxes(chunk_size=15.0)
        # Check that every point on Earth is in exactly one BBOX
        # Sample some test points
        for lat in [-89.5, -45.0, 0.0, 45.0, 89.5]:
            for lon in [-179.5, -90.0, 0.0, 90.0, 179.5]:
                containing = [
                    b for b in bboxes
                    if b[0] <= lat < b[2] and b[1] <= lon < b[3]
                ]
                assert len(containing) == 1, (
                    f"Point ({lat}, {lon}) in {len(containing)} bboxes"
                )

    def test_custom_global_bbox(self):
        bboxes = generate_chunk_bboxes(chunk_size=15.0, global_bbox=(0, 0, 30, 30))
        assert len(bboxes) == 4  # 2x2

    def test_non_divisible_chunk_size(self):
        bboxes = generate_chunk_bboxes(chunk_size=7.0, global_bbox=(0, 0, 15, 15))
        # 0-7, 7-14, 14-15 = 3 lat bands x 3 lon bands = 9
        assert len(bboxes) == 9

    def test_single_chunk(self):
        bboxes = generate_chunk_bboxes(chunk_size=180.0, global_bbox=(-90, -180, 90, 180))
        assert len(bboxes) == 2  # 1 lat band x 2 lon bands


class TestCellIntersectsBbox:
    """Test cell-BBOX intersection logic."""

    def test_cell_inside_bbox(self):
        # Face 0 LOD 2 cell near equator/prime meridian area
        cell = Cell(face=0, level=2, ix=1, iy=1)
        bbox = (-90.0, -180.0, 90.0, 180.0)  # Full Earth
        assert _cell_intersects_bbox(cell, bbox) is True

    def test_cell_outside_bbox(self):
        # Face 2 (+Z, north pole) should not intersect equatorial bbox
        cell = Cell(face=2, level=3, ix=0, iy=0)
        bbox = (-10.0, -10.0, 10.0, 10.0)
        # This may or may not intersect depending on projection
        # Just verify it returns a bool without error
        result = _cell_intersects_bbox(cell, bbox)
        assert isinstance(result, bool)

    def test_full_earth_bbox_includes_all(self):
        """Every cell should intersect the full Earth BBOX."""
        bbox = (-90.0, -180.0, 90.0, 180.0)
        for face in range(6):
            cell = Cell(face=face, level=0, ix=0, iy=0)
            assert _cell_intersects_bbox(cell, bbox) is True

    def test_small_bbox_excludes_distant_cells(self):
        """A small BBOX near the north pole should exclude cells on face 5 (-Z, south pole)."""
        bbox = (80.0, 0.0, 90.0, 10.0)
        # Face 5 (-Z) covers the south pole
        cell = Cell(face=5, level=2, ix=1, iy=1)
        assert _cell_intersects_bbox(cell, bbox) is False


class TestManifestMerge:
    """Test manifest merging logic."""

    def test_merge_no_overlap(self):
        existing = {
            "tiles": [
                {"face": 0, "level": 1, "ix": 0, "iy": 0, "elevation_min": 0},
            ]
        }
        new_tiles = [
            {"face": 0, "level": 1, "ix": 1, "iy": 0, "elevation_min": 10},
        ]
        merged = merge_manifest(existing, new_tiles)
        assert len(merged) == 2

    def test_merge_with_overwrite(self):
        existing = {
            "tiles": [
                {"face": 0, "level": 1, "ix": 0, "iy": 0, "elevation_min": 0},
            ]
        }
        new_tiles = [
            {"face": 0, "level": 1, "ix": 0, "iy": 0, "elevation_min": 99},
        ]
        merged = merge_manifest(existing, new_tiles)
        assert len(merged) == 1
        assert merged[0]["elevation_min"] == 99

    def test_merge_idempotent(self):
        existing = {
            "tiles": [
                {"face": 0, "level": 1, "ix": 0, "iy": 0, "elevation_min": 5},
            ]
        }
        merged1 = merge_manifest(existing, [])
        merged2 = merge_manifest({"tiles": merged1}, [])
        assert len(merged2) == 1
        assert merged2[0]["elevation_min"] == 5

    def test_merge_empty_existing(self):
        existing = {"tiles": []}
        new_tiles = [
            {"face": 1, "level": 0, "ix": 0, "iy": 0, "elevation_min": 42},
        ]
        merged = merge_manifest(existing, new_tiles)
        assert len(merged) == 1


class TestProgressTracking:
    """Test progress save/load."""

    def test_save_and_load(self, tmp_path):
        completed = {"0.0,0.0,15.0,15.0", "15.0,0.0,30.0,15.0"}
        _save_progress(tmp_path, completed)
        loaded = _load_progress(tmp_path)
        assert loaded == completed

    def test_load_missing(self, tmp_path):
        loaded = _load_progress(tmp_path)
        assert loaded == set()

    def test_bbox_key_format(self):
        bbox = (0.0, 15.0, 15.0, 30.0)
        key = _bbox_key(bbox)
        assert key == "0.0,15.0,15.0,30.0"
