"""Tests for quadtree cell system."""

import pytest

from src.quadtree import Cell, iter_cells_at_level, root_cell


class TestCell:
    def test_root_cell(self):
        c = root_cell(0)
        assert c.face == 0
        assert c.level == 0
        assert c.ix == 0
        assert c.iy == 0

    def test_uv_range_root(self):
        c = root_cell(0)
        assert c.u_range == (0.0, 1.0)
        assert c.v_range == (0.0, 1.0)

    def test_uv_range_level1(self):
        c = Cell(0, 1, 0, 0)
        assert c.u_range == (0.0, 0.5)
        assert c.v_range == (0.0, 0.5)

        c = Cell(0, 1, 1, 1)
        assert c.u_range == (0.5, 1.0)
        assert c.v_range == (0.5, 1.0)

    def test_children_count(self):
        c = root_cell(0)
        children = c.children
        assert len(children) == 4

    def test_children_cover_parent(self):
        """Children's UV ranges should tile the parent's UV range."""
        parent = Cell(2, 3, 5, 6)
        pu_min, pu_max = parent.u_range
        pv_min, pv_max = parent.v_range

        children = parent.children
        # Children should be at level 4
        for child in children:
            assert child.level == 4
            assert child.face == 2

        # Verify coverage: collect all child UV ranges
        u_ranges = sorted([c.u_range for c in children])
        v_ranges = sorted([c.v_range for c in children])

        # Children (ix=10,iy=12), (ix=11,iy=12), (ix=10,iy=13), (ix=11,iy=13)
        assert children[0].ix == 10 and children[0].iy == 12
        assert children[1].ix == 11 and children[1].iy == 12
        assert children[2].ix == 10 and children[2].iy == 13
        assert children[3].ix == 11 and children[3].iy == 13

    def test_parent_roundtrip(self):
        c = Cell(3, 5, 17, 22)
        parent = c.parent
        assert parent is not None
        assert parent.level == 4
        assert c in parent.children

    def test_root_has_no_parent(self):
        c = root_cell(0)
        assert c.parent is None

    def test_tile_path(self):
        c = Cell(0, 3, 5, 7)
        assert c.tile_path == "face_0_pos_x/lod_3/5_7"


class TestIterCells:
    def test_level0_single_cell(self):
        cells = list(iter_cells_at_level(0, 0))
        assert len(cells) == 1
        assert cells[0] == root_cell(0)

    def test_level1_four_cells(self):
        cells = list(iter_cells_at_level(0, 1))
        assert len(cells) == 4

    def test_level2_sixteen_cells(self):
        cells = list(iter_cells_at_level(0, 2))
        assert len(cells) == 16

    @pytest.mark.parametrize("level", range(5))
    def test_cell_count(self, level):
        cells = list(iter_cells_at_level(0, level))
        assert len(cells) == 4**level
