"""
Quadtree cell system for cube-sphere faces.

Each cube face is recursively subdivided into a quadtree. A cell is identified
by (face, level, ix, iy) where:
    - face: 0-5 (cube face index)
    - level: 0 = full face, higher = finer subdivision
    - ix, iy: cell indices in [0, 2^level - 1]

At level L, each face has 2^L × 2^L = 4^L cells.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Cell:
    """A quadtree cell on a cube face."""

    face: int
    level: int
    ix: int
    iy: int

    @property
    def size(self) -> int:
        """Number of cells per edge at this level."""
        return 1 << self.level

    @property
    def u_range(self) -> tuple[float, float]:
        """UV range along u-axis: (u_min, u_max) in [0, 1]."""
        s = self.size
        return self.ix / s, (self.ix + 1) / s

    @property
    def v_range(self) -> tuple[float, float]:
        """UV range along v-axis: (v_min, v_max) in [0, 1]."""
        s = self.size
        return self.iy / s, (self.iy + 1) / s

    @property
    def children(self) -> list[Cell]:
        """Return the 4 child cells at the next level."""
        child_level = self.level + 1
        bx, by = self.ix * 2, self.iy * 2
        return [
            Cell(self.face, child_level, bx, by),
            Cell(self.face, child_level, bx + 1, by),
            Cell(self.face, child_level, bx, by + 1),
            Cell(self.face, child_level, bx + 1, by + 1),
        ]

    @property
    def parent(self) -> Cell | None:
        """Return the parent cell, or None if at level 0."""
        if self.level == 0:
            return None
        return Cell(self.face, self.level - 1, self.ix // 2, self.iy // 2)

    @property
    def tile_path(self) -> str:
        """Relative path for this cell's tile file (without extension)."""
        from .config import FACE_NAMES

        return f"{FACE_NAMES[self.face]}/lod_{self.level}/{self.ix}_{self.iy}"

    def __str__(self) -> str:
        return f"Cell(face={self.face}, L{self.level}, {self.ix}_{self.iy})"


def iter_cells_at_level(face: int, level: int):
    """Yield all cells for a given face at a given level."""
    size = 1 << level
    for iy in range(size):
        for ix in range(size):
            yield Cell(face, level, ix, iy)


def root_cell(face: int) -> Cell:
    """Return the root cell (level 0) for a face."""
    return Cell(face, 0, 0, 0)
