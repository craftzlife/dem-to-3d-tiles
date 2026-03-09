"""Tests for cube-sphere projection."""

import math

import numpy as np
import pytest

from src.cube_sphere import (
    face_uv_to_latlng,
    face_uv_to_xyz,
    face_uv_to_xyz_batch,
    latlng_to_face_uv,
    latlng_to_xyz,
    xyz_to_face,
    xyz_to_face_uv,
    xyz_to_latlng,
)


class TestLatLngToXYZ:
    def test_equator_prime_meridian(self):
        x, y, z = latlng_to_xyz(0, 0)
        assert abs(x - 1.0) < 1e-10
        assert abs(y) < 1e-10
        assert abs(z) < 1e-10

    def test_north_pole(self):
        x, y, z = latlng_to_xyz(90, 0)
        assert abs(x) < 1e-10
        assert abs(y) < 1e-10
        assert abs(z - 1.0) < 1e-10

    def test_equator_90east(self):
        x, y, z = latlng_to_xyz(0, 90)
        assert abs(x) < 1e-10
        assert abs(y - 1.0) < 1e-10
        assert abs(z) < 1e-10


class TestXYZToFace:
    def test_positive_x(self):
        assert xyz_to_face(1, 0.5, 0.3) == 0

    def test_positive_y(self):
        assert xyz_to_face(0.3, 1, 0.5) == 1

    def test_positive_z(self):
        assert xyz_to_face(0.3, 0.5, 1) == 2

    def test_negative_x(self):
        assert xyz_to_face(-1, 0.5, 0.3) == 3

    def test_negative_y(self):
        assert xyz_to_face(0.3, -1, 0.5) == 4

    def test_negative_z(self):
        assert xyz_to_face(0.3, 0.5, -1) == 5


class TestRoundTrip:
    """Test that face_uv_to_xyz and xyz_to_face_uv are inverses."""

    @pytest.mark.parametrize("face", range(6))
    def test_roundtrip_center(self, face):
        """Center of each face should roundtrip exactly."""
        x, y, z = face_uv_to_xyz(face, 0.5, 0.5)
        f2, u2, v2 = xyz_to_face_uv(x, y, z)
        assert f2 == face
        assert abs(u2 - 0.5) < 1e-10
        assert abs(v2 - 0.5) < 1e-10

    @pytest.mark.parametrize("face", range(6))
    def test_roundtrip_various_points(self, face):
        """Various UV points should roundtrip correctly."""
        for u, v in [(0.2, 0.3), (0.8, 0.1), (0.5, 0.9), (0.1, 0.7)]:
            x, y, z = face_uv_to_xyz(face, u, v)
            f2, u2, v2 = xyz_to_face_uv(x, y, z)
            assert f2 == face, f"Face mismatch: {face} -> {f2} at uv=({u},{v})"
            assert abs(u2 - u) < 1e-10, f"U mismatch on face {face}"
            assert abs(v2 - v) < 1e-10, f"V mismatch on face {face}"

    def test_latlng_roundtrip(self):
        """lat/lon -> face_uv -> lat/lon should roundtrip."""
        test_points = [
            (10, 105),   # Vietnam
            (45, -73),   # Montreal
            (-33, 151),  # Sydney
            (0, 0),      # Gulf of Guinea
            (89, 0),     # Near north pole
        ]
        for lat, lon in test_points:
            face, u, v = latlng_to_face_uv(lat, lon)
            lat2, lon2 = face_uv_to_latlng(face, u, v)
            assert abs(lat2 - lat) < 1e-8, f"Lat mismatch for ({lat}, {lon})"
            assert abs(lon2 - lon) < 1e-8, f"Lon mismatch for ({lat}, {lon})"


class TestBatchConsistency:
    """Batch operations should match scalar operations."""

    @pytest.mark.parametrize("face", range(6))
    def test_face_uv_to_xyz_batch(self, face):
        u = np.array([0.1, 0.5, 0.9])
        v = np.array([0.2, 0.5, 0.8])
        bx, by, bz = face_uv_to_xyz_batch(face, u, v)

        for i in range(len(u)):
            sx, sy, sz = face_uv_to_xyz(face, u[i], v[i])
            assert abs(bx[i] - sx) < 1e-10
            assert abs(by[i] - sy) < 1e-10
            assert abs(bz[i] - sz) < 1e-10


class TestUnitSphere:
    """All outputs should lie on the unit sphere."""

    @pytest.mark.parametrize("face", range(6))
    def test_face_uv_to_xyz_on_sphere(self, face):
        for u in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
                x, y, z = face_uv_to_xyz(face, u, v)
                r = math.sqrt(x * x + y * y + z * z)
                assert abs(r - 1.0) < 1e-10, f"Not on unit sphere: r={r}"
