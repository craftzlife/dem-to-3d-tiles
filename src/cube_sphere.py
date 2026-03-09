"""
S2-inspired cube-sphere projection.

Maps between geographic coordinates (lat/lon) and cube-face coordinates (face, u, v).
Uses gnomonic projection onto 6 cube faces, following S2 geometry conventions.

Face layout:
    Face 0: +X    Face 3: -X
    Face 1: +Y    Face 4: -Y
    Face 2: +Z    Face 5: -Z

UV coordinates are in [0, 1] range, mapped from the gnomonic [-1, 1] range.
"""

import math
import numpy as np


def latlng_to_xyz(lat_deg: float, lon_deg: float) -> tuple[float, float, float]:
    """Convert latitude/longitude (degrees) to unit sphere XYZ."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    x = math.cos(lat) * math.cos(lon)
    y = math.cos(lat) * math.sin(lon)
    z = math.sin(lat)
    return x, y, z


def latlng_to_xyz_batch(
    lat_deg: np.ndarray, lon_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized lat/lon to XYZ conversion."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return x, y, z


def xyz_to_latlng(x: float, y: float, z: float) -> tuple[float, float]:
    """Convert unit sphere XYZ to latitude/longitude (degrees)."""
    lat = math.degrees(math.asin(np.clip(z, -1.0, 1.0)))
    lon = math.degrees(math.atan2(y, x))
    return lat, lon


def xyz_to_face(x: float, y: float, z: float) -> int:
    """Determine which cube face a unit-sphere point belongs to."""
    ax, ay, az = abs(x), abs(y), abs(z)
    if ax >= ay and ax >= az:
        return 0 if x > 0 else 3
    elif ay >= ax and ay >= az:
        return 1 if y > 0 else 4
    else:
        return 2 if z > 0 else 5


def xyz_to_face_uv(
    x: float, y: float, z: float
) -> tuple[int, float, float]:
    """
    Convert unit sphere XYZ to face + UV coordinates.

    Returns (face, u, v) where u, v are in [0, 1].
    Uses gnomonic projection consistent with face_uv_to_xyz.
    """
    face = xyz_to_face(x, y, z)
    raw_u, raw_v = _xyz_to_raw_uv(face, x, y, z)
    # Map from [-1, 1] to [0, 1]
    u = (raw_u + 1.0) / 2.0
    v = (raw_v + 1.0) / 2.0
    return face, u, v


def _xyz_to_raw_uv(
    face: int, x: float, y: float, z: float
) -> tuple[float, float]:
    """Get raw UV in [-1, 1] for a given face."""
    if face == 0:
        return y / x, z / x
    elif face == 1:
        return -x / y, z / y
    elif face == 2:
        return -x / z, y / z
    elif face == 3:
        return y / x, -z / x
    elif face == 4:
        return -x / y, -z / y
    else:  # face == 5
        return -x / z, y / z


def face_uv_to_xyz(
    face: int, u: float, v: float
) -> tuple[float, float, float]:
    """
    Convert face + UV coordinates to unit sphere XYZ.

    Args:
        face: Cube face index (0-5).
        u, v: Coordinates in [0, 1] range.

    Returns:
        Normalized (x, y, z) on unit sphere.
    """
    # Map from [0, 1] to [-1, 1]
    s = u * 2.0 - 1.0
    t = v * 2.0 - 1.0

    if face == 0:
        x, y, z = 1.0, s, t
    elif face == 1:
        x, y, z = -s, 1.0, t
    elif face == 2:
        x, y, z = -s, t, 1.0
    elif face == 3:
        x, y, z = -1.0, -s, t
    elif face == 4:
        x, y, z = s, -1.0, t
    else:  # face == 5
        x, y, z = s, -t, -1.0

    norm = math.sqrt(x * x + y * y + z * z)
    return x / norm, y / norm, z / norm


def face_uv_to_xyz_batch(
    face: int, u: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized face UV to XYZ conversion."""
    s = u * 2.0 - 1.0
    t = v * 2.0 - 1.0

    if face == 0:
        x, y, z = np.ones_like(s), s, t
    elif face == 1:
        x, y, z = -s, np.ones_like(s), t
    elif face == 2:
        x, y, z = -s, t, np.ones_like(s)
    elif face == 3:
        x, y, z = -np.ones_like(s), -s, t
    elif face == 4:
        x, y, z = s, -np.ones_like(s), t
    else:  # face == 5
        x, y, z = s, -t, -np.ones_like(s)

    norm = np.sqrt(x * x + y * y + z * z)
    return x / norm, y / norm, z / norm


def face_uv_to_latlng(face: int, u: float, v: float) -> tuple[float, float]:
    """Convert face + UV to lat/lon (degrees)."""
    x, y, z = face_uv_to_xyz(face, u, v)
    return xyz_to_latlng(x, y, z)


def face_uv_to_latlng_batch(
    face: int, u: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized face UV to lat/lon conversion."""
    x, y, z = face_uv_to_xyz_batch(face, u, v)
    lat = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon


def latlng_to_face_uv(lat_deg: float, lon_deg: float) -> tuple[int, float, float]:
    """Convert lat/lon (degrees) to face + UV coordinates."""
    x, y, z = latlng_to_xyz(lat_deg, lon_deg)
    return xyz_to_face_uv(x, y, z)
