"""Configuration constants for DEM to 3D tiles pipeline."""

# Cube-sphere geometry
NUM_FACES = 6
SPHERE_RADIUS = 100.0  # Unity units

# Quadtree LOD levels
MIN_LOD = 0
MAX_LOD = 14  # 0=coarsest (full face), 7=finest

# Mesh resolution: vertices per cell edge at each LOD
MESH_RESOLUTION = 256

# Elevation
ELEVATION_BASE_SCALE = 1.0 / 6_371_000  # Convert meters to unit-sphere scale

# DEM source
DEM_NODATA = -32768.0

# Output format
OUTPUT_FORMAT = "exr"

# Face names for directory structure
FACE_NAMES = [
    "face_0_pos_x",
    "face_1_pos_y",
    "face_2_pos_z",
    "face_3_neg_x",
    "face_4_neg_y",
    "face_5_neg_z",
]
