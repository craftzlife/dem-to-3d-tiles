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
# ELEVATION_SCALE. Set it to 1.0 for real-world proportions:
#   ┌─────────────────┬───────────────────────┬────────────────────┐
#   │     Setting     │ Anime-style (current) │ Original/realistic │
#   ├─────────────────┼───────────────────────┼────────────────────┤
#   │ ELEVATION_SCALE │ 30.0                  │ 1.0                │
#   └─────────────────┴───────────────────────┴────────────────────┘
ELEVATION_SCALE = 5.0  # Exaggeration factor for anime-style terrain

ELEVATION_BASE_SCALE = 1.0 / 6_371_000  # Convert meters to unit-sphere scale

# DEM source
DEM_NODATA = -32768.0

# Output formats
OUTPUT_FORMAT_OBJ = "obj"
OUTPUT_FORMAT_PLY = "ply"
DEFAULT_OUTPUT_FORMAT = OUTPUT_FORMAT_OBJ

# Face names for directory structure
FACE_NAMES = [
    "face_0_pos_x",
    "face_1_pos_y",
    "face_2_pos_z",
    "face_3_neg_x",
    "face_4_neg_y",
    "face_5_neg_z",
]
