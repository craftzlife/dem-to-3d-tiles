# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Converts Copernicus DEM 30M GeoTIFF elevation data into cube-sphere terrain mesh tiles for a Unity anime-styled global map. The globe is projected onto 6 cube faces using S2 geometry principles, with quadtree-based multi-LOD subdivision.

## Build & Run (Docker)

```bash
# Build the Docker image
docker compose build

# Run the full pipeline
docker compose run dem2tiles process --input-dir /app/data/raw --output-dir /app/data/processed

# Run with custom parameters
docker compose run dem2tiles process \
  --input-dir /app/data/raw \
  --output-dir /app/data/processed \
  --min-lod 0 --max-lod 5 \
  --resolution 64 \
  --elevation-scale 30.0 \
  --faces 0,1

# Run tests
docker compose run test

# Run a single test file
docker compose run test -v tests/test_cube_sphere.py

# Run a specific test
docker compose run test -v tests/test_cube_sphere.py::TestRoundTrip::test_latlng_roundtrip

# Verbose pipeline output
docker compose run dem2tiles -v process --input-dir /app/data/raw --output-dir /app/data/processed
```

The Docker image is based on `osgeo/gdal:ubuntu-small-3.8.4` which provides GDAL/rasterio support. Source files are volume-mounted for development — code changes take effect without rebuilding.

## Architecture

### Processing Pipeline (`src/pipeline.py`)

```
DEM GeoTIFFs → DEMReader index → enumerate cells per LOD → filter by data overlap → generate mesh → write OBJ tile
```

The pipeline only generates tiles for quadtree cells that overlap with available DEM data, skipping empty ocean/no-data regions.

### Cube-Sphere Projection (`src/cube_sphere.py`)

Implements S2-style gnomonic projection between geographic coordinates and cube-face coordinates:

- **6 faces**: 0=+X, 1=+Y, 2=+Z, 3=-X, 4=-Y, 5=-Z
- **UV coordinates**: [0, 1] range on each face (mapped from gnomonic [-1, 1])
- **Forward**: `face_uv_to_xyz(face, u, v)` → normalized (x, y, z) on unit sphere
- **Inverse**: `xyz_to_face_uv(x, y, z)` → (face, u, v)
- **Batch variants**: `_batch` suffix functions accept numpy arrays for vectorized processing

The forward/inverse transforms are exact inverses — verified by roundtrip tests.

### Quadtree Cell System (`src/quadtree.py`)

`Cell(face, level, ix, iy)` — immutable dataclass identifying a tile:
- Level 0 = full face (1 cell), Level N = 4^N cells per face
- UV range: `[ix/2^L, (ix+1)/2^L]` × `[iy/2^L, (iy+1)/2^L]`
- `cell.children` → 4 child cells, `cell.parent` → parent cell
- `cell.tile_path` → output directory path like `face_0_pos_x/lod_3/5_7`

### Mesh Generation (`src/mesh_generator.py`)

For each cell: creates a UV grid → samples DEM elevation via lat/lon lookup → positions vertices on the cube-sphere surface with radial elevation displacement. Elevation is scaled by `ELEVATION_BASE_SCALE * ELEVATION_SCALE * SPHERE_RADIUS` to convert meters to exaggerated Unity units.

### Key Configuration (`src/config.py`)

| Constant | Default | Purpose |
|----------|---------|---------|
| `SPHERE_RADIUS` | 100.0 | Unity-space sphere radius |
| `ELEVATION_SCALE` | 30.0 | Terrain exaggeration for anime style |
| `MAX_LOD` | 7 | Deepest quadtree level |
| `MESH_RESOLUTION` | 64 | Vertices per cell edge (64×64 grid) |

### Output Structure

```
data/processed/
├── manifest.json              # Tile hierarchy + metadata
└── face_0_pos_x/
    ├── lod_0/0_0.obj          # Full-face mesh
    ├── lod_1/0_0.obj .. 1_1.obj
    └── lod_N/ix_iy.obj
```

Each OBJ contains vertices (with elevation), normals, UVs, and triangle faces. The manifest.json lists all generated tiles with their cell coordinates and mesh stats.

### DEM Reader (`src/dem_reader.py`)

Indexes all `.tif` files by geographic bounds on init. Provides `get_elevation(lat, lon)` and batch `get_elevations(lats, lons)` with rasterio. Datasets are opened lazily and cached. Returns 0.0 for coordinates outside DEM coverage.

## Data

- **Input**: `data/raw/` — Copernicus DEM 30M COG GeoTIFF tiles (1°×1° each, ~30m resolution)
- **Output**: `data/processed/` — OBJ mesh tiles + manifest.json
- Both directories are gitignored; only code is tracked
