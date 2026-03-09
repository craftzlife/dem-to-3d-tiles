# DEM to 3D Tiles

Convert [Copernicus DEM 30M](https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model) GeoTIFF elevation data into cube-sphere terrain mesh tiles, designed as the data pipeline for a Unity anime-styled global map.

The globe is projected onto **6 cube faces** using S2-style gnomonic projection, with **quadtree-based multi-LOD** subdivision. Only cells overlapping actual DEM data produce output — empty regions are skipped.

## Quick Start

Requires Docker.

```bash
# Build
docker compose build

# Run the pipeline
docker compose run --rm dem2tiles process \
  --input-dir /app/data/raw \
  --output-dir /app/data/processed

# Run tests
docker compose run --rm test
```

Place Copernicus DEM GeoTIFF files (`.tif`) anywhere under `data/raw/` before running.

## CLI Usage

```bash
docker compose run --rm dem2tiles [OPTIONS] process [PROCESS_OPTIONS]
```

### Global Options

| Flag | Description |
|------|-------------|
| `-v, --verbose` | Enable debug logging |

### Process Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input-dir PATH` | *(required)* | Directory containing DEM GeoTIFF files |
| `--output-dir PATH` | *(required)* | Output directory for mesh tiles |
| `--min-lod` | `0` | Minimum LOD level (coarsest) |
| `--max-lod` | `14` | Maximum LOD level (finest) |
| `--resolution` | `256` | Mesh vertices per cell edge |
| `--sphere-radius` | `100.0` | Sphere radius in Unity units |
| `--elevation-scale` | `30.0` | Elevation exaggeration factor |
| `--faces` | all | Comma-separated face indices, e.g. `0,1,2` |

### Examples

```bash
# Quick preview with low LOD
docker compose run --rm dem2tiles process \
  --input-dir /app/data/raw \
  --output-dir /app/data/processed \
  --max-lod 3 --resolution 32

# Realistic elevation (no exaggeration)
docker compose run --rm dem2tiles process \
  --input-dir /app/data/raw \
  --output-dir /app/data/processed \
  --elevation-scale 1.0

# Process only specific cube faces
docker compose run --rm dem2tiles process \
  --input-dir /app/data/raw \
  --output-dir /app/data/processed \
  --faces 1,3
```

## Cube-Sphere Model

The globe is projected onto 6 cube faces, then each face is inflated onto a sphere (gnomonic projection, following [S2 Geometry](https://s2geometry.io/) principles):

```
        ┌─────────┐
        │  Face 2  │
        │   +Z     │
        │ (north)  │
  ┌─────┼─────────┼─────────┬─────────┐
  │ F3  │  Face 0  │  Face 1 │  Face 4 │
  │ -X  │   +X     │   +Y    │   -Y    │
  └─────┼─────────┼─────────┴─────────┘
        │  Face 5  │
        │   -Z     │
        │ (south)  │
        └─────────┘
```

| Face | Axis | Directory |
|------|------|-----------|
| 0 | +X | `face_0_pos_x/` |
| 1 | +Y | `face_1_pos_y/` |
| 2 | +Z (north pole) | `face_2_pos_z/` |
| 3 | -X | `face_3_neg_x/` |
| 4 | -Y | `face_4_neg_y/` |
| 5 | -Z (south pole) | `face_5_neg_z/` |

Each face is recursively subdivided via a quadtree. Level 0 = 1 tile per face, level N = 4^N tiles per face.

## Output Structure

```
data/processed/
├── manifest.json                  # Tile index with metadata
├── face_1_pos_y/
│   ├── lod_0/0_0.obj              # Coarse: entire face
│   ├── lod_1/1_1.obj              # 2×2 grid
│   ├── lod_2/2_2.obj              # 4×4 grid
│   └── lod_3/4_4.obj, 5_4.obj    # 8×8 grid
└── face_3_neg_x/
    └── ...
```

- **`manifest.json`** — lists all generated tiles with face, level, cell coordinates, vertex/triangle counts
- **`{ix}_{iy}.obj`** — Wavefront OBJ mesh with vertices, normals, UVs, and triangles
- Only tiles overlapping DEM data are generated

## Elevation Scaling

Terrain height is computed as:

```
vertex_radius = sphere_radius + elevation_meters × (1/6,371,000) × elevation_scale × sphere_radius
```

| `elevation_scale` | Effect | Use case |
|-------------------|--------|----------|
| `1.0` | True-to-life proportions | Realistic globe |
| `5.0` | Subtle exaggeration | Visible terrain on zoomed-out view |
| `30.0` | Dramatic terrain (default) | Anime-styled maps |

With scale `1.0`, Mount Everest (8,849m) displaces only ~0.14 units on a 100-unit sphere — Earth is very smooth at global scale.

## Architecture

```
DEM GeoTIFFs → DEMReader index → enumerate quadtree cells per LOD
  → filter by DEM overlap → sample elevation → generate mesh → write OBJ
```

| Module | Purpose |
|--------|---------|
| `src/cube_sphere.py` | S2-style gnomonic projection (lat/lon <-> face UV <-> XYZ) |
| `src/quadtree.py` | `Cell(face, level, ix, iy)` with parent/children traversal |
| `src/dem_reader.py` | Indexes and reads Copernicus GeoTIFF tiles via rasterio |
| `src/mesh_generator.py` | Samples DEM, builds triangle mesh on cube-sphere surface |
| `src/tile_writer.py` | Writes OBJ files and manifest.json |
| `src/pipeline.py` | Orchestrates the full processing flow |
| `src/main.py` | Click CLI entry point |

## Testing

```bash
# All tests
docker compose run --rm test

# Specific test file
docker compose run --rm test -v tests/test_cube_sphere.py

# Single test
docker compose run --rm test -v tests/test_cube_sphere.py::TestRoundTrip::test_latlng_roundtrip
```

## Preview Model

Generate a standalone cube-sphere OBJ to visualize the 6-face structure:

```bash
docker compose run --rm \
  -v ./scripts:/app/scripts \
  -v ./data:/app/data \
  --entrypoint python3 dem2tiles \
  scripts/generate_cubesphere_preview.py
```

Output: `data/cubesphere_preview.obj` — open in Blender, Unity, or any 3D viewer. Faces are separated by small gaps to show the cube structure.

## Input Data

- **Format**: Copernicus DEM 30M COG GeoTIFF (1° × 1° tiles, ~30m resolution)
- **Location**: `data/raw/`
- **Source**: [Copernicus Space Data](https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model)

Both `data/raw/` and `data/processed/` are gitignored.
