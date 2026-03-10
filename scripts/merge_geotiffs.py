"""Merge multiple GeoTIFF files into a single output file.

Uses rasterio to mosaic overlapping or adjacent GeoTIFF tiles into one
continuous raster, preserving the CRS and resolution of the input files.

Usage:
    python scripts/merge_geotiffs.py INPUT_DIR OUTPUT_FILE [--pattern GLOB] [--nodata VALUE] [--resampling METHOD]

Examples:
    # Merge all .tif files in a directory
    python scripts/merge_geotiffs.py data/raw/ data/merged/dem_merged.tif

    # Merge only files matching a pattern
    python scripts/merge_geotiffs.py data/raw/ data/merged/dem_merged.tif --pattern "Copernicus_*.tif"

    # Specify nodata value and resampling method
    python scripts/merge_geotiffs.py data/raw/ data/merged/dem_merged.tif --nodata -9999 --resampling bilinear
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling


def merge_geotiffs(
    input_dir: Path,
    output_path: Path,
    pattern: str = "*.tif",
    nodata: float | None = None,
    resampling: str = "nearest",
) -> None:
    tif_files = sorted(input_dir.rglob(pattern))
    if not tif_files:
        print(f"No files matching '{pattern}' found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(tif_files)} GeoTIFF files to merge")

    datasets = [rasterio.open(f) for f in tif_files]

    try:
        resampling_method = Resampling[resampling]
    except KeyError:
        valid = [r.name for r in Resampling]
        print(f"Invalid resampling method '{resampling}'. Valid: {', '.join(valid)}", file=sys.stderr)
        sys.exit(1)

    # Use nodata from first file if not specified
    if nodata is None:
        nodata = datasets[0].nodata

    print(f"CRS: {datasets[0].crs}")
    print(f"Nodata: {nodata}")
    print(f"Resampling: {resampling}")
    print("Merging...")

    mosaic, transform = merge(datasets, nodata=nodata, resampling=resampling_method)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    profile = datasets[0].profile.copy()
    profile.update(
        driver="GTiff",
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        transform=transform,
        count=mosaic.shape[0],
        nodata=nodata,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mosaic)

    for ds in datasets:
        ds.close()

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Written: {output_path} ({size_mb:.1f} MB)")
    print(f"Shape: {mosaic.shape[2]}x{mosaic.shape[1]} ({mosaic.shape[0]} band(s))")


def main():
    parser = argparse.ArgumentParser(description="Merge multiple GeoTIFF files into one.")
    parser.add_argument("input_dir", type=Path, help="Directory containing input GeoTIFF files")
    parser.add_argument("output", type=Path, help="Output GeoTIFF file path")
    parser.add_argument("--pattern", default="*.tif", help="Glob pattern for input files (default: *.tif)")
    parser.add_argument("--nodata", type=float, default=None, help="Nodata value (default: from first input file)")
    parser.add_argument("--resampling", default="nearest", help="Resampling method (default: nearest)")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"Input directory not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    merge_geotiffs(args.input_dir, args.output, args.pattern, args.nodata, args.resampling)


if __name__ == "__main__":
    main()
