"""CLI entry point for DEM to 3D tiles pipeline."""

import logging
import os

import click

from .chunk_pipeline import process_chunk as _process_chunk, process_global as _process_global
from .config import MAX_LOD, MESH_RESOLUTION, MIN_LOD, SPHERE_RADIUS
from .pipeline import run_pipeline


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def cli(verbose: bool):
    """Convert Copernicus DEM GeoTIFF to cube-sphere terrain heightmap tiles."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@cli.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True),
    required=True,
    help="Directory containing DEM GeoTIFF files.",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    required=True,
    help="Output directory for heightmap tiles.",
)
@click.option("--min-lod", type=int, default=MIN_LOD, help="Minimum LOD level.")
@click.option("--max-lod", type=int, default=MAX_LOD, help="Maximum LOD level.")
@click.option(
    "--resolution",
    type=int,
    default=MESH_RESOLUTION,
    help="Heightmap pixels per cell edge.",
)
@click.option(
    "--sphere-radius",
    type=float,
    default=SPHERE_RADIUS,
    help="Sphere radius in output units.",
)
@click.option(
    "--faces",
    type=str,
    default=None,
    help="Comma-separated face indices to process (e.g., '0,1,2'). Default: all.",
)
@click.option(
    "--workers",
    type=int,
    default=None,
    help=(
        "Number of parallel worker processes. "
        f"Defaults to min(cpu_count, 8) = {min(os.cpu_count() or 1, 8)}. "
        "Use 1 for sequential/debug mode."
    ),
)
def process(
    input_dir: str,
    output_dir: str,
    min_lod: int,
    max_lod: int,
    resolution: int,
    sphere_radius: float,
    faces: str | None,
    workers: int | None,
):
    """Process DEM data into cube-sphere terrain heightmap tiles."""
    face_list = None
    if faces is not None:
        face_list = [int(f.strip()) for f in faces.split(",")]

    run_pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        min_lod=min_lod,
        max_lod=max_lod,
        resolution=resolution,
        sphere_radius=sphere_radius,
        faces=face_list,
        num_workers=workers,
    )


def _parse_bbox(bbox_str: str) -> tuple[float, float, float, float]:
    """Parse a 'lat_min,lon_min,lat_max,lon_max' string into a tuple."""
    parts = [float(x.strip()) for x in bbox_str.split(",")]
    if len(parts) != 4:
        raise click.BadParameter("BBOX must have 4 values: lat_min,lon_min,lat_max,lon_max")
    return (parts[0], parts[1], parts[2], parts[3])


@cli.command("process-chunk")
@click.option("--bbox", type=str, required=True, help="lat_min,lon_min,lat_max,lon_max")
@click.option("--download-dir", type=click.Path(), required=True, help="Directory for temporary DEM downloads.")
@click.option("--output-dir", type=click.Path(), required=True, help="Output directory for tiles.")
@click.option("--min-lod", type=int, default=MIN_LOD, help="Minimum LOD level.")
@click.option("--max-lod", type=int, default=MAX_LOD, help="Maximum LOD level.")
@click.option("--resolution", type=int, default=MESH_RESOLUTION, help="Heightmap pixels per cell edge.")
@click.option("--sphere-radius", type=float, default=SPHERE_RADIUS, help="Sphere radius in output units.")
@click.option("--faces", type=str, default=None, help="Comma-separated face indices.")
@click.option("--workers", type=int, default=None, help="Number of parallel worker processes.")
@click.option("--cleanup/--no-cleanup", default=False, help="Remove downloaded DEMs after processing.")
def process_chunk_cmd(
    bbox: str,
    download_dir: str,
    output_dir: str,
    min_lod: int,
    max_lod: int,
    resolution: int,
    sphere_radius: float,
    faces: str | None,
    workers: int | None,
    cleanup: bool,
):
    """Download and process DEM data for a single geographic BBOX."""
    parsed_bbox = _parse_bbox(bbox)
    face_list = None
    if faces is not None:
        face_list = [int(f.strip()) for f in faces.split(",")]

    _process_chunk(
        bbox=parsed_bbox,
        download_dir=download_dir,
        output_dir=output_dir,
        min_lod=min_lod,
        max_lod=max_lod,
        resolution=resolution,
        sphere_radius=sphere_radius,
        faces=face_list,
        num_workers=workers,
        cleanup=cleanup,
    )


@cli.command("process-global")
@click.option("--download-dir", type=click.Path(), required=True, help="Directory for temporary DEM downloads.")
@click.option("--output-dir", type=click.Path(), required=True, help="Output directory for tiles.")
@click.option("--chunk-size", type=float, default=15.0, help="Degrees per chunk edge.")
@click.option("--global-bbox", type=str, default=None, help="lat_min,lon_min,lat_max,lon_max (default: full Earth).")
@click.option("--min-lod", type=int, default=MIN_LOD, help="Minimum LOD level.")
@click.option("--max-lod", type=int, default=MAX_LOD, help="Maximum LOD level.")
@click.option("--resolution", type=int, default=MESH_RESOLUTION, help="Heightmap pixels per cell edge.")
@click.option("--sphere-radius", type=float, default=SPHERE_RADIUS, help="Sphere radius in output units.")
@click.option("--faces", type=str, default=None, help="Comma-separated face indices.")
@click.option("--workers", type=int, default=None, help="Number of parallel worker processes.")
@click.option("--cleanup/--no-cleanup", default=False, help="Remove downloaded DEMs after processing.")
@click.option("--force/--no-force", default=False, help="Reprocess completed chunks.")
def process_global_cmd(
    download_dir: str,
    output_dir: str,
    chunk_size: float,
    global_bbox: str | None,
    min_lod: int,
    max_lod: int,
    resolution: int,
    sphere_radius: float,
    faces: str | None,
    workers: int | None,
    cleanup: bool,
    force: bool,
):
    """Download and process DEM data for the entire globe in chunks."""
    parsed_global_bbox = None
    if global_bbox is not None:
        parsed_global_bbox = _parse_bbox(global_bbox)

    face_list = None
    if faces is not None:
        face_list = [int(f.strip()) for f in faces.split(",")]

    _process_global(
        download_dir=download_dir,
        output_dir=output_dir,
        chunk_size=chunk_size,
        global_bbox=parsed_global_bbox,
        min_lod=min_lod,
        max_lod=max_lod,
        resolution=resolution,
        sphere_radius=sphere_radius,
        faces=face_list,
        num_workers=workers,
        cleanup=cleanup,
        force=force,
    )


if __name__ == "__main__":
    cli()
