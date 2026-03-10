"""CLI entry point for DEM to 3D tiles pipeline."""

import logging
import os

import click

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


if __name__ == "__main__":
    cli()
