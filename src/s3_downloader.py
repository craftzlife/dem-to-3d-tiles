"""
Download Copernicus DEM 30M tiles from the public S3 bucket.

Tiles are 1x1 degree GeoTIFFs stored in the copernicus-dem-30m bucket.
No AWS credentials required (public bucket with unsigned access).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import botocore
from botocore import UNSIGNED
from botocore.config import Config

logger = logging.getLogger(__name__)

BUCKET = "copernicus-dem-30m"
REGION = "eu-central-1"


def copernicus_tile_key(lat: int, lon: int) -> str:
    """
    Generate the S3 object key for a Copernicus DEM 30M 1x1 degree tile.

    Args:
        lat: Latitude of the SW corner (integer degrees, -90 to 89).
        lon: Longitude of the SW corner (integer degrees, -180 to 179).

    Returns:
        S3 object key string.
    """
    lat_prefix = "N" if lat >= 0 else "S"
    lon_prefix = "E" if lon >= 0 else "W"
    lat_str = f"{abs(lat):02d}"
    lon_str = f"{abs(lon):03d}"

    tile_name = (
        f"Copernicus_DSM_COG_10_{lat_prefix}{lat_str}_00_{lon_prefix}{lon_str}_00_DEM"
    )
    return f"{tile_name}/{tile_name}.tif"


def download_chunk(
    bbox: tuple[float, float, float, float],
    output_dir: str | Path,
    bucket: str = BUCKET,
    region: str = REGION,
    max_workers: int = 8,
) -> list[Path]:
    """
    Download all DEM tiles within a geographic bounding box from S3.

    Args:
        bbox: (lat_min, lon_min, lat_max, lon_max) in degrees.
        output_dir: Directory to save downloaded .tif files.
        bucket: S3 bucket name.
        region: AWS region for the bucket.
        max_workers: Number of parallel download threads.

    Returns:
        List of paths to successfully downloaded files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lat_min, lon_min, lat_max, lon_max = bbox

    # Build list of (lat, lon) integer tile corners within the BBOX
    tiles_to_download: list[tuple[int, int]] = []
    lat = int(lat_min) if lat_min == int(lat_min) else int(lat_min) if lat_min >= 0 else int(lat_min) - 1
    # Simplify: iterate integer lats from floor(lat_min) to ceil(lat_max)-1
    import math
    lat_start = math.floor(lat_min)
    lat_end = math.ceil(lat_max)
    lon_start = math.floor(lon_min)
    lon_end = math.ceil(lon_max)

    for lat in range(lat_start, lat_end):
        for lon in range(lon_start, lon_end):
            tiles_to_download.append((lat, lon))

    logger.info(
        f"Downloading chunk bbox={bbox}: {len(tiles_to_download)} potential tiles"
    )

    s3_config = Config(
        signature_version=UNSIGNED,
        region_name=region,
        retries={"max_attempts": 3, "mode": "standard"},
        max_pool_connections=max_workers,
        read_timeout=60,
        connect_timeout=10,
    )
    s3_client = boto3.client("s3", config=s3_config)

    downloaded: list[Path] = []
    max_retries = 3

    def _download_one(lat: int, lon: int) -> Path | None:
        key = copernicus_tile_key(lat, lon)
        filename = key.split("/")[-1]
        local_path = output_dir / filename
        if local_path.exists():
            logger.debug(f"Already exists: {filename}")
            return local_path

        for attempt in range(1, max_retries + 1):
            try:
                # Use get_object instead of download_file to avoid s3transfer
                # spawning extra threads that exhaust the connection pool.
                response = s3_client.get_object(Bucket=bucket, Key=key)
                # Write to a temp file first, then rename atomically.
                # This prevents interrupted downloads from leaving corrupt
                # files that the resume logic would skip.
                tmp_path = local_path.with_suffix(".tif.tmp")
                try:
                    with open(tmp_path, "wb") as f:
                        for chunk in response["Body"].iter_chunks(1024 * 1024):
                            f.write(chunk)
                    tmp_path.rename(local_path)
                except BaseException:
                    tmp_path.unlink(missing_ok=True)
                    raise
                logger.debug(f"Downloaded: {filename}")
                return local_path
            except botocore.exceptions.ClientError as e:
                error_code = e.response["Error"]["Code"]
                if error_code in ("404", "NoSuchKey"):
                    logger.debug(f"Not found (ocean/no data): {key}")
                    return None
                if attempt < max_retries:
                    logger.warning(f"Retry {attempt}/{max_retries} for {filename}: {e}")
                    continue
                logger.error(f"Failed after {max_retries} attempts: {filename}: {e}")
                return None
            except (TimeoutError, OSError, botocore.exceptions.BotoCoreError) as e:
                if attempt < max_retries:
                    logger.warning(f"Retry {attempt}/{max_retries} for {filename}: {e}")
                    continue
                logger.error(f"Failed after {max_retries} attempts: {filename}: {e}")
                return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_download_one, lat, lon): (lat, lon)
            for lat, lon in tiles_to_download
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                downloaded.append(result)

    logger.info(
        f"Downloaded {len(downloaded)}/{len(tiles_to_download)} tiles for bbox={bbox}"
    )
    return downloaded


def cleanup_chunk(download_dir: str | Path) -> None:
    """Remove all .tif files in the directory to free disk space."""
    download_dir = Path(download_dir)
    removed = 0
    for tif in download_dir.glob("*.tif"):
        tif.unlink()
        removed += 1
    logger.info(f"Cleaned up {removed} .tif files from {download_dir}")
