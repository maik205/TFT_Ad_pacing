"""
Object Storage Data Synchronization Script.

Enables seamless upload/download of raw iPinYou bidding datasets (`train.csv` and `test.csv`)
to/from any S3-compatible object store (AWS S3, Cloudflare R2, Railway Buckets, MinIO, Wasabi)
or direct HTTPS download.

Usage:
  # Upload from Windows:
  python scripts/sync_data.py upload --bucket my-rtb-bucket --prefix data/

  # Download on Mac M4:
  python scripts/sync_data.py download --bucket my-rtb-bucket --prefix data/
"""

import os
import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

RAW_FILES = ["train.csv", "test.csv"]


def get_s3_client(endpoint_url=None, region_name=None):
    try:
        import boto3
    except ImportError:
        logger.error("boto3 is required for object storage sync. Install it via: pip install boto3")
        sys.exit(1)

    endpoint = endpoint_url or os.environ.get("S3_ENDPOINT_URL")
    region = region_name or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

    client_kwargs = {
        'service_name': 's3',
        'region_name': region,
    }
    if endpoint:
        client_kwargs['endpoint_url'] = endpoint
    if key_id and secret_key:
        client_kwargs['aws_access_key_id'] = key_id
        client_kwargs['aws_secret_access_key'] = secret_key

    return boto3.client(**client_kwargs)


def upload_files(bucket: str, prefix: str = "", endpoint_url: str = None, region: str = None):
    s3 = get_s3_client(endpoint_url=endpoint_url, region_name=region)
    logger.info(f"Uploading raw CSV files to s3://{bucket}/{prefix}...")

    for fname in RAW_FILES:
        if not os.path.exists(fname):
            logger.warning(f"File {fname} not found locally. Skipping.")
            continue

        file_size_gb = os.path.getsize(fname) / (1024 ** 3)
        key = f"{prefix.rstrip('/')}/{fname}".lstrip('/')
        logger.info(f"Uploading {fname} ({file_size_gb:.2f} GB) to {key}...")
        
        # Upload with progress callback
        s3.upload_file(fname, bucket, key)
        logger.info(f"  Successfully uploaded {fname} -> s3://{bucket}/{key}")


def download_files(bucket: str, prefix: str = "", endpoint_url: str = None, region: str = None):
    s3 = get_s3_client(endpoint_url=endpoint_url, region_name=region)
    logger.info(f"Downloading raw CSV files from s3://{bucket}/{prefix}...")

    for fname in RAW_FILES:
        key = f"{prefix.rstrip('/')}/{fname}".lstrip('/')
        logger.info(f"Downloading s3://{bucket}/{key} -> {fname}...")
        try:
            s3.download_file(bucket, key, fname)
            file_size_gb = os.path.getsize(fname) / (1024 ** 3)
            logger.info(f"  Successfully downloaded {fname} ({file_size_gb:.2f} GB).")
        except Exception as e:
            logger.error(f"  Failed to download {fname}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Sync raw iPinYou CSV data with object storage.")
    parser.add_argument("action", choices=["upload", "download"], help="Action to perform.")
    parser.add_argument("--bucket", type=str, default=os.environ.get("S3_BUCKET_NAME", "ipinyou-dataset"), help="Bucket name.")
    parser.add_argument("--prefix", type=str, default="", help="Key prefix in bucket.")
    parser.add_argument("--endpoint_url", type=str, default=None, help="Custom S3 endpoint (e.g. for R2 or MinIO).")
    parser.add_argument("--region", type=str, default=None, help="AWS region.")
    args = parser.parse_args()

    if args.action == "upload":
        upload_files(args.bucket, args.prefix, args.endpoint_url, args.region)
    elif args.action == "download":
        download_files(args.bucket, args.prefix, args.endpoint_url, args.region)


if __name__ == '__main__':
    main()
