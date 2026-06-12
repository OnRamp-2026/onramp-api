from functools import lru_cache

from app.config import get_settings
from app.storage.base import ObjectStorage
from app.storage.s3 import S3ObjectStorage


@lru_cache
def get_storage() -> ObjectStorage:
    import boto3

    settings = get_settings()
    client = boto3.client(
        "s3",
        endpoint_url=settings.storage_endpoint_url or None,
        region_name=settings.storage_region,
        aws_access_key_id=settings.storage_access_key or None,
        aws_secret_access_key=settings.storage_secret_key or None,
    )
    return S3ObjectStorage(client, settings.storage_bucket)
