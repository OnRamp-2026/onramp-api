from functools import lru_cache

from app.config import get_settings
from app.storage.base import ObjectStorage
from app.storage.s3 import S3ObjectStorage


@lru_cache
def get_storage() -> ObjectStorage:
    import boto3

    settings = get_settings()
    access_key = settings.storage_access_key.get_secret_value()
    secret_key = settings.storage_secret_key.get_secret_value()
    client = boto3.client(
        "s3",
        endpoint_url=settings.storage_endpoint_url or None,
        region_name=settings.storage_region,
        aws_access_key_id=access_key or None,
        aws_secret_access_key=secret_key or None,
    )
    presign_client = None
    if settings.storage_public_endpoint_url:
        presign_client = boto3.client(
            "s3",
            endpoint_url=settings.storage_public_endpoint_url,
            region_name=settings.storage_region,
            aws_access_key_id=access_key or None,
            aws_secret_access_key=secret_key or None,
        )
    return S3ObjectStorage(client, settings.storage_bucket, presign_client=presign_client)
