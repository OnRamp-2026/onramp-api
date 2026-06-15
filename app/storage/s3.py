from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

from app.storage.base import ObjectMetadata, ObjectNotFoundError, ObjectStorageError, PresignedUpload


class S3ObjectStorage:
    def __init__(self, client: Any, bucket: str, *, presign_client: Any | None = None) -> None:
        self.client = client
        self.presign_client = presign_client or client
        self.bucket = bucket

    async def create_presigned_upload(
        self,
        object_key: str,
        *,
        content_type: str,
        expires_in_seconds: int,
    ) -> PresignedUpload:
        try:
            url = await asyncio.to_thread(
                self.presign_client.generate_presigned_url,
                "put_object",
                Params={
                    "Bucket": self.bucket,
                    "Key": object_key,
                    "ContentType": content_type,
                },
                ExpiresIn=expires_in_seconds,
            )
        except Exception as exc:
            raise ObjectStorageError("Failed to create presigned upload URL.") from exc
        return PresignedUpload(
            method="PUT",
            url=url,
            headers={"Content-Type": content_type},
            expires_at=datetime.now(UTC) + timedelta(seconds=expires_in_seconds),
        )

    async def head(self, object_key: str) -> ObjectMetadata:
        try:
            response = await asyncio.to_thread(
                self.client.head_object,
                Bucket=self.bucket,
                Key=object_key,
            )
        except Exception as exc:
            response = getattr(exc, "response", {})
            status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if status == 404:
                raise ObjectNotFoundError(f"Object not found: {object_key}") from exc
            raise ObjectStorageError(f"Failed to inspect object: {object_key}") from exc

        return ObjectMetadata(
            object_key=object_key,
            size_bytes=int(response["ContentLength"]),
            content_type=str(response.get("ContentType") or "application/octet-stream"),
            etag=str(response.get("ETag") or ""),
        )
