from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class PresignedUpload:
    method: str
    url: str
    headers: dict[str, str]
    expires_at: datetime


@dataclass(frozen=True)
class ObjectMetadata:
    object_key: str
    size_bytes: int
    content_type: str
    etag: str


class ObjectStorageError(Exception):
    pass


class ObjectNotFoundError(ObjectStorageError):
    pass


class ObjectStorage(Protocol):
    async def create_presigned_upload(
        self,
        object_key: str,
        *,
        content_type: str,
        expires_in_seconds: int,
    ) -> PresignedUpload: ...

    async def head(self, object_key: str) -> ObjectMetadata: ...
