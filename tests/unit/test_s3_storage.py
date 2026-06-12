from __future__ import annotations

import pytest

from app.storage.s3 import S3ObjectStorage


class FakeS3Client:
    def __init__(self) -> None:
        self.presign_call: tuple[str, dict[str, object], int] | None = None

    def generate_presigned_url(
        self,
        operation: str,
        **kwargs: object,
    ) -> str:
        params = kwargs["Params"]
        expires_in = kwargs["ExpiresIn"]
        assert isinstance(params, dict)
        assert isinstance(expires_in, int)
        self.presign_call = (operation, params, expires_in)
        return "https://storage.test/signed"

    def head_object(self, **kwargs: object) -> dict[str, object]:
        assert kwargs["Bucket"] == "onramp-stt"
        assert kwargs["Key"] == "tenants/tenant-a/source.m4a"
        return {
            "ContentLength": 1024,
            "ContentType": "audio/mp4",
            "ETag": '"etag-1"',
        }


@pytest.mark.asyncio
async def test_s3_storage_presigns_exact_key_and_content_type() -> None:
    client = FakeS3Client()
    storage = S3ObjectStorage(client, "onramp-stt")

    upload = await storage.create_presigned_upload(
        "tenants/tenant-a/source.m4a",
        content_type="audio/mp4",
        expires_in_seconds=900,
    )

    assert upload.method == "PUT"
    assert upload.url == "https://storage.test/signed"
    assert upload.headers == {"Content-Type": "audio/mp4"}
    assert client.presign_call == (
        "put_object",
        {
            "Bucket": "onramp-stt",
            "Key": "tenants/tenant-a/source.m4a",
            "ContentType": "audio/mp4",
        },
        900,
    )


@pytest.mark.asyncio
async def test_s3_storage_reads_head_metadata() -> None:
    storage = S3ObjectStorage(FakeS3Client(), "onramp-stt")

    metadata = await storage.head("tenants/tenant-a/source.m4a")

    assert metadata.size_bytes == 1024
    assert metadata.content_type == "audio/mp4"
    assert metadata.etag == '"etag-1"'
