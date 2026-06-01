"""임베딩 — 색인/검색 공용. provider 추상화(OpenAI P0 / self-hosted BGE-M3 P1)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from openai import AsyncOpenAI

from app.config import Settings, get_settings


@runtime_checkable
class Embedder(Protocol):
    dim: int

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...


class OpenAIEmbedder:
    """OpenAI text-embedding-3-small (P0)."""

    def __init__(self, settings: Settings | None = None, client: AsyncOpenAI | None = None) -> None:
        self.settings = settings or get_settings()
        self.model = self.settings.embedding_model
        self.dim = self.settings.embedding_dim
        self._client = client

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        return self._client

    async def embed_documents(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        # rate limit 고려해 batch_size 단위로 분할 요청
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            resp = await self.client.embeddings.create(model=self.model, input=texts[start : start + batch_size])
            vectors.extend(item.embedding for item in resp.data)
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        resp = await self.client.embeddings.create(model=self.model, input=[text])
        return resp.data[0].embedding


class SelfHostedEmbedder:
    """BGE-M3 self-hosted (VesslAI GPU). P1 — 미구현."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.dim = self.settings.embedding_dim

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("self-hosted 임베딩은 P1")

    async def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError("self-hosted 임베딩은 P1")


_embedder: Embedder | None = None


def get_embedder(settings: Settings | None = None) -> Embedder:
    """llm_provider에 따라 임베더 싱글톤 반환."""
    global _embedder
    if _embedder is None:
        settings = settings or get_settings()
        if settings.llm_provider == "self_hosted":
            _embedder = SelfHostedEmbedder(settings)
        else:  # openai | azure | 기본
            _embedder = OpenAIEmbedder(settings)
    return _embedder


def reset_embedder() -> None:
    """테스트용 싱글톤 초기화."""
    global _embedder
    _embedder = None
