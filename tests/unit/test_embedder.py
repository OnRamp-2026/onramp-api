import pytest

from app.config import Settings
from app.rag.embedder import (
    OpenAIEmbedder,
    SelfHostedEmbedder,
    get_embedder,
    reset_embedder,
)

DIM = 8


class _FakeEmbeddings:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.call_count = 0

    async def create(self, model, input):  # noqa: A002 - OpenAI SDK 시그니처
        self.call_count += 1
        data = [type("Item", (), {"embedding": [0.1] * self.dim})() for _ in input]
        return type("Resp", (), {"data": data})()


class _FakeClient:
    def __init__(self, dim: int) -> None:
        self.embeddings = _FakeEmbeddings(dim)


@pytest.fixture(autouse=True)
def _reset():
    reset_embedder()
    yield
    reset_embedder()


@pytest.mark.asyncio
async def test_embed_query_returns_dim_vector():
    embedder = OpenAIEmbedder(settings=Settings(embedding_dim=DIM), client=_FakeClient(DIM))
    vec = await embedder.embed_query("질문")
    assert len(vec) == DIM


@pytest.mark.asyncio
async def test_embed_documents_returns_one_vector_per_text():
    embedder = OpenAIEmbedder(settings=Settings(embedding_dim=DIM), client=_FakeClient(DIM))
    vecs = await embedder.embed_documents(["a", "b", "c"])
    assert len(vecs) == 3
    assert all(len(v) == DIM for v in vecs)


@pytest.mark.asyncio
async def test_embed_documents_batches():
    fake = _FakeClient(DIM)
    embedder = OpenAIEmbedder(settings=Settings(embedding_dim=DIM), client=fake)
    await embedder.embed_documents(["x"] * 5, batch_size=2)
    assert fake.embeddings.call_count == 3  # 2 + 2 + 1


def test_get_embedder_dispatch_openai():
    assert isinstance(get_embedder(Settings(llm_provider="openai")), OpenAIEmbedder)


def test_get_embedder_dispatch_self_hosted():
    assert isinstance(get_embedder(Settings(llm_provider="self_hosted")), SelfHostedEmbedder)


@pytest.mark.asyncio
async def test_self_hosted_not_implemented():
    embedder = SelfHostedEmbedder(settings=Settings(embedding_dim=DIM))
    with pytest.raises(NotImplementedError):
        await embedder.embed_query("q")
