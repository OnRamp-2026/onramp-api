from qdrant_client.models import ScoredPoint

from app.agents.retriever import tools as tools_mod
from app.agents.retriever.search import SearchFilters
from app.agents.retriever.tools import SearchToolContext, search_bm25, search_dense
from app.config import Settings
from app.db.opensearch import OpenSearchHit


class _Embedder:
    async def embed_query(self, query: str) -> list[float]:
        return [0.1, 0.2]


async def test_search_dense_uses_pure_dense_path(monkeypatch) -> None:
    captured: dict = {}

    async def fake_dense(query_vector, top_k, *, domain, filters, settings):
        captured.update(
            query_vector=query_vector,
            top_k=top_k,
            domain=domain,
            filters=filters,
            settings=settings,
        )
        return [ScoredPoint(id="c1", version=0, score=0.9, payload={"chunk_id": "c1", "content": "dense"})]

    settings = Settings(hybrid_search_enabled=True)
    filters = SearchFilters(version="2.4", pinned_doc_keys=("apache:mpm",))
    context = SearchToolContext(
        tenant_id="tenant1-onramp",
        domain="manual",
        filters=filters,
        top_k=7,
        settings=settings,
    )
    monkeypatch.setattr(tools_mod, "get_embedder", lambda: _Embedder())
    monkeypatch.setattr(tools_mod, "dense_search", fake_dense)

    result = await search_dense("설정 방법", context=context)

    assert result[0].payload["content"] == "dense"
    assert captured["query_vector"] == [0.1, 0.2]
    assert captured["top_k"] == 7
    assert captured["domain"] == "manual"
    assert captured["filters"] is filters


async def test_search_bm25_forces_tenant_and_server_filters(monkeypatch) -> None:
    captured: dict = {}

    class _OpenSearch:
        async def search(self, query: str, **kwargs):
            captured.update(query=query, **kwargs)
            return [OpenSearchHit(id="c2", score=3.2, payload={"chunk_id": "c2", "content": "bm25"})]

    filters = SearchFilters(
        version="v1.33",
        pinned_doc_keys=("k8s:upgrade",),
        excluded_doc_keys=("k8s:done",),
    )
    context = SearchToolContext(
        tenant_id="tenant1-onramp",
        domain="manual",
        filters=filters,
        top_k=9,
        settings=Settings(),
    )
    monkeypatch.setattr(tools_mod, "get_opensearch", lambda: _OpenSearch())

    result = await search_bm25("v1.33 upgrade", context=context)

    assert result[0].id == "c2"
    assert captured == {
        "query": "v1.33 upgrade",
        "top_k": 9,
        "tenant_id": "tenant1-onramp",
        "domain": "manual",
        "version": "v1.33",
        "pinned_doc_keys": ("k8s:upgrade",),
        "excluded_doc_keys": ("k8s:done",),
    }
