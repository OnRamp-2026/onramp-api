"""OpenSearchClient 문서(원문) 인덱스 — MockTransport로 upsert/get/search 검증."""

import httpx

from app.config import Settings
from app.db.opensearch import OpenSearchClient


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    method = request.method
    # ensure_documents_index: alias 없음 → concrete head 404 → create
    if path == "/_alias/onramp-documents":
        return httpx.Response(404, json={})
    if path == "/onramp-documents-v1" and method == "HEAD":
        return httpx.Response(404)
    if path == "/onramp-documents-v1" and method == "PUT":
        return httpx.Response(200, json={"acknowledged": True})
    # upsert
    if path.startswith("/onramp-documents/_doc/") and method == "PUT":
        return httpx.Response(201, json={"result": "created"})
    # get_document
    if path == "/onramp-documents/_doc/onramp:d1" and method == "GET":
        return httpx.Response(200, json={"_source": {"doc_id": "d1", "content": "박병선 교수님 자문 내용"}})
    if path == "/onramp-documents/_doc/onramp:missing" and method == "GET":
        return httpx.Response(404, json={"found": False})
    # mget
    if path == "/onramp-documents/_mget":
        return httpx.Response(
            200,
            json={"docs": [{"found": True, "_source": {"doc_id": "d1"}}, {"found": False}]},
        )
    # search
    if path == "/onramp-documents/_search":
        return httpx.Response(
            200,
            json={"hits": {"hits": [{"_id": "onramp:d1", "_score": 4.2, "_source": {"doc_id": "d1"}}]}},
        )
    return httpx.Response(404, json={})


def _client() -> OpenSearchClient:
    http = httpx.AsyncClient(base_url="http://os:9200", transport=httpx.MockTransport(_handler))
    return OpenSearchClient(settings=Settings(), http_client=http)


async def test_upsert_documents_creates_index_and_puts():
    os = _client()
    await os.upsert_documents([{"tenant_id": "onramp", "doc_id": "d1", "content": "원문"}])  # 예외 없으면 성공


async def test_upsert_documents_empty_noop():
    os = _client()
    await os.upsert_documents([])  # 빈 입력 → no-op


async def test_get_document_found_and_missing():
    os = _client()
    found = await os.get_document("d1", tenant_id="onramp")
    assert found is not None and found["content"] == "박병선 교수님 자문 내용"
    missing = await os.get_document("missing", tenant_id="onramp")
    assert missing is None


async def test_get_documents_filters_not_found():
    os = _client()
    docs = await os.get_documents(["d1", "d2"], tenant_id="onramp")
    assert [d["doc_id"] for d in docs] == ["d1"]  # found=false 제외


async def test_search_documents_returns_hits():
    os = _client()
    hits = await os.search_documents("박병선", top_k=5, tenant_id="onramp")
    assert len(hits) == 1
    assert hits[0].id == "onramp:d1" and hits[0].score == 4.2
