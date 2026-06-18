import httpx
import pytest

from app.config import Settings
from app.db.opensearch import OpenSearchClient


def _bulk_handler(record: list[str], *, errors: bool = False):
    """alias 존재(ensure_index no-op) + /_bulk 응답을 주는 핸들러. bulk 본문을 record에 적재."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "GET" and path == "/_alias/onramp-chunks":
            return httpx.Response(200, json={})  # alias 존재 → mapping 보강 경로
        if path == "/onramp-chunks/_mapping":
            return httpx.Response(200, json={"acknowledged": True})
        if path == "/_bulk":
            record.append(request.read().decode())
            items = [{"index": {"status": 400, "error": "bad"}}] if errors else []
            return httpx.Response(200, json={"errors": errors, "items": items})
        return httpx.Response(200, json={})

    return handler


def _bulk_client(handler) -> OpenSearchClient:
    return OpenSearchClient(
        settings=Settings(),
        http_client=httpx.AsyncClient(base_url="http://opensearch:9200", transport=httpx.MockTransport(handler)),
    )


async def test_upsert_chunks_uses_bulk_ndjson():
    # #212: per-doc PUT 대신 _bulk 1요청(NDJSON: action+source × N).
    bodies: list[str] = []
    client = _bulk_client(_bulk_handler(bodies))
    await client.upsert_chunks([{"chunk_id": "p_000", "content": "a"}, {"chunk_id": "p_001", "content": "b"}])
    assert len(bodies) == 1  # 2건 → 1 bulk 요청
    lines = bodies[0].strip().split("\n")
    assert len(lines) == 4  # action+source × 2
    assert '"_id": "p_000"' in lines[0]


async def test_upsert_chunks_batches_over_limit():
    # 문서 수 > _BULK_MAX_DOCS면 여러 bulk 요청으로 쪼갠다.
    bodies: list[str] = []
    client = _bulk_client(_bulk_handler(bodies))
    docs = [{"chunk_id": f"p_{i:04d}", "content": "x"} for i in range(1100)]  # 500*2 + 100
    await client.upsert_chunks(docs)
    assert len(bodies) == 3


async def test_upsert_chunks_raises_on_bulk_errors():
    # _bulk가 errors:true면 부분 실패를 즉시 드러낸다.
    client = _bulk_client(_bulk_handler([], errors=True))
    with pytest.raises(RuntimeError, match="bulk 색인 실패"):
        await client.upsert_chunks([{"chunk_id": "p_000", "content": "a"}])


async def test_ensure_index_creates_index_with_alias_when_missing():
    requests: list[tuple[str, str, dict | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read()
        requests.append((request.method, request.url.path, httpx.Response(200, content=body).json() if body else None))
        if request.method == "GET" and request.url.path == "/_alias/onramp-chunks":
            return httpx.Response(404)
        if request.method == "HEAD" and request.url.path == "/onramp-chunks-v1":
            return httpx.Response(404)
        return httpx.Response(200, json={"acknowledged": True})

    client = OpenSearchClient(
        settings=Settings(),
        http_client=httpx.AsyncClient(base_url="http://opensearch:9200", transport=httpx.MockTransport(handler)),
    )

    await client.ensure_index()

    assert ("PUT", "/onramp-chunks-v1") == requests[-1][:2]
    assert requests[-1][2]["aliases"] == {"onramp-chunks": {}}


async def test_delete_index_deletes_concrete(monkeypatch):
    # #212: 임시 평가 인덱스 정리 — concrete 인덱스를 DELETE한다.
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        return httpx.Response(200, json={"acknowledged": True})

    client = OpenSearchClient(
        settings=Settings(opensearch_index_v1="onramp-eval-token-abc-v1"),
        http_client=httpx.AsyncClient(base_url="http://opensearch:9200", transport=httpx.MockTransport(handler)),
    )

    await client.delete_index()
    assert ("DELETE", "/onramp-eval-token-abc-v1") in seen


async def test_delete_index_missing_is_noop():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)  # 없는 인덱스 → 에러 아님

    client = OpenSearchClient(
        settings=Settings(),
        http_client=httpx.AsyncClient(base_url="http://opensearch:9200", transport=httpx.MockTransport(handler)),
    )
    await client.delete_index()  # 예외 없이 통과해야 한다


async def test_search_builds_tenant_and_ladder_filters():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.read().decode()
        return httpx.Response(
            200, json={"hits": {"hits": [{"_id": "c1", "_score": 3.0, "_source": {"chunk_id": "c1"}}]}}
        )

    client = OpenSearchClient(
        settings=Settings(),
        http_client=httpx.AsyncClient(base_url="http://opensearch:9200", transport=httpx.MockTransport(handler)),
    )

    hits = await client.search(
        "장애 대응",
        top_k=10,
        tenant_id="tenant-a",
        domain="incident",
        version="v1",
        pinned_doc_keys=("doc-a",),
        excluded_doc_keys=("doc-b",),
    )

    assert hits[0].id == "c1"
    assert '"tenant_id":"tenant-a"' in captured["body"]
    assert '"domain":"incident"' in captured["body"]
    assert '"product_version":"v1"' in captured["body"]
    assert '"doc-a"' in captured["body"]
    assert '"doc-b"' in captured["body"]
