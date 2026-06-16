import httpx

from app.config import Settings
from app.db.opensearch import OpenSearchClient


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
