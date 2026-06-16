"""OpenSearch BM25 index adapter for hybrid retrieval."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

_client: OpenSearchClient | None = None


@dataclass(frozen=True)
class OpenSearchHit:
    """Provider-neutral BM25 hit used before RRF fusion."""

    id: str
    score: float
    payload: dict[str, Any]


class OpenSearchClient:
    """Tiny async REST client.

    OpenSearch Python client is intentionally avoided; httpx is already in the
    runtime and keeps the deployment surface small.
    """

    def __init__(self, settings: Settings | None = None, http_client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings or get_settings()
        self._owns_client = http_client is None
        self._http = http_client or httpx.AsyncClient(
            base_url=self._base_url(),
            timeout=self.settings.opensearch_timeout_seconds,
            auth=self._auth(),
        )

    def _base_url(self) -> str:
        return f"{self.settings.opensearch_scheme}://{self.settings.opensearch_host}:{self.settings.opensearch_port}"

    def _auth(self) -> tuple[str, str] | None:
        password = self.settings.opensearch_password.get_secret_value()
        if not self.settings.opensearch_username or not password:
            return None
        return (self.settings.opensearch_username, password)

    async def close(self) -> None:
        if self._owns_client:
            await self._http.aclose()

    async def ensure_index(self) -> None:
        alias = self.settings.opensearch_index
        concrete = self.settings.opensearch_index_v1
        alias_resp = await self._http.get(f"/_alias/{alias}")
        if alias_resp.status_code == 200:
            return
        if alias_resp.status_code != 404:
            alias_resp.raise_for_status()

        index_resp = await self._http.head(f"/{concrete}")
        if index_resp.status_code == 404:
            create_resp = await self._http.put(f"/{concrete}", json=_index_body(alias))
            create_resp.raise_for_status()
            return
        if index_resp.status_code >= 400:
            index_resp.raise_for_status()

        alias_create = await self._http.post(
            "/_aliases",
            json={"actions": [{"add": {"index": concrete, "alias": alias}}]},
        )
        alias_create.raise_for_status()

    async def upsert_chunks(self, documents: Sequence[Mapping[str, Any]]) -> None:
        if not documents:
            return
        await self.ensure_index()
        for document in documents:
            chunk_id = str(document["chunk_id"])
            resp = await self._http.put(f"/{self.settings.opensearch_index}/_doc/{chunk_id}", json=dict(document))
            resp.raise_for_status()

    async def delete_chunks(self, chunk_ids: Sequence[str]) -> None:
        if not chunk_ids:
            return
        for chunk_id in chunk_ids:
            resp = await self._http.delete(f"/{self.settings.opensearch_index}/_doc/{chunk_id}")
            if resp.status_code not in (200, 404):
                resp.raise_for_status()

    async def search(
        self,
        query: str,
        *,
        top_k: int,
        tenant_id: str,
        domain: str | None = None,
        version: str | None = None,
        pinned_doc_keys: tuple[str, ...] = (),
        excluded_doc_keys: tuple[str, ...] = (),
    ) -> list[OpenSearchHit]:
        body = _search_body(
            query,
            top_k=top_k,
            tenant_id=tenant_id,
            domain=domain,
            version=version,
            pinned_doc_keys=pinned_doc_keys,
            excluded_doc_keys=excluded_doc_keys,
        )
        resp = await self._http.post(f"/{self.settings.opensearch_index}/_search", json=body)
        resp.raise_for_status()
        hits = resp.json().get("hits", {}).get("hits", [])
        return [
            OpenSearchHit(
                id=str(hit.get("_id", "")), score=float(hit.get("_score") or 0.0), payload=hit.get("_source") or {}
            )
            for hit in hits
        ]


def _index_body(alias: str) -> dict[str, Any]:
    return {
        "aliases": {alias: {}},
        "settings": {
            "analysis": {
                "analyzer": {
                    "onramp_ko": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase"]},
                    "onramp_code": {"type": "custom", "tokenizer": "whitespace", "filter": ["lowercase"]},
                }
            }
        },
        "mappings": {
            "dynamic": "false",
            "properties": {
                "tenant_id": {"type": "keyword"},
                "chunk_id": {"type": "keyword"},
                "parent_id": {"type": "keyword"},
                "page_id": {"type": "keyword"},
                "page_title": {"type": "text", "analyzer": "onramp_ko"},
                "content": {"type": "text", "analyzer": "onramp_ko"},
                "embedding_text": {"type": "text", "analyzer": "onramp_ko"},
                "heading_path": {"type": "text", "analyzer": "onramp_ko"},
                "domain": {"type": "keyword"},
                "section_type": {"type": "keyword"},
                "chunking_profile": {"type": "keyword"},
                "tags": {"type": "keyword"},
                "keywords": {"type": "keyword"},
                "code_languages": {"type": "keyword"},
                "has_code": {"type": "boolean"},
                "has_table": {"type": "boolean"},
                "source_url": {"type": "keyword"},
                "space_key": {"type": "keyword"},
                "last_modified": {"type": "date", "ignore_malformed": True},
                "hash": {"type": "keyword"},
                "site": {"type": "keyword"},
                "product_version": {"type": "keyword"},
                "doc_key": {"type": "keyword"},
                "is_eol": {"type": "boolean"},
            },
        },
    }


def _search_body(
    query: str,
    *,
    top_k: int,
    tenant_id: str,
    domain: str | None,
    version: str | None,
    pinned_doc_keys: tuple[str, ...],
    excluded_doc_keys: tuple[str, ...],
) -> dict[str, Any]:
    filters: list[dict[str, Any]] = [{"term": {"tenant_id": tenant_id}}]
    if domain:
        filters.append({"term": {"domain": domain}})
    if version:
        filters.append({"term": {"product_version": version}})
    if pinned_doc_keys:
        filters.append({"terms": {"doc_key": list(pinned_doc_keys)}})

    must_not: list[dict[str, Any]] = []
    if excluded_doc_keys:
        must_not.append({"terms": {"doc_key": list(excluded_doc_keys)}})

    return {
        "size": top_k,
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": [
                                "content^3",
                                "embedding_text^2",
                                "page_title^2",
                                "heading_path",
                                "keywords^2",
                                "tags",
                                "code_languages",
                            ],
                            "type": "best_fields",
                        }
                    }
                ],
                "filter": filters,
                "must_not": must_not,
            }
        },
    }


def get_opensearch() -> OpenSearchClient:
    global _client
    if _client is None:
        _client = OpenSearchClient()
    return _client


async def close_opensearch() -> None:
    global _client
    if _client is not None:
        await _client.close()
        _client = None
