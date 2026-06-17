"""OpenSearch BM25 index adapter for hybrid retrieval."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

_client: OpenSearchClient | None = None


def _doc_url(index: str, doc_id: str) -> str:
    """`_doc/{id}` 경로 생성. _id는 '/'·'#'·':' 등을 포함할 수 있어(예: gh:repo:docs/x.md) URL path에선
    반드시 percent-encode한다. (인코딩 누락 시 '/'→경로분리로 400, '#'→fragment로 잘림.)"""
    return f"/{index}/_doc/{quote(doc_id, safe='')}"


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
        self._chunk_fields_ensured = False

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

    # 기존 인덱스(옛 매핑)에 뒤늦게 추가된 청크 필드 — additive PUT _mapping으로 자가치유.
    # 새로 만든 인덱스는 _index_body에 이미 포함되므로 no-op. (재색인 시 domain_source 필터 가능하게)
    _ADDED_CHUNK_FIELDS = {
        "domain_source": {"type": "keyword"},
        "domain_confidence": {"type": "float"},
    }

    async def _ensure_chunk_fields(self, alias: str) -> None:
        if self._chunk_fields_ensured:
            return
        resp = await self._http.put(f"/{alias}/_mapping", json={"properties": self._ADDED_CHUNK_FIELDS})
        resp.raise_for_status()
        self._chunk_fields_ensured = True

    async def ensure_index(self) -> None:
        alias = self.settings.opensearch_index
        concrete = self.settings.opensearch_index_v1
        alias_resp = await self._http.get(f"/_alias/{alias}")
        if alias_resp.status_code == 200:
            await self._ensure_chunk_fields(alias)
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
            resp = await self._http.put(_doc_url(self.settings.opensearch_index, chunk_id), json=dict(document))
            resp.raise_for_status()

    async def delete_chunks(self, chunk_ids: Sequence[str]) -> None:
        if not chunk_ids:
            return
        for chunk_id in chunk_ids:
            resp = await self._http.delete(_doc_url(self.settings.opensearch_index, chunk_id))
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

    # ── 문서(원문) 인덱스 — document_tools (get_document_by_id, search_documents_by_text) ──
    async def ensure_documents_index(self) -> None:
        alias = self.settings.opensearch_documents_index
        concrete = self.settings.opensearch_documents_index_v1
        alias_resp = await self._http.get(f"/_alias/{alias}")
        if alias_resp.status_code == 200:
            return
        if alias_resp.status_code != 404:
            alias_resp.raise_for_status()
        index_resp = await self._http.head(f"/{concrete}")
        if index_resp.status_code == 404:
            create_resp = await self._http.put(f"/{concrete}", json=_documents_index_body(alias))
            create_resp.raise_for_status()
            return
        if index_resp.status_code >= 400:
            index_resp.raise_for_status()
        alias_create = await self._http.post(
            "/_aliases", json={"actions": [{"add": {"index": concrete, "alias": alias}}]}
        )
        alias_create.raise_for_status()

    async def upsert_documents(self, documents: Sequence[Mapping[str, Any]]) -> None:
        """원문 문서 upsert. _id = '{tenant_id}:{doc_id}' (테넌트 격리). 각 doc는 tenant_id·doc_id·content 포함."""
        if not documents:
            return
        await self.ensure_documents_index()
        index = self.settings.opensearch_documents_index
        for document in documents:
            doc_id = f"{document['tenant_id']}:{document['doc_id']}"
            resp = await self._http.put(_doc_url(index, doc_id), json=dict(document))
            resp.raise_for_status()

    async def get_document(self, doc_id: str, *, tenant_id: str) -> dict[str, Any] | None:
        resp = await self._http.get(_doc_url(self.settings.opensearch_documents_index, f"{tenant_id}:{doc_id}"))
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        source = resp.json().get("_source")
        return source if isinstance(source, dict) else None

    async def get_documents(self, doc_ids: Sequence[str], *, tenant_id: str) -> list[dict[str, Any]]:
        if not doc_ids:
            return []
        ids = [f"{tenant_id}:{doc_id}" for doc_id in doc_ids]
        resp = await self._http.post(f"/{self.settings.opensearch_documents_index}/_mget", json={"ids": ids})
        resp.raise_for_status()
        return [doc["_source"] for doc in resp.json().get("docs", []) if doc.get("found")]

    async def search_documents(
        self, query: str, *, top_k: int, tenant_id: str, domain: str | None = None, source: str | None = None
    ) -> list[OpenSearchHit]:
        """원문 전체 BM25 검색(search_documents_by_text). 청크 검색과 별개 — 문서 단위."""
        body = _documents_search_body(query, top_k=top_k, tenant_id=tenant_id, domain=domain, source=source)
        resp = await self._http.post(f"/{self.settings.opensearch_documents_index}/_search", json=body)
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
                "domain_source": {"type": "keyword"},
                "domain_confidence": {"type": "float"},
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


def _documents_index_body(alias: str) -> dict[str, Any]:
    """원문 문서 인덱스 매핑. content=원문(cleaned_markdown) BM25, _source로 원문 반환."""
    return {
        "aliases": {alias: {}},
        "settings": {
            "analysis": {
                "analyzer": {"onramp_ko": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase"]}}
            }
        },
        "mappings": {
            "dynamic": "false",
            "properties": {
                "tenant_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},  # = source_document.page_id (gh:repo:path 등)
                "source": {"type": "keyword"},  # confluence | github
                "title": {"type": "text", "analyzer": "onramp_ko"},
                "content": {"type": "text", "analyzer": "onramp_ko"},  # 원문 전체(BM25 + _source 반환)
                "domain": {"type": "keyword"},
                "space_key": {"type": "keyword"},
                "source_url": {"type": "keyword"},
                "last_modified": {"type": "date", "ignore_malformed": True},
            },
        },
    }


def _documents_search_body(
    query: str, *, top_k: int, tenant_id: str, domain: str | None, source: str | None
) -> dict[str, Any]:
    filters: list[dict[str, Any]] = [{"term": {"tenant_id": tenant_id}}]
    if domain:
        filters.append({"term": {"domain": domain}})
    if source:
        filters.append({"term": {"source": source}})
    return {
        "size": top_k,
        "query": {
            "bool": {
                "must": [{"multi_match": {"query": query, "fields": ["title^3", "content"], "type": "best_fields"}}],
                "filter": filters,
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
