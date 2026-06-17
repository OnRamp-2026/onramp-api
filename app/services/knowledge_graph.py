"""지식맵 그래프 빌더 — 색인 메타데이터에서 entity-relation 그래프를 구성한다.

RAG 파이프라인 무접촉(읽기전용 소비). 이 모듈은 **순수 변환**(I/O 없음) — 입력 문서는 호출측이
원장(source_document, source=confluence|github 우선) 또는 Qdrant payload(site 폴백)에서 모아 넘긴다.
멀티소스-ready: `source`는 입력 문서가 들고오므로 github 색인 후에도 재작업 없이 그대로 확장된다.

노드: source · domain · document · concept(keyword) · (version 계보는 엣지로 표현)
엣지: FROM_SOURCE · BELONGS_TO · MENTIONS · VERSION_OF
  → cross-domain 연결은 doc—MENTIONS→concept←MENTIONS—doc (개념 허브)로 자연 표현된다.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace

_OTHER = "기타"


@dataclass(frozen=True)
class GraphDocument:
    """그래프 1개 문서(=원장 1행 또는 Qdrant page 묶음). keywords는 그 문서 청크들의 합집합."""

    page_id: str
    title: str
    source: str  # confluence | github | (site 폴백: apache/datadog/...)
    domain: str
    version: str = ""
    doc_key: str = ""  # 버전 형제 묶음 키 (빈 값 = 계보 없음)
    last_modified: str = ""
    keywords: tuple[str, ...] = ()


@dataclass
class GraphNode:
    id: str
    label: str
    type: str  # source | domain | document | concept
    count: int = 0
    meta: dict[str, str] = field(default_factory=dict)


@dataclass
class GraphEdge:
    source: str
    target: str
    rel: str  # FROM_SOURCE | BELONGS_TO | MENTIONS | VERSION_OF


@dataclass
class KnowledgeGraph:
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    coverage: dict[str, int]  # domain -> 문서 수
    sources: dict[str, int]  # source -> 문서 수


def _norm(value: str) -> str:
    return (value or "").strip()


def _keywords_of(doc: GraphDocument) -> set[str]:
    return {k for k in (_norm(kw) for kw in doc.keywords) if k}


def build_graph(documents: list[GraphDocument], *, min_concept_docs: int = 2) -> KnowledgeGraph:
    """문서 목록 → entity-relation 그래프.

    min_concept_docs: 이 수 이상의 서로 다른 문서에 등장한 keyword만 Concept 노드로 승격(단발 노이즈 억제).
    page_id 중복은 첫 항목만 유지.
    """
    if min_concept_docs < 1:
        raise ValueError("min_concept_docs must be >= 1")

    # page_id 중복 제거(첫 항목 유지) + 빈 page_id 제외. 정규화된 page_id로 통일해 이후 전 구간 일관성 보장.
    seen: set[str] = set()
    docs: list[GraphDocument] = []
    for doc in documents:
        pid = _norm(doc.page_id)
        if not pid or pid in seen:
            continue
        seen.add(pid)
        docs.append(replace(doc, page_id=pid))

    source_count: Counter[str] = Counter()
    domain_count: Counter[str] = Counter()
    concept_docs: dict[str, set[str]] = defaultdict(set)  # keyword -> {page_id}
    lineage: dict[str, list[str]] = defaultdict(list)  # doc_key -> [page_id]

    for doc in docs:
        source_count[_norm(doc.source) or _OTHER] += 1
        domain_count[_norm(doc.domain) or _OTHER] += 1
        for kw in _keywords_of(doc):
            concept_docs[kw].add(doc.page_id)
        if _norm(doc.doc_key):
            lineage[_norm(doc.doc_key)].append(doc.page_id)

    concepts = {kw for kw, pages in concept_docs.items() if len(pages) >= min_concept_docs}

    nodes: list[GraphNode] = []
    nodes += [GraphNode(f"src::{s}", s, "source", count=n) for s, n in source_count.items()]
    nodes += [GraphNode(f"dom::{d}", d, "domain", count=n) for d, n in domain_count.items()]
    nodes += [GraphNode(f"concept::{k}", k, "concept", count=len(concept_docs[k])) for k in concepts]
    nodes += [
        GraphNode(
            f"doc::{doc.page_id}",
            _norm(doc.title) or doc.page_id,
            "document",
            meta={
                "source": _norm(doc.source) or _OTHER,
                "domain": _norm(doc.domain) or _OTHER,
                "version": _norm(doc.version),
                "last_modified": _norm(doc.last_modified),
            },
        )
        for doc in docs
    ]

    edges: list[GraphEdge] = []
    for doc in docs:
        did = f"doc::{doc.page_id}"
        edges.append(GraphEdge(did, f"src::{_norm(doc.source) or _OTHER}", "FROM_SOURCE"))
        edges.append(GraphEdge(did, f"dom::{_norm(doc.domain) or _OTHER}", "BELONGS_TO"))
        edges += [GraphEdge(did, f"concept::{kw}", "MENTIONS") for kw in _keywords_of(doc) if kw in concepts]

    # 버전 계보 — 같은 doc_key를 공유하는 문서 쌍 (확장 방향: SUPERSEDES 방향성은 version 비교로)
    for pages in lineage.values():
        uniq = list(dict.fromkeys(pages))
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                edges.append(GraphEdge(f"doc::{uniq[i]}", f"doc::{uniq[j]}", "VERSION_OF"))

    return KnowledgeGraph(
        nodes=nodes,
        edges=edges,
        coverage=dict(domain_count),
        sources=dict(source_count),
    )
