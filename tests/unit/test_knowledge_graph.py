import pytest

from app.services.knowledge_graph import GraphDocument, build_graph


def _doc(pid, source, domain, keywords=(), version="", doc_key="", title=""):
    return GraphDocument(
        page_id=pid,
        title=title or pid,
        source=source,
        domain=domain,
        version=version,
        doc_key=doc_key,
        keywords=tuple(keywords),
    )


def _ids(nodes, type_):
    return {n.id for n in nodes if n.type == type_}


def _edges(graph, rel):
    return {(e.source, e.target) for e in graph.edges if e.rel == rel}


def test_multisource_source_nodes_and_edges():
    # github 색인 후에도 재작업 0 — source는 입력 문서가 들고온다
    g = build_graph(
        [
            _doc("c1", "confluence", "manual"),
            _doc("g1", "github", "incident"),
        ]
    )
    assert _ids(g.nodes, "source") == {"src::confluence", "src::github"}
    assert ("doc::g1", "src::github") in _edges(g, "FROM_SOURCE")
    assert g.sources == {"confluence": 1, "github": 1}


def test_concept_promoted_only_when_shared_and_links_cross_domain():
    # 같은 keyword가 incident·manual 두 도메인 문서에 등장 → 개념 허브로 cross-domain 연결
    g = build_graph(
        [
            _doc("a", "confluence", "incident", keywords=["CrashLoopBackOff", "OnlyHere"]),
            _doc("b", "confluence", "manual", keywords=["CrashLoopBackOff"]),
        ],
        min_concept_docs=2,
    )
    assert "concept::CrashLoopBackOff" in _ids(g.nodes, "concept")
    assert "concept::OnlyHere" not in _ids(g.nodes, "concept")  # 단발 keyword 필터
    mentions = _edges(g, "MENTIONS")
    assert ("doc::a", "concept::CrashLoopBackOff") in mentions
    assert ("doc::b", "concept::CrashLoopBackOff") in mentions  # a(incident)↔b(manual) 개념 허브로 연결


def test_version_lineage_edges_from_doc_key():
    g = build_graph(
        [
            _doc("v22", "confluence", "api_reference", version="2.2", doc_key="apache_modrewrite"),
            _doc("v24", "confluence", "api_reference", version="2.4", doc_key="apache_modrewrite"),
            _doc("solo", "confluence", "manual", doc_key=""),
        ]
    )
    assert _edges(g, "VERSION_OF") == {("doc::v22", "doc::v24")}


def test_coverage_and_empty_source_domain_fallback():
    g = build_graph(
        [
            _doc("a", "", "incident"),  # 빈 source → 기타
            _doc("b", "confluence", ""),  # 빈 domain → 기타
            _doc("c", "confluence", "incident"),
        ]
    )
    assert g.coverage == {"incident": 2, "기타": 1}
    assert g.sources == {"기타": 1, "confluence": 2}


def test_min_concept_docs_must_be_positive():
    with pytest.raises(ValueError):
        build_graph([_doc("a", "confluence", "manual")], min_concept_docs=0)


def test_page_id_normalized_consistently_across_nodes_and_edges():
    # 공백 섞인 page_id — dedup·노드ID·concept/lineage가 모두 정규화 값으로 일관
    g = build_graph(
        [
            _doc(" p1 ", "confluence", "incident", keywords=["K"], doc_key="dk"),
            _doc("p1", "confluence", "incident", keywords=["K"], doc_key="dk"),  # 정규화 시 동일 → dedup
        ],
        min_concept_docs=1,
    )
    docnodes = [n for n in g.nodes if n.type == "document"]
    assert len(docnodes) == 1
    assert docnodes[0].id == "doc::p1"  # 정규화된 id
    assert ("doc::p1", "concept::K") in {(e.source, e.target) for e in g.edges if e.rel == "MENTIONS"}


def test_dedup_page_id_and_doc_belongs_to_domain():
    g = build_graph(
        [
            _doc("dup", "confluence", "manual"),
            _doc("dup", "confluence", "manual"),  # 중복 page_id → 1개만
        ]
    )
    assert len([n for n in g.nodes if n.type == "document"]) == 1
    assert ("doc::dup", "dom::manual") in _edges(g, "BELONGS_TO")
