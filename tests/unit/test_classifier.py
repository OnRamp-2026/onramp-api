from app.rag.chunker import ChildChunk
from app.rag.classifier import ChunkMetadataClassifier, DocumentProfileClassifier


def _child(
    content: str,
    domain: str = "운영매뉴얼",
    section_type: str = "general",
    page_title: str = "Kubernetes 장애 대응",
    heading_path: list[str] | None = None,
) -> ChildChunk:
    return ChildChunk(
        chunk_id="pg_000",
        parent_id="pg_p000",
        page_id="pg",
        page_title=page_title,
        content=content,
        embedding_text=content,
        heading_path=heading_path or ["장애 대응", "원인 분석"],
        chunk_index=0,
        token_count=100,
        overlap_from_previous=0,
        source_url="https://example.atlassian.net/wiki/spaces/OnRamp/pages/pg",
        space_key="OnRamp",
        last_modified="2026-06-01T00:00:00.000+0900",
        hash="hash",
        domain=domain,
        section_type=section_type,
        block_types=["paragraph", "code"],
        keywords=["kubectl logs api-0"],
        has_code=True,
        code_languages=["bash"],
    )


def test_document_profile_classifier_detects_control_like_documents() -> None:
    markdown = """
    # 주간 회의록

    ## 결정사항

    - 담당자: 플랫폼팀
    - 액션아이템: Qdrant collection 정책을 확정한다.
    - 기한: 2026-06-03
    """

    profile = DocumentProfileClassifier().classify_page("주간 회의록", markdown)

    assert profile == "control_like"


def test_document_profile_classifier_defaults_to_runbook_like_for_operational_docs() -> None:
    markdown = """
    # Kubernetes 장애 대응 Runbook

    ## 검증

    ```bash
    kubectl get pods
    ```
    """

    profile = DocumentProfileClassifier().classify_page("Kubernetes 장애 대응 Runbook", markdown)

    assert profile == "runbook_like"


def test_chunk_metadata_classifier_refines_domain_tags_keywords_and_embedding_text() -> None:
    chunk = _child("CrashLoopBackOff 원인 분석을 위해 kubectl logs api-0 명령을 실행한다.")

    classified = ChunkMetadataClassifier().classify_chunk(chunk, "runbook_like")

    assert classified.chunking_profile == "runbook_like"
    assert classified.domain == "incident"
    assert classified.section_type == "root_cause"
    assert "incident" in (classified.tags or [])
    assert "kubernetes" in (classified.tags or [])
    assert "bash" in (classified.tags or [])
    assert "kubectl logs api-0" in (classified.keywords or [])
    assert "도메인: incident" in classified.embedding_text
    assert "청킹 프로필: runbook_like" in classified.embedding_text
    assert "태그:" in classified.embedding_text
    assert "CrashLoopBackOff" in classified.embedding_text


def test_chunk_metadata_classifier_maps_existing_korean_domain_when_no_stronger_signal() -> None:
    chunk = _child(
        "운영 절차를 순서대로 확인한다.",
        domain="운영매뉴얼",
        section_type="procedure",
        page_title="운영 매뉴얼",
        heading_path=["운영", "절차"],
    )

    classified = ChunkMetadataClassifier().classify_chunk(chunk, "runbook_like")

    assert classified.domain == "manual"
    assert classified.section_type == "procedure"
    assert "manual" in (classified.tags or [])


def _batch_child(content: str, chunk_id: str, parent_id: str = "p1") -> ChildChunk:
    """도메인이 content로만 추론되도록 page_title·heading·domain을 중립으로 둔 child."""
    return ChildChunk(
        chunk_id=chunk_id,
        parent_id=parent_id,
        page_id="pg",
        page_title="문서",
        content=content,
        embedding_text=content,
        heading_path=["섹션"],
        chunk_index=0,
        token_count=10,
        overlap_from_previous=0,
        source_url="u",
        space_key="OnRamp",
        last_modified="",
        hash="h",
        domain="",
        section_type="general",
        block_types=["paragraph"],
        keywords=[],
        code_languages=[],
    )


def test_classify_batch_unifies_parent_domain_by_majority() -> None:
    children = [
        _batch_child("kubectl 설치 절차 운영 매뉴얼", "c0"),  # manual
        _batch_child("운영 절차 troubleshooting 디버그", "c1"),  # manual
        _batch_child("api endpoint request response 요청 응답", "c2"),  # api_reference
    ]

    out = ChunkMetadataClassifier().classify_batch(children, "runbook_like")

    assert {c.domain for c in out} == {"manual"}  # 다수결(manual 2 > api_reference 1)로 통일
    c2 = next(c for c in out if c.chunk_id == "c2")
    assert "domain:api_reference" in (c2.tags or [])  # child 자기 추론은 보조 태그로 보존
    assert "도메인: manual" in c2.embedding_text  # embedding_text도 통일 도메인으로 재생성


def test_classify_batch_separate_parents_independent() -> None:
    children = [
        _batch_child("api endpoint response", "a0", parent_id="pa"),  # api_reference
        _batch_child("kubectl 운영 절차", "b0", parent_id="pb"),  # manual
    ]

    out = ChunkMetadataClassifier().classify_batch(children, "runbook_like")
    by_id = {c.chunk_id: c for c in out}
    assert by_id["a0"].domain == "api_reference"  # parent별 독립
    assert by_id["b0"].domain == "manual"


def test_classify_chunk_inherits_explicit_primary_domain() -> None:
    child = _batch_child("api endpoint response", "c0")  # 단독 추론은 api_reference

    out = ChunkMetadataClassifier().classify_chunk(child, "runbook_like", primary_domain="manual")

    assert out.domain == "manual"  # 명시 primary_domain 상속
    assert "domain:api_reference" in (out.tags or [])
