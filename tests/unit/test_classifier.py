from app.rag.chunker import ChildChunk
from app.rag.classifier import AutoClassifier


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


def test_classifier_refines_domain_tags_keywords_and_embedding_text() -> None:
    chunk = _child("CrashLoopBackOff 원인 분석을 위해 kubectl logs api-0 명령을 실행한다.")

    classified = AutoClassifier().classify_chunk(chunk)

    assert classified.domain == "incident"
    assert classified.section_type == "root_cause"
    assert "incident" in (classified.tags or [])
    assert "kubernetes" in (classified.tags or [])
    assert "bash" in (classified.tags or [])
    assert "kubectl logs api-0" in (classified.keywords or [])
    assert "도메인: incident" in classified.embedding_text
    assert "태그:" in classified.embedding_text
    assert "CrashLoopBackOff" in classified.embedding_text


def test_classifier_maps_existing_korean_domain_when_no_stronger_signal() -> None:
    chunk = _child(
        "운영 절차를 순서대로 확인한다.",
        domain="운영매뉴얼",
        section_type="procedure",
        page_title="운영 매뉴얼",
        heading_path=["운영", "절차"],
    )

    classified = AutoClassifier().classify_chunk(chunk)

    assert classified.domain == "manual"
    assert classified.section_type == "procedure"
    assert "manual" in (classified.tags or [])
