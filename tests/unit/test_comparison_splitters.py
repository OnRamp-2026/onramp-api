"""비교군 baseline splitter 단위 테스트 (#212) — 외부 I/O 없음(tiktoken 로컬)."""

import pytest

from app.rag.chunker import MarkdownPage
from app.rag.comparison_splitters import ComparisonSplitter, ComparisonStrategy

_MARKDOWN = """# 설치 가이드

EKS 클러스터를 설치하는 절차를 설명합니다. 먼저 kubectl을 준비합니다.

## 사전 준비

helm과 awscli가 필요합니다. 버전을 확인하세요.

## 배포

helm install 명령으로 배포합니다. 롤백은 helm rollback을 씁니다.
"""


def _page() -> MarkdownPage:
    return MarkdownPage(
        page_id="107194",
        page_title="설치 가이드",
        markdown=_MARKDOWN,
        source_url="https://x/107194",
        site="confluence",
        product_version="v1.2",
        doc_key="install-guide",
    )


@pytest.mark.parametrize("strategy", list(ComparisonStrategy))
def test_produces_indexable_childchunks(strategy) -> None:
    chunks = ComparisonSplitter(strategy, chunk_tokens=64, chunk_overlap=8).chunk(_page())
    assert chunks, "최소 1개 청크는 나와야 한다"
    for i, c in enumerate(chunks):
        assert c.chunk_id == f"107194_{i:03d}"  # page-level 지표 호환 포맷
        assert c.parent_id == ""  # flat baseline
        assert c.page_id == "107194"
        assert c.content.strip()
        assert c.embedding_text == c.content  # metadata prefix 없는 plain 텍스트
        assert c.token_count > 0
        assert c.chunking_profile == f"baseline:{strategy.value}"
        # page 메타 전파 (재색인·검색 메타 일관)
        assert c.source_url == "https://x/107194"
        assert c.doc_key == "install-guide"


def test_chunk_id_format_enables_page_collapse() -> None:
    # chunk_to_page가 baseline chunk_id에서 page_id를 복원할 수 있어야 page-level 비교가 동작한다.
    from app.eval.metrics import collapse_to_pages

    chunks = ComparisonSplitter(ComparisonStrategy.TOKEN, chunk_tokens=32, chunk_overlap=4).chunk(_page())
    pages = collapse_to_pages([c.chunk_id for c in chunks])
    assert pages == ["107194"]  # 한 페이지 → distinct page 1개


def test_markdown_strategy_preserves_heading_path() -> None:
    chunks = ComparisonSplitter(ComparisonStrategy.MARKDOWN, chunk_tokens=64, chunk_overlap=8).chunk(_page())
    # 헤더 기반 분할이므로 최소 한 청크는 heading_path를 가진다.
    assert any(c.heading_path for c in chunks)
    # 헤딩 경로는 분할 기준 헤더(h1/h2)에서 온다.
    paths = {tuple(c.heading_path) for c in chunks if c.heading_path}
    assert ("설치 가이드",) in paths or any("설치 가이드" in p for p in paths)


def test_token_strategy_respects_chunk_size() -> None:
    # 작은 chunk_tokens면 더 많은 청크로 쪼개진다(분할이 실제로 token 크기를 따른다).
    few = ComparisonSplitter(ComparisonStrategy.TOKEN, chunk_tokens=256, chunk_overlap=0).chunk(_page())
    many = ComparisonSplitter(ComparisonStrategy.TOKEN, chunk_tokens=16, chunk_overlap=0).chunk(_page())
    assert len(many) > len(few)


def test_invalid_overlap_raises() -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        ComparisonSplitter(ComparisonStrategy.TOKEN, chunk_tokens=100, chunk_overlap=100)


def test_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError):
        ComparisonSplitter("nonsense")
