"""청킹 A/B 실험 헬퍼 (#212 Phase 1, step 5) — 순수 함수(인프라 I/O 없음).

`SourceDocument` 행을 `MarkdownPage`로, 전략명을 `ChildChunk` 리스트로 매핑하고,
config-hash 임시 컬렉션 이름을 만든다. 실제 재색인/검색은 `scripts/eval_chunking_ab.py`가 한다.

전략:
- `onramp`    — `SemanticChunker`(구조 인식 parent-child). children만 색인.
- `token`/`markdown`/`recursive` — `ComparisonSplitter` baseline(flat child).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from app.rag.chunker import ChildChunk, MarkdownPage, SemanticChunker
from app.rag.comparison_splitters import ComparisonSplitter, ComparisonStrategy

ONRAMP = "onramp"
VALID_STRATEGIES: tuple[str, ...] = (ONRAMP, *sorted(s.value for s in ComparisonStrategy))


@dataclass(frozen=True)
class ChunkingConfig:
    """한 청킹 구성(전략 + 크기). 임시 컬렉션 이름의 config-hash 근거."""

    strategy: str
    chunk_tokens: int = 400
    chunk_overlap: int = 50

    def __post_init__(self) -> None:
        if self.strategy not in VALID_STRATEGIES:
            raise ValueError(f"알 수 없는 전략: {self.strategy!r} (가능: {VALID_STRATEGIES})")

    @property
    def hash(self) -> str:
        """전략·크기 결정 파라미터의 짧은 해시 — 임시 컬렉션 이름 충돌 방지·재현 추적."""
        raw = f"{self.strategy}:{self.chunk_tokens}:{self.chunk_overlap}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]

    def collection_name(self, prefix: str = "onramp-eval") -> str:
        """production 컬렉션과 절대 겹치지 않는 임시 이름 (#212 — index 분리 원칙)."""
        return f"{prefix}-{self.strategy}-{self.hash}"


def page_from_row(
    *,
    page_id: str,
    title: str | None,
    markdown: str | None,
    source_url: str | None = "",
    space_key: str | None = "OnRamp",
    last_modified: str = "",
) -> MarkdownPage:
    """`SourceDocument` 필드 → `MarkdownPage`. 버전 계보 메타는 게이트(dense)에 불필요하므로 생략."""
    return MarkdownPage(
        page_id=str(page_id),
        page_title=title or "",
        markdown=markdown or "",
        source_url=source_url or "",
        space_key=space_key or "OnRamp",
        last_modified=last_modified or "",
    )


def chunk_page(config: ChunkingConfig, page: MarkdownPage) -> list[ChildChunk]:
    """구성 전략으로 한 페이지를 child 청크로 분할한다."""
    if config.strategy == ONRAMP:
        _, children = SemanticChunker().chunk(page)
        return children
    splitter = ComparisonSplitter(config.strategy, chunk_tokens=config.chunk_tokens, chunk_overlap=config.chunk_overlap)
    return splitter.chunk(page)
