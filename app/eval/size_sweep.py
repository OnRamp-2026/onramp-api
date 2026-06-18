"""onramp parent/child 토큰 사이즈 sweep 헬퍼 (#212) — 순수 함수(인프라 I/O 없음).

기존 청킹 A/B 하네스(`scripts/eval_chunking_ab.py`, `app/eval/chunking_experiment.py`, #238/#247)는
onramp 전략을 `SemanticChunker()` **기본 사이즈**로만 색인한다(splitter 종류 비교용). 이 모듈은
child/parent 토큰 target을 파라미터로 받는 `SemanticChunker`를 만들어 **사이즈 ablation**을 가능케 한다.

설계: target만 바꾸고 child_min/max·parent_max·overlap은 현재 기본 비율로 파생한다.
overlap은 child의 15% 고정 — **토큰 수만 최적화**하고 overlap은 따로 최적화하지 않는다(#212 결정).
재색인/검색은 `scripts/eval_size_sweep.py`가 한다. (chunking_experiment 의존을 피해 torch 로드를 회피.)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from app.rag.chunker import ChildChunk, MarkdownPage, SemanticChunker

DEFAULT_CHILD_TARGET = 400
DEFAULT_PARENT_TARGET = 1200
# 파생 비율 — SemanticChunker 기본값(child_min 50/400, child_max 650/400, parent_max 1400/1200)에서
# 그대로 따오되, overlap만 0.30→0.15(10~20% 중)로 낮춰 고정한다.
_CHILD_MIN_RATIO = 0.125
_CHILD_MAX_RATIO = 1.625
_PARENT_MAX_RATIO = 7 / 6
_OVERLAP_RATIO = 0.15


def build_onramp_chunker(child_target: int, parent_target: int) -> SemanticChunker:
    """child/parent target에서 나머지 파라미터를 기본 비율로 파생한 `SemanticChunker`."""
    if child_target < 1 or parent_target < 1:
        raise ValueError("child_target/parent_target 은 1 이상이어야 합니다")
    return SemanticChunker(
        child_min_tokens=max(1, round(child_target * _CHILD_MIN_RATIO)),
        child_target_tokens=child_target,
        child_max_tokens=round(child_target * _CHILD_MAX_RATIO),
        parent_target_tokens=parent_target,
        parent_max_tokens=round(parent_target * _PARENT_MAX_RATIO),
        overlap_tokens=round(child_target * _OVERLAP_RATIO),
    )


def chunk_page_sized(page: MarkdownPage, child_target: int, parent_target: int) -> list[ChildChunk]:
    """사이즈 지정 onramp 청킹 — children만 반환(parent 구조는 검색 게이트에 불필요)."""
    _, children = build_onramp_chunker(child_target, parent_target).chunk(page)
    return children


@dataclass(frozen=True)
class SizeConfig:
    """한 사이즈 구성 — 임시 인덱스 이름의 근거(사이즈 변형 간 충돌 방지·재현 추적)."""

    child_target: int = DEFAULT_CHILD_TARGET
    parent_target: int = DEFAULT_PARENT_TARGET

    @property
    def hash(self) -> str:
        raw = f"onramp-size:{self.child_target}:{self.parent_target}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]

    def collection_name(self, prefix: str = "onramp-eval-size") -> str:
        """production 인덱스와 절대 겹치지 않는 임시 이름."""
        return f"{prefix}-c{self.child_target}-p{self.parent_target}-{self.hash}"


def page_from_row(
    *,
    page_id: str,
    title: str | None,
    markdown: str | None,
    source_url: str | None = "",
    space_key: str | None = "OnRamp",
    last_modified: str = "",
) -> MarkdownPage:
    """`SourceDocument` 필드 → `MarkdownPage`. 버전 계보 메타는 게이트(검색)에 불필요해 생략."""
    return MarkdownPage(
        page_id=str(page_id),
        page_title=title or "",
        markdown=markdown or "",
        source_url=source_url or "",
        space_key=space_key or "OnRamp",
        last_modified=last_modified or "",
    )
