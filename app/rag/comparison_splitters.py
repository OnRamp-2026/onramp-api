"""비교군 청킹 baseline (#212 Phase 1) — Token/MarkdownHeader/Recursive.

OnRamp `SemanticChunker`(구조 인식 parent-child)와 정량 비교하기 위한 표준 baseline 3종.
langchain 표준 구현으로 `page.markdown`을 **flat child 청크**로 분할한다.

baseline의 핵심(= OnRamp 대비 무엇이 빠졌나):
- **parent 구조 없음** (`parent_id=""`) — child-only.
- **metadata prefix 없는 plain `embedding_text`** — OnRamp의 구조 인식 enrichment 미적용.
- chunk_id는 `{page_id}_{idx:03d}` 포맷 유지 → page-level 지표(#212 §2-2)가 동일하게 동작.

`token_count`는 `SemanticChunker._count_tokens`와 동일한 근사 카운터를 써서 splitter 간
token/cost 비교가 apples-to-apples가 되게 한다(분할 크기 기준은 tiktoken).
"""

from __future__ import annotations

import hashlib
import re
from enum import StrEnum
from typing import TYPE_CHECKING

from app.rag.chunker import ChildChunk, MarkdownPage

# NOTE: langchain_text_splitters(→ transformers → torch)는 무거워서 **메서드 내부에서 지연 import**한다.
# 이 모듈을 import만 하는 경로(예: onramp 전략의 ChunkingConfig 검증)가 torch 로드 비용을 물지 않게 하기 위함.
if TYPE_CHECKING:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

# 분할 크기 기준 토크나이저(tiktoken) — OpenAI 임베딩과 동일 계열.
_ENCODING = "cl100k_base"
# token_count 보고용 — SemanticChunker._count_tokens와 동일한 근사(비교 일관성).
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
# MarkdownHeader 분할 기준 헤더 레벨.
_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def _count_tokens(text: str) -> int:
    """SemanticChunker._count_tokens와 동일한 근사 토큰 카운터(외부 토크나이저 무관)."""
    return len(_TOKEN_RE.findall(text))


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class ComparisonStrategy(StrEnum):
    """비교군 baseline 청킹 전략."""

    TOKEN = "token"  # 고정 token 단위 분할 — 가장 단순한 comparison baseline
    MARKDOWN = "markdown"  # heading 기반 구조 분할 — 문서 구조 보존 baseline
    RECURSIVE = "recursive"  # 일반 recursive splitter — 실무 표준 baseline


class ComparisonSplitter:
    """비교군 baseline splitter — `page.markdown`을 flat `ChildChunk` 리스트로 분할.

    `SemanticChunker.chunk`가 `(parents, children)`를 돌려주는 것과 달리 children만 만든다
    (baseline은 parent 구조가 없다). 인덱서/리포지토리는 동일한 `ChildChunk`를 받는다.
    """

    def __init__(
        self,
        strategy: ComparisonStrategy | str,
        *,
        chunk_tokens: int = 400,
        chunk_overlap: int = 50,
    ) -> None:
        self.strategy = ComparisonStrategy(strategy)
        if chunk_overlap >= chunk_tokens:
            raise ValueError(f"chunk_overlap({chunk_overlap}) < chunk_tokens({chunk_tokens}) 여야 합니다")
        self.chunk_tokens = chunk_tokens
        self.chunk_overlap = chunk_overlap

    def chunk(self, page: MarkdownPage) -> list[ChildChunk]:
        pieces = self._split(page.markdown)
        return [self._to_child(page, content, heading_path, idx) for idx, (content, heading_path) in enumerate(pieces)]

    def _split(self, markdown: str) -> list[tuple[str, list[str]]]:
        """분할 → (content, heading_path) 쌍 리스트. Token/Recursive는 heading_path 빈 리스트."""
        from langchain_text_splitters import MarkdownHeaderTextSplitter, TokenTextSplitter

        if self.strategy is ComparisonStrategy.TOKEN:
            splitter = TokenTextSplitter(
                encoding_name=_ENCODING, chunk_size=self.chunk_tokens, chunk_overlap=self.chunk_overlap
            )
            return [(t, []) for t in splitter.split_text(markdown) if t.strip()]

        if self.strategy is ComparisonStrategy.RECURSIVE:
            return [(t, []) for t in self._recursive().split_text(markdown) if t.strip()]

        # MARKDOWN: 헤더로 섹션 분할 후, 큰 섹션은 token 단위로 재분할(heading_path 보존).
        header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_HEADERS, strip_headers=False)
        sub = self._recursive()
        out: list[tuple[str, list[str]]] = []
        for doc in header_splitter.split_text(markdown):
            heading_path = [doc.metadata[key] for _, key in _HEADERS if doc.metadata.get(key)]
            for piece in sub.split_text(doc.page_content):
                if piece.strip():
                    out.append((piece, heading_path))
        return out

    def _recursive(self) -> RecursiveCharacterTextSplitter:
        """token 길이 기준 recursive splitter (chunk_size = tiktoken 토큰)."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=_ENCODING, chunk_size=self.chunk_tokens, chunk_overlap=self.chunk_overlap
        )

    def _to_child(self, page: MarkdownPage, content: str, heading_path: list[str], idx: int) -> ChildChunk:
        return ChildChunk(
            chunk_id=f"{page.page_id}_{idx:03d}",  # page-level 지표 호환 포맷
            parent_id="",  # flat baseline — parent 구조 없음
            page_id=page.page_id,
            page_title=page.page_title,
            content=content,
            embedding_text=content,  # baseline: metadata prefix 없는 plain 텍스트
            heading_path=heading_path,
            chunk_index=idx,
            token_count=_count_tokens(content),
            overlap_from_previous=0,
            source_url=page.source_url,
            space_key=page.space_key,
            last_modified=page.last_modified,
            hash=_hash(content),
            chunking_profile=f"baseline:{self.strategy.value}",
            site=page.site,
            product_version=page.product_version,
            doc_key=page.doc_key,
            is_eol=page.is_eol,
        )
