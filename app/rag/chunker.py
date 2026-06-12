"""Hierarchical parent-child chunking for cleaned Markdown."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "장애대응": (
        "장애",
        "incident",
        "outage",
        "postmortem",
        "타임라인",
        "영향",
        "원인",
        "조치",
        "재발",
        "복구",
    ),
    "운영매뉴얼": (
        "runbook",
        "매뉴얼",
        "절차",
        "운영",
        "troubleshooting",
        "debug",
        "디버그",
        "설치",
        "검증",
        "롤백",
    ),
    "API명세": ("api", "endpoint", "request", "response", "http", "요청", "응답", "에러 코드", "status code"),
    "회의록": ("회의", "minutes", "참석", "결정사항", "action item", "액션아이템", "논의"),
    "기획서": ("기획", "요구사항", "prd", "정책", "목표", "범위", "사용자 시나리오"),
}

SECTION_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "timeline": ("타임라인", "timeline", "chronology"),
    "impact": ("영향", "impact", "affected", "범위"),
    "root_cause": ("원인", "root cause", "cause"),
    "prevention": ("재발", "예방", "follow-up", "action item"),
    "mitigation": ("조치", "대응", "mitigation", "resolution", "복구"),
    "procedure": ("절차", "방법", "steps", "how to", "사용법"),
    "verification": ("검증", "확인", "check", "verify", "test"),
    "rollback": ("롤백", "rollback", "revert"),
    "api_contract": ("api", "endpoint", "request", "response", "요청", "응답"),
    "decision": ("결정", "decision", "결정사항"),
}

CONTROL_SECTION_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "decision": ("결정", "decision", "결정사항", "의사결정"),
    "action_item": ("action item", "액션아이템", "액션 아이템", "todo", "해야 할 일"),
    "owner_due_date": ("담당자", "owner", "assignee", "기한", "due date", "마감", "상태"),
    "risk": ("리스크", "risk", "블로커", "blocker", "이슈", "issue"),
    "requirement_change": ("요구사항 변경", "변경사항", "scope change", "정책 변경", "policy change"),
    "policy": ("정책", "policy", "운영 기준", "가이드라인"),
    "handoff": ("인수인계", "handoff", "전달사항"),
    "agenda": ("안건", "agenda", "회의 주제", "논의"),
}


@dataclass(frozen=True)
class MarkdownPage:
    """Input page for chunking."""

    page_id: str
    page_title: str
    markdown: str
    source_url: str = ""
    space_key: str = "OnRamp"
    last_modified: str = ""
    # 버전 계보 메타 (#94 — Confluence 라벨 파생, app/rag/labels.py)
    site: str = ""
    product_version: str = ""
    doc_key: str = ""
    is_eol: bool = False


@dataclass(frozen=True)
class MarkdownBlock:
    """A parsed Markdown block with structural type and heading context."""

    kind: str
    content: str
    heading_path: list[str]
    heading_level: int | None = None
    language: str = ""


@dataclass(frozen=True)
class ParentChunk:
    """Parent context chunk used for LLM prompt reconstruction."""

    parent_id: str
    page_id: str
    page_title: str
    heading_path: list[str]
    content: str
    token_count: int
    chunk_index: int
    source_url: str
    space_key: str
    last_modified: str
    hash: str
    domain: str = ""
    section_type: str = ""
    block_types: list[str] | None = None


@dataclass(frozen=True)
class ChildChunk:
    """Search chunk intended for vector indexing."""

    chunk_id: str
    parent_id: str
    page_id: str
    page_title: str
    content: str
    embedding_text: str
    heading_path: list[str]
    chunk_index: int
    token_count: int
    overlap_from_previous: int
    source_url: str
    space_key: str
    last_modified: str
    hash: str
    chunking_profile: str = ""
    domain: str = ""
    section_type: str = ""
    block_types: list[str] | None = None
    keywords: list[str] | None = None
    tags: list[str] | None = None
    has_code: bool = False
    has_table: bool = False
    has_list: bool = False
    code_languages: list[str] | None = None
    # 버전 계보 메타 (#94) — Qdrant payload로 그대로 흘러간다 (indexer._payload는 asdict)
    site: str = ""
    product_version: str = ""
    doc_key: str = ""  # 버전 형제 묶음 키 (빈 값 = 계보 없음)
    is_eol: bool = False
    content_vector: list[float] | None = None


class SemanticChunker:
    """Build parent context chunks and child retrieval chunks from Markdown."""

    def __init__(
        self,
        child_min_tokens: int = 50,
        child_target_tokens: int = 400,
        child_max_tokens: int = 650,
        parent_target_tokens: int = 1200,
        parent_max_tokens: int = 1400,
        overlap_tokens: int = 120,
    ) -> None:
        self.child_min_tokens = child_min_tokens
        self.child_target_tokens = child_target_tokens
        self.child_max_tokens = child_max_tokens
        self.parent_target_tokens = parent_target_tokens
        self.parent_max_tokens = parent_max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, page: MarkdownPage) -> tuple[list[ParentChunk], list[ChildChunk]]:
        """Chunk a Markdown page into parent and child chunks."""

        blocks = self._parse_blocks(page.markdown)
        parents = self._build_parent_chunks(page, blocks)
        children: list[ChildChunk] = []
        for parent in parents:
            children.extend(self._build_child_chunks(page, parent, len(children)))
        return parents, children

    def _parse_blocks(self, markdown: str) -> list[MarkdownBlock]:
        blocks: list[MarkdownBlock] = []
        heading_stack: list[str] = []
        lines = markdown.splitlines()
        index = 0

        while index < len(lines):
            line = lines[index]

            if not line.strip():
                index += 1
                continue

            heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
            if heading:
                level = len(heading.group(1))
                title = heading.group(2).strip()
                heading_stack = heading_stack[: level - 1]
                heading_stack.append(title)
                blocks.append(MarkdownBlock("heading", line, heading_stack.copy(), heading_level=level))
                index += 1
                continue

            if line.startswith("```"):
                block_lines = [line]
                language_parts = line.removeprefix("```").strip().split(maxsplit=1)
                language = language_parts[0] if language_parts else ""
                index += 1
                while index < len(lines):
                    block_lines.append(lines[index])
                    if lines[index].startswith("```"):
                        index += 1
                        break
                    index += 1
                blocks.append(MarkdownBlock("code", "\n".join(block_lines), heading_stack.copy(), language=language))
                continue

            if self._is_table_line(line):
                table_lines = [line]
                index += 1
                while index < len(lines) and self._is_table_line(lines[index]):
                    table_lines.append(lines[index])
                    index += 1
                blocks.append(MarkdownBlock("table", "\n".join(table_lines), heading_stack.copy()))
                continue

            if self._is_list_line(line):
                list_lines = [line]
                index += 1
                while index < len(lines) and (
                    self._is_list_line(lines[index]) or lines[index].startswith(("  ", "\t"))
                ):
                    list_lines.append(lines[index])
                    index += 1
                blocks.append(MarkdownBlock("list", "\n".join(list_lines), heading_stack.copy()))
                continue

            paragraph_lines = [line]
            index += 1
            while index < len(lines):
                next_line = lines[index]
                if not next_line.strip():
                    break
                if re.match(r"^(#{1,6})\s+", next_line) or next_line.startswith("```"):
                    break
                if self._is_table_line(next_line) or self._is_list_line(next_line):
                    break
                paragraph_lines.append(next_line)
                index += 1
            blocks.append(MarkdownBlock("paragraph", "\n".join(paragraph_lines), heading_stack.copy()))

        return blocks

    def _build_parent_chunks(self, page: MarkdownPage, blocks: list[MarkdownBlock]) -> list[ParentChunk]:
        parents: list[ParentChunk] = []
        current_blocks: list[MarkdownBlock] = []

        for block in blocks:
            proposed = [*current_blocks, block]
            if current_blocks and block.kind == "heading" and self._is_semantic_boundary(block):
                parents.extend(self._flush_parent_blocks(page, current_blocks, len(parents)))
                current_blocks = [block]
                continue
            if current_blocks and self._blocks_token_count(proposed) > self.parent_max_tokens:
                parents.extend(self._flush_parent_blocks(page, current_blocks, len(parents)))
                current_blocks = [block]
                continue
            current_blocks.append(block)

        if current_blocks:
            parents.extend(self._flush_parent_blocks(page, current_blocks, len(parents)))

        return self._merge_short_parents(page, parents)

    def _flush_parent_blocks(
        self, page: MarkdownPage, blocks: list[MarkdownBlock], start_index: int
    ) -> list[ParentChunk]:
        if self._blocks_token_count(blocks) <= self.parent_max_tokens:
            return [self._make_parent(page, blocks, start_index)]

        parents: list[ParentChunk] = []
        current: list[MarkdownBlock] = []
        for block in blocks:
            proposed = [*current, block]
            if current and self._blocks_token_count(proposed) > self.parent_target_tokens:
                parents.append(self._make_parent(page, current, start_index + len(parents)))
                current = [block]
            else:
                current.append(block)

        if current:
            parents.append(self._make_parent(page, current, start_index + len(parents)))
        return parents

    def _make_parent(self, page: MarkdownPage, blocks: list[MarkdownBlock], chunk_index: int) -> ParentChunk:
        content = self._join_blocks(blocks)
        heading_path = self._dominant_heading_path(blocks)
        domain = self._infer_domain(page.page_title, heading_path, content)
        section_type = self._infer_section_type(heading_path, content)
        return ParentChunk(
            parent_id=f"{page.page_id}_p{chunk_index:03d}",
            page_id=page.page_id,
            page_title=page.page_title,
            heading_path=heading_path,
            content=content,
            token_count=self._count_tokens(content),
            chunk_index=chunk_index,
            source_url=page.source_url,
            space_key=page.space_key,
            last_modified=page.last_modified,
            hash=self._hash(content),
            domain=domain,
            section_type=section_type,
            block_types=self._block_types(blocks),
        )

    def _merge_short_parents(self, page: MarkdownPage, parents: list[ParentChunk]) -> list[ParentChunk]:
        if len(parents) < 2:
            return parents

        merged_contents: list[str] = []
        merged_paths: list[list[str]] = []
        merged_section_types: list[str] = []
        for parent in parents:
            can_merge = False
            if merged_contents:
                merged_candidate = f"{merged_contents[-1]}\n\n{parent.content}"
                can_merge = (
                    parent.token_count < self.child_min_tokens
                    and merged_section_types[-1] == parent.section_type
                    and self._count_tokens(merged_candidate) <= self.parent_max_tokens
                )
            if can_merge:
                merged_contents[-1] = f"{merged_contents[-1]}\n\n{parent.content}"
            else:
                merged_contents.append(parent.content)
                merged_paths.append(parent.heading_path)
                merged_section_types.append(parent.section_type)

        rebuilt: list[ParentChunk] = []
        for index, content in enumerate(merged_contents):
            blocks = [MarkdownBlock("paragraph", content, merged_paths[min(index, len(merged_paths) - 1)])]
            rebuilt.append(self._make_parent(page, blocks, index))
        return rebuilt

    def _build_child_chunks(self, page: MarkdownPage, parent: ParentChunk, start_index: int) -> list[ChildChunk]:
        blocks = self._parse_blocks(parent.content)
        block_groups = self._group_blocks_for_children(blocks)
        children: list[ChildChunk] = []
        previous_tail = ""

        for group in block_groups:
            content = self._join_blocks(group)
            if (
                children
                and self._count_tokens(content) < self.child_min_tokens
                and not self._contains_structure_sensitive_block(group)
            ):
                last = children.pop()
                content = f"{last.content}\n\n{content}"
                previous_tail = ""

            base_tokens = self._count_tokens(content)
            overlap_budget = max(0, self.child_max_tokens - base_tokens)
            overlap = (
                self._take_token_tail(previous_tail, min(self.overlap_tokens, overlap_budget))
                if previous_tail and overlap_budget > 0
                else ""
            )
            if self._count_tokens(overlap) > overlap_budget:
                overlap = ""
            final_content = f"{overlap}\n\n{content}".strip() if overlap else content
            chunk_index = start_index + len(children)
            heading_path = self._dominant_heading_path(group) or parent.heading_path
            domain = self._infer_domain(page.page_title, heading_path, final_content)
            section_type = self._infer_section_type(heading_path, final_content)
            block_types = self._block_types(group)
            keywords = self._extract_keywords(final_content, heading_path)
            code_languages = self._code_languages(group)
            embedding_text = build_embedding_text(
                page_title=page.page_title,
                heading_path=heading_path,
                content=final_content,
                domain=domain,
                section_type=section_type,
                block_types=block_types,
                keywords=keywords,
                tags=[],
            )
            children.append(
                ChildChunk(
                    chunk_id=f"{page.page_id}_{chunk_index:03d}",
                    parent_id=parent.parent_id,
                    page_id=page.page_id,
                    page_title=page.page_title,
                    content=final_content,
                    embedding_text=embedding_text,
                    heading_path=heading_path,
                    chunk_index=chunk_index,
                    token_count=self._count_tokens(final_content),
                    overlap_from_previous=self._count_tokens(overlap),
                    source_url=page.source_url,
                    space_key=page.space_key,
                    last_modified=page.last_modified,
                    hash=self._hash(final_content),
                    chunking_profile="",
                    domain=domain,
                    section_type=section_type,
                    block_types=block_types,
                    keywords=keywords,
                    has_code="code" in block_types,
                    has_table="table" in block_types,
                    has_list="list" in block_types,
                    code_languages=code_languages,
                    site=page.site,
                    product_version=page.product_version,
                    doc_key=page.doc_key,
                    is_eol=page.is_eol,
                )
            )
            previous_tail = content

        return children

    def _contains_structure_sensitive_block(self, blocks: list[MarkdownBlock]) -> bool:
        return any(block.kind in {"code", "table"} for block in blocks)

    def _group_blocks_for_children(self, blocks: list[MarkdownBlock]) -> list[list[MarkdownBlock]]:
        groups: list[list[MarkdownBlock]] = []
        current: list[MarkdownBlock] = []
        previous_context: MarkdownBlock | None = None

        for block in blocks:
            split_blocks = self._split_oversized_block(block)
            for split_block in split_blocks:
                if (
                    split_block.kind in {"code", "table"}
                    and not current
                    and previous_context is not None
                    and previous_context.kind in {"paragraph", "list"}
                ):
                    current.append(previous_context)
                proposed = [*current, split_block]
                if current and self._blocks_token_count(proposed) > self.child_target_tokens:
                    groups.append(current)
                    current = []
                    if (
                        split_block.kind in {"code", "table"}
                        and previous_context is not None
                        and previous_context.kind in {"paragraph", "list"}
                    ):
                        current.append(previous_context)
                    current.append(split_block)
                else:
                    current.append(split_block)
                if split_block.kind in {"paragraph", "list"}:
                    previous_context = split_block

        if current:
            groups.append(current)
        return groups

    def _split_oversized_block(self, block: MarkdownBlock) -> list[MarkdownBlock]:
        if self._count_tokens(block.content) <= self.child_max_tokens:
            return [block]
        if block.kind == "table":
            return self._split_table_block(block)
        if block.kind == "code":
            return self._split_code_block(block)
        return self._split_text_block(block)

    def _split_table_block(self, block: MarkdownBlock) -> list[MarkdownBlock]:
        lines = block.content.splitlines()
        if len(lines) <= 3:
            return [block]
        header = lines[:2]
        rows = lines[2:]
        chunks: list[MarkdownBlock] = []
        current_rows: list[str] = []
        for row in rows:
            proposed = [*header, *current_rows, row]
            if current_rows and self._count_tokens("\n".join(proposed)) > self.child_max_tokens:
                chunks.append(MarkdownBlock("table", "\n".join([*header, *current_rows]), block.heading_path))
                current_rows = [row]
            else:
                current_rows.append(row)
        if current_rows:
            chunks.append(MarkdownBlock("table", "\n".join([*header, *current_rows]), block.heading_path))
        return chunks or [block]

    def _split_code_block(self, block: MarkdownBlock) -> list[MarkdownBlock]:
        lines = block.content.splitlines()
        if len(lines) <= 3:
            return [block]
        opening = lines[0]
        closing = lines[-1] if lines[-1].startswith("```") else "```"
        body = lines[1:-1] if lines[-1].startswith("```") else lines[1:]
        chunks: list[MarkdownBlock] = []
        current: list[str] = []
        for line in body:
            proposed = [opening, *current, line, closing]
            if current and self._count_tokens("\n".join(proposed)) > self.child_max_tokens:
                chunks.append(
                    MarkdownBlock(
                        "code",
                        "\n".join([opening, *current, closing]),
                        block.heading_path,
                        language=block.language,
                    )
                )
                current = [line]
            else:
                current.append(line)
        if current:
            chunks.append(
                MarkdownBlock(
                    "code",
                    "\n".join([opening, *current, closing]),
                    block.heading_path,
                    language=block.language,
                )
            )
        return chunks or [block]

    def _split_text_block(self, block: MarkdownBlock) -> list[MarkdownBlock]:
        sentences = re.split(r"(?<=[.!?。！？])\s+", block.content)  # noqa: RUF001
        chunks: list[MarkdownBlock] = []
        current: list[str] = []
        for sentence in sentences:
            proposed = " ".join([*current, sentence]).strip()
            if current and self._count_tokens(proposed) > self.child_max_tokens:
                chunks.append(MarkdownBlock(block.kind, " ".join(current).strip(), block.heading_path))
                current = [sentence]
            else:
                current.append(sentence)
        if current:
            chunks.append(MarkdownBlock(block.kind, " ".join(current).strip(), block.heading_path))
        return chunks or [block]

    def _infer_domain(self, page_title: str, heading_path: list[str], content: str) -> str:
        haystack = self._normalize_for_match(" ".join([page_title, *heading_path, content[:1200]]))
        for domain, keywords in DOMAIN_KEYWORDS.items():
            if any(keyword in haystack for keyword in keywords):
                return domain
        return "운영매뉴얼"

    def _infer_section_type(self, heading_path: list[str], content: str) -> str:
        haystack = self._normalize_for_match(" ".join([*heading_path, content[:800]]))
        for section_type, keywords in SECTION_TYPE_KEYWORDS.items():
            if any(keyword in haystack for keyword in keywords):
                return section_type
        return "general"

    def _is_semantic_boundary(self, block: MarkdownBlock) -> bool:
        if block.heading_level and block.heading_level > 3:
            return False
        section_type = self._infer_section_type(block.heading_path, block.content)
        return section_type != "general"

    def _block_types(self, blocks: list[MarkdownBlock]) -> list[str]:
        return sorted({block.kind for block in blocks if block.kind != "heading"})

    def _code_languages(self, blocks: list[MarkdownBlock]) -> list[str]:
        return sorted({block.language for block in blocks if block.kind == "code" and block.language})

    def _extract_keywords(self, content: str, heading_path: list[str]) -> list[str]:
        candidates: list[str] = []
        candidates.extend(re.findall(r"`([^`\n]{2,80})`", content))
        candidates.extend(re.findall(r"\b(?:kubectl|docker|helm|curl|systemctl|journalctl)\s+[^\n`]{1,80}", content))
        candidates.extend(re.findall(r"\b[A-Z][A-Za-z0-9_]*(?:Error|Exception|BackOff|Failed|Timeout)\b", content))
        candidates.extend(heading_path[-2:])

        return self._dedupe_keywords(candidates, limit=8)

    def _normalize_for_match(self, text: str) -> str:
        return text.lower()

    def _dominant_heading_path(self, blocks: list[MarkdownBlock]) -> list[str]:
        for block in blocks:
            if block.heading_path:
                return block.heading_path
        return []

    def _join_blocks(self, blocks: list[MarkdownBlock]) -> str:
        return "\n\n".join(block.content.strip() for block in blocks if block.content.strip()).strip()

    def _blocks_token_count(self, blocks: list[MarkdownBlock]) -> int:
        return self._count_tokens(self._join_blocks(blocks))

    def _count_tokens(self, text: str) -> int:
        """Approximate tokens without external tokenizer dependency."""

        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))

    def _take_token_tail(self, text: str, max_tokens: int) -> str:
        selected: list[str] = []
        token_count = 0
        for line in reversed(text.splitlines()):
            selected.append(line)
            token_count += self._count_tokens(line)
            if token_count >= max_tokens:
                break
        tail = "\n".join(reversed(selected)).strip()
        return self._balance_code_fences(tail)

    def _balance_code_fences(self, text: str) -> str:
        fence_count = sum(1 for line in text.splitlines() if line.startswith("```"))
        if fence_count % 2 == 0:
            return text

        first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
        if first_line.startswith("```"):
            return f"{text}\n```"
        return f"```\n{text}"

    def _is_table_line(self, line: str) -> bool:
        return line.strip().startswith("|") and line.strip().endswith("|")

    def _is_list_line(self, line: str) -> bool:
        return bool(re.match(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", line))

    def _hash(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _dedupe_keywords(self, candidates: list[str], limit: int) -> list[str]:
        seen: set[str] = set()
        keywords: list[str] = []
        for candidate in candidates:
            normalized = re.sub(r"\s+", " ", candidate).strip(" .,;:()[]")
            key = normalized.lower()
            if not normalized or key in seen:
                continue
            seen.add(key)
            keywords.append(normalized)
            if len(keywords) >= limit:
                break
        return keywords


class ControlDocChunker(SemanticChunker):
    """Chunk control-like documents around decisions, ownership, and follow-up work."""

    def _infer_domain(self, page_title: str, heading_path: list[str], content: str) -> str:
        haystack = self._normalize_for_match(" ".join([page_title, *heading_path, content[:1200]]))
        if any(keyword in haystack for keyword in ("요구사항", "prd", "rfc", "정책", "설계", "기획")):
            return "기획서"
        return "회의록"

    def _infer_section_type(self, heading_path: list[str], content: str) -> str:
        heading_haystack = self._normalize_for_match(" ".join(heading_path))
        for section_type, keywords in CONTROL_SECTION_TYPE_KEYWORDS.items():
            if any(keyword in heading_haystack for keyword in keywords):
                return section_type
        haystack = self._normalize_for_match(content[:800])
        for section_type, keywords in CONTROL_SECTION_TYPE_KEYWORDS.items():
            if any(keyword in haystack for keyword in keywords):
                return section_type
        return super()._infer_section_type(heading_path, content)

    def _is_semantic_boundary(self, block: MarkdownBlock) -> bool:
        if block.heading_level and block.heading_level > 3:
            return False
        return self._infer_section_type(block.heading_path, block.content) in CONTROL_SECTION_TYPE_KEYWORDS

    def _extract_keywords(self, content: str, heading_path: list[str]) -> list[str]:
        candidates = super()._extract_keywords(content, heading_path)
        candidates.extend(re.findall(r"(?:담당자|owner|assignee)\s*[:：]\s*([^\n|]{1,40})", content, re.IGNORECASE))  # noqa: RUF001
        candidates.extend(re.findall(r"(?:기한|due date|마감)\s*[:：]\s*([^\n|]{1,40})", content, re.IGNORECASE))  # noqa: RUF001
        candidates.extend(heading_path[-2:])

        return self._dedupe_keywords(candidates, limit=12)


JsonlRow = dict[str, Any] | ParentChunk | ChildChunk


def build_embedding_text(
    *,
    page_title: str,
    heading_path: list[str],
    content: str,
    domain: str,
    section_type: str,
    block_types: list[str],
    keywords: list[str],
    tags: list[str],
    chunking_profile: str = "",
) -> str:
    """Build the final text sent to the embedding model."""

    heading_text = " > ".join(heading_path) if heading_path else page_title
    lines = [
        f"문서: {page_title}",
        f"도메인: {domain}",
        f"경로: {heading_text}",
    ]
    if chunking_profile:
        lines.append(f"청킹 프로필: {chunking_profile}")
    if section_type:
        lines.append(f"섹션 유형: {section_type}")
    if block_types:
        lines.append(f"블록 유형: {', '.join(block_types)}")
    if tags:
        lines.append(f"태그: {', '.join(tags)}")
    if keywords:
        lines.append(f"키워드: {', '.join(keywords)}")
    prefix = "\n".join(lines)
    return f"{prefix}\n\n{content}"


def child_chunk_to_index_record(chunk: ChildChunk) -> dict[str, Any]:
    """Convert a child chunk to the JSONL contract used before embedding/indexing."""

    return {
        "chunk_id": chunk.chunk_id,
        "parent_id": chunk.parent_id,
        "page_id": chunk.page_id,
        "title": chunk.page_title,
        "content": chunk.content,
        "embedding_text": chunk.embedding_text,
        "source_url": chunk.source_url,
        "hash": chunk.hash,
        "metadata": {
            "space_key": chunk.space_key,
            "last_modified": chunk.last_modified,
            "heading_path": chunk.heading_path,
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
            "overlap_from_previous": chunk.overlap_from_previous,
            "chunking_profile": chunk.chunking_profile,
            "domain": chunk.domain,
            "section_type": chunk.section_type,
            "block_types": chunk.block_types or [],
            "keywords": chunk.keywords or [],
            "tags": chunk.tags or [],
            "has_code": chunk.has_code,
            "has_table": chunk.has_table,
            "has_list": chunk.has_list,
            "code_languages": chunk.code_languages or [],
        },
    }


def write_jsonl(path: Path, rows: list[JsonlRow]) -> None:
    """Write dataclass or mapping rows as JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            payload = row if isinstance(row, dict) else asdict(row)
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")
