"""Rule-based chunk metadata classifier for P0 ingestion."""

from __future__ import annotations

import re
from dataclasses import replace

from app.rag.chunker import ChildChunk, build_embedding_text

DOMAIN_RULES: dict[str, tuple[str, ...]] = {
    "incident": (
        "장애",
        "incident",
        "outage",
        "postmortem",
        "root cause",
        "원인",
        "영향",
        "복구",
        "재발",
        "CrashLoopBackOff",
    ),
    "api_reference": ("api", "endpoint", "request", "response", "http", "status code", "요청", "응답", "에러 코드"),
    "meeting_note": ("회의", "minutes", "참석", "결정사항", "action item", "액션아이템", "논의"),
    "planning": ("기획", "요구사항", "prd", "rfc", "정책", "목표", "범위", "사용자 시나리오", "architecture"),
    "manual": (
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
        "kubectl",
        "helm",
    ),
}

KOREAN_DOMAIN_MAP = {
    "장애대응": "incident",
    "운영매뉴얼": "manual",
    "API명세": "api_reference",
    "회의록": "meeting_note",
    "기획서": "planning",
}

TAG_RULES: dict[str, tuple[str, ...]] = {
    "kubernetes": ("kubectl", "kubernetes", "pod", "deployment", "secret", "namespace", "node"),
    "prometheus": ("prometheus", "promql", "alertmanager", "metric"),
    "datadog": ("datadog", "monitor", "apm", "log explorer"),
    "apache": ("apache", "httpd", "virtualhost"),
    "incident": ("장애", "incident", "outage", "postmortem", "복구", "재발"),
    "runbook": ("runbook", "매뉴얼", "절차", "운영", "검증", "롤백"),
    "api": ("api", "endpoint", "request", "response", "http"),
}


class AutoClassifier:
    """Refine chunk metadata without sending unmasked content to an LLM."""

    def classify_chunk(self, chunk: ChildChunk) -> ChildChunk:
        """Return a copy of the chunk with final P0 metadata."""

        domain = self._infer_domain(chunk)
        section_type = self._infer_section_type(chunk)
        keywords = self._merge_keywords(chunk)
        tags = self._infer_tags(chunk, domain, keywords)
        block_types = chunk.block_types or []
        embedding_text = build_embedding_text(
            page_title=chunk.page_title,
            heading_path=chunk.heading_path,
            content=chunk.content,
            domain=domain,
            section_type=section_type,
            block_types=block_types,
            keywords=keywords,
            tags=tags,
        )
        return replace(
            chunk,
            domain=domain,
            section_type=section_type,
            keywords=keywords,
            tags=tags,
            embedding_text=embedding_text,
        )

    def classify_batch(self, chunks: list[ChildChunk]) -> list[ChildChunk]:
        """Classify chunks in order."""

        return [self.classify_chunk(chunk) for chunk in chunks]

    def _infer_domain(self, chunk: ChildChunk) -> str:
        existing = KOREAN_DOMAIN_MAP.get(chunk.domain, chunk.domain)
        haystack = self._haystack(chunk)
        for domain, keywords in DOMAIN_RULES.items():
            if any(keyword.lower() in haystack for keyword in keywords):
                return domain
        return existing if existing in DOMAIN_RULES else "manual"

    def _infer_section_type(self, chunk: ChildChunk) -> str:
        if chunk.section_type and chunk.section_type != "general":
            return chunk.section_type
        haystack = self._haystack(chunk)
        if any(word in haystack for word in ("결정", "decision", "결정사항")):
            return "decision"
        if any(word in haystack for word in ("action item", "액션아이템", "follow-up", "재발")):
            return "prevention"
        if any(word in haystack for word in ("롤백", "rollback", "revert")):
            return "rollback"
        if any(word in haystack for word in ("검증", "확인", "check", "verify")):
            return "verification"
        if any(word in haystack for word in ("절차", "방법", "steps", "how to")):
            return "procedure"
        if any(word in haystack for word in ("원인", "root cause", "cause")):
            return "root_cause"
        if any(word in haystack for word in ("api", "endpoint", "request", "response", "요청", "응답")):
            return "api_contract"
        if any(word in haystack for word in ("영향", "impact", "affected")):
            return "impact"
        if any(word in haystack for word in ("조치", "대응", "복구", "resolution")):
            return "mitigation"
        return "general"

    def _merge_keywords(self, chunk: ChildChunk) -> list[str]:
        candidates = list(chunk.keywords or [])
        candidates.extend(
            re.findall(r"\b(?:kubectl|docker|helm|curl|systemctl|journalctl)\s+[^\n`]{1,80}", chunk.content)
        )
        candidates.extend(
            re.findall(r"\b[A-Z][A-Za-z0-9_]*(?:Error|Exception|BackOff|Failed|Timeout)\b", chunk.content)
        )
        candidates.extend(chunk.heading_path[-2:])
        return self._dedupe(candidates, limit=12)

    def _infer_tags(self, chunk: ChildChunk, domain: str, keywords: list[str]) -> list[str]:
        haystack = " ".join([self._haystack(chunk), " ".join(keywords)]).lower()
        tags = [domain]
        for tag, rules in TAG_RULES.items():
            if any(rule.lower() in haystack for rule in rules):
                tags.append(tag)
        if chunk.has_code:
            tags.append("code")
        if chunk.has_table:
            tags.append("table")
        if chunk.has_list:
            tags.append("list")
        tags.extend(chunk.code_languages or [])
        return self._dedupe(tags, limit=12)

    def _haystack(self, chunk: ChildChunk) -> str:
        return " ".join(
            [chunk.page_title, *chunk.heading_path, chunk.domain, chunk.section_type, chunk.content[:1500]]
        ).lower()

    def _dedupe(self, values: list[str], limit: int) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            normalized = re.sub(r"\s+", " ", value).strip(" .,;:()[]")
            key = normalized.lower()
            if not normalized or key in seen:
                continue
            seen.add(key)
            result.append(normalized)
            if len(result) >= limit:
                break
        return result
