"""문서 도메인 분류 — 페이지 단위 멀티라벨 계약 (P1, #49).

스키마/채택 규칙(결정론) + 페이지 단위 LLM 분류(고정 temp, 재시도→rule fallback).
분류기는 페이지 텍스트를 받아 도메인만 판정한다 — 마스킹/캐시/재색인은 색인 파이프라인의 책임이라 여기서 다루지 않는다
(입력 텍스트의 민감정보 마스킹은 호출측 upstream에서 끝낸 상태를 전제).

계약: https://github.com/OnRamp-2026/docs/blob/main/Jihong/fixes/49_doc_domain_classifier.md
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from app.config import Settings, get_settings
from app.middleware.error_handler import LLMError
from app.rag.classifier import DOMAIN_RULES
from app.rag.domains import DOMAIN_KEYS, domain_definition_block
from app.services.llm_selector import call_llm

logger = logging.getLogger(__name__)

# 프롬프트가 바뀌면 올린다 → dry-run 결과 재사용 키의 일부(같은 페이지라도 재분류).
DOC_CLASSIFIER_PROMPT_VERSION = "1"

# LLM에 보낼 페이지 텍스트 길이 상한.
_MAX_CONTENT_CHARS = 6000
_HEADING_RE = re.compile(r"^#{1,6}\s")

# secondary 채택 임계값 — confidence가 이 값 이상이고 evidence_headings가 있을 때만 채택.
# (Step 2에서 config로 노출·튜닝)
DEFAULT_SECONDARY_THRESHOLD = 0.6
MAX_SECONDARY = 2  # primary 외 최대 secondary 수


class DomainEvidence(BaseModel):
    """한 도메인에 대한 페이지 분류 근거. evidence_headings는 청크 단위 매칭에 쓰므로 heading 참조여야 한다."""

    domain: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_headings: list[str] = Field(default_factory=list)

    @field_validator("evidence_headings")
    @classmethod
    def _strip_blank_headings(cls, value: list[str]) -> list[str]:
        # 공백뿐인 heading([" "])이 근거로 인정돼 secondary 채택을 우회하지 못하게 정규화
        return [h.strip() for h in value if h and h.strip()]

    @model_validator(mode="after")
    def _check_domain(self) -> DomainEvidence:
        if self.domain not in DOMAIN_KEYS:
            raise ValueError(f"알 수 없는 도메인: {self.domain!r} (가능: {', '.join(DOMAIN_KEYS)})")
        return self


class PageDomainClassification(BaseModel):
    """페이지당 LLM 출력. primary 1개 + 도메인별 근거. domains[0]은 반드시 primary."""

    primary_domain: str
    domains: list[DomainEvidence]

    @model_validator(mode="after")
    def _check_structure(self) -> PageDomainClassification:
        if self.primary_domain not in DOMAIN_KEYS:
            raise ValueError(f"알 수 없는 primary_domain: {self.primary_domain!r}")
        if not self.domains:
            raise ValueError("domains는 최소 1개(primary)를 포함해야 합니다")
        if self.domains[0].domain != self.primary_domain:
            raise ValueError("domains[0].domain은 primary_domain과 일치해야 합니다")
        keys = [d.domain for d in self.domains]
        if len(keys) != len(set(keys)):
            raise ValueError(f"domains에 중복 도메인이 있습니다: {keys}")
        return self


def adopt_domains(
    classification: PageDomainClassification,
    *,
    secondary_threshold: float = DEFAULT_SECONDARY_THRESHOLD,
) -> list[str]:
    """색인할 domains[] 결정 — primary + (임계값 이상 & 근거 heading 있는) secondary.

    단순 키워드 등장(근거 없음)·저신뢰 secondary는 버린다. 반환 첫 값은 항상 primary.
    secondary는 **confidence 내림차순 상위 MAX_SECONDARY개** — LLM 출력 순서에 의존하지 않는다.
    """
    qualified = [
        ev for ev in classification.domains[1:] if ev.confidence >= secondary_threshold and ev.evidence_headings
    ]
    qualified.sort(key=lambda ev: ev.confidence, reverse=True)
    return [classification.primary_domain, *(ev.domain for ev in qualified[:MAX_SECONDARY])]


def build_doc_classifier_system_prompt() -> str:
    """공유 ontology에서 문서 분류 프롬프트를 생성한다(문서 관점). 마스킹된 본문만 입력으로 받는다."""
    return f"""너는 사내 기술 문서 분류기다. 마스킹된 문서를 읽고 어떤 도메인의 질문에 근거를 제공하는지 판정한다.

[5도메인 정의] {domain_definition_block("document")}

[판정 규칙]
- primary_domain은 정확히 1개. 문서의 대표 도메인.
- secondary는 그 도메인의 '검색 질문에 실제 근거를 제공'할 때만 추가한다(단순 키워드 등장만으론 불가). 최대 {MAX_SECONDARY}개.
- 각 도메인의 evidence_headings는 근거가 된 실제 heading을 적는다(자유 문장 아님).
- domains[0]은 반드시 primary_domain.

[출력 형식] — JSON만 반환한다. 설명 없이.
{{"primary_domain": "<key>", "domains": [{{"domain": "<key>", "confidence": <0~1>, "evidence_headings": ["..."]}}]}}
"""


@dataclass(frozen=True)
class ClassificationResult:
    """페이지 분류 결과. source로 LLM 성공/규칙 폴백을 구분한다(폴백은 검수 시 신뢰도 낮음 표시)."""

    classification: PageDomainClassification
    adopted_domains: list[str]  # adopt_domains 적용 후 색인할 domains[]
    source: Literal["llm", "rule_fallback"]


def _rule_primary(text: str) -> str:
    """키워드 규칙으로 primary 1개 추론 — LLM 실패 시 폴백용(기존 DOMAIN_RULES 재사용)."""
    low = text.lower()
    for domain, keywords in DOMAIN_RULES.items():
        if any(keyword.lower() in low for keyword in keywords):
            return domain
    return "manual"


def rule_fallback_classification(page_title: str, content: str) -> PageDomainClassification:
    """LLM 실패 시 규칙 기반 단일 도메인(secondary 없음). confidence 0.0 — 근거 없음 신호."""
    primary = _rule_primary(f"{page_title}\n{content}")
    return PageDomainClassification(
        primary_domain=primary,
        domains=[DomainEvidence(domain=primary, confidence=0.0, evidence_headings=[])],
    )


def _split_sections(markdown: str) -> list[tuple[str, str]]:
    """(heading_line, body) 목록으로 분할. heading 없는 선두 본문은 ("", body)로 둔다."""
    sections: list[tuple[str, str]] = []
    heading = ""
    body_lines: list[str] = []
    for line in markdown.splitlines():
        if _HEADING_RE.match(line):
            if heading or body_lines:
                sections.append((heading, "\n".join(body_lines).strip()))
            heading, body_lines = line, []
        else:
            body_lines.append(line)
    if heading or body_lines:
        sections.append((heading, "\n".join(body_lines).strip()))
    return sections


def _sample_headings(headings: list[str], max_chars: int) -> str:
    """heading만으로도 상한 초과 시: 앞·중간·뒤를 균등 샘플해 들어가는 최대 개수만 남긴다(tail 절단 아님)."""
    n = len(headings)
    for k in range(n, 0, -1):
        idxs = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)}) if k > 1 else [0]
        text = "\n".join(headings[i] for i in idxs)
        if len(text) <= max_chars:
            return text
    return headings[0][:max_chars]


def heading_aware_sample(markdown: str, *, max_chars: int = _MAX_CONTENT_CHARS) -> str:
    """heading 골격을 우선 보존하며 본문을 균등 샘플링한다(단순 앞부분 절단 금지).

    - heading 합이 상한 이내면 **모든 heading 보존** + 남은 예산을 섹션 본문에 균등 배분.
    - heading 합이 상한 초과면 **앞·중간·뒤 heading을 균등 샘플**(tail 절단으로 뒤 heading이 사라지지 않게).
    evidence_headings 판정을 위해 LLM이 문서 전체의 heading 골격을 보게 하는 것이 목적.
    """
    if len(markdown) <= max_chars:
        return markdown

    sections = _split_sections(markdown)
    headings = [h for h, _ in sections if h]
    if not headings:  # heading 없는 문서 → 앞부분 절단 폴백
        return markdown[:max_chars]

    heading_cost = sum(len(h) + 1 for h in headings)  # +1: 줄 구분
    if heading_cost > max_chars:
        return _sample_headings(headings, max_chars)

    # 모든 heading 보존 + 남은 예산을 본문에 균등 배분.
    # 본문 예산에서 섹션당 줄바꿈 1칸을 빼 두면 join 후에도 상한을 넘지 않아 tail 절단이 불필요(heading 안전).
    bodied = sum(1 for _, b in sections if b)
    per_body = max((max_chars - heading_cost - bodied) // max(bodied, 1), 0)
    parts: list[str] = []
    for h, body in sections:
        if h:
            parts.append(h)
        if body and per_body:
            parts.append(body[:per_body])
    return "\n".join(parts)


def _build_user_prompt(page_title: str, content: str) -> str:
    return f"제목: {page_title}\n\n본문:\n{heading_aware_sample(content)}"


class DocumentDomainClassifier:
    """페이지 텍스트를 LLM으로 분류해 멀티라벨 도메인을 낸다. 실패 시 규칙 폴백.

    입력 텍스트의 마스킹은 upstream(색인 파이프라인) 책임이며 여기서 수행하지 않는다.
    temperature는 0.0 고정(결정론).
    """

    def __init__(self, settings: Settings | None = None, *, max_retries: int = 1) -> None:
        self.settings = settings or get_settings()
        self.max_retries = max_retries

    async def classify_page(
        self,
        *,
        page_title: str,
        content: str,
        secondary_threshold: float = DEFAULT_SECONDARY_THRESHOLD,
    ) -> ClassificationResult:
        system = build_doc_classifier_system_prompt()
        user = _build_user_prompt(page_title, content)
        for attempt in range(self.max_retries + 1):
            try:
                raw = await call_llm(
                    system,
                    user,
                    model=self.settings.classifier_model,
                    temperature=0.0,
                    json_mode=True,
                    settings=self.settings,
                )
                classification = PageDomainClassification.model_validate_json(raw)
                adopted = adopt_domains(classification, secondary_threshold=secondary_threshold)
                return ClassificationResult(classification, adopted, "llm")
            except (LLMError, ValidationError) as exc:
                logger.warning("문서 도메인 분류 실패(%d/%d): %s", attempt + 1, self.max_retries + 1, exc)
        # 재시도 소진 → 규칙 폴백
        fallback = rule_fallback_classification(page_title, content)
        return ClassificationResult(
            fallback, adopt_domains(fallback, secondary_threshold=secondary_threshold), "rule_fallback"
        )
