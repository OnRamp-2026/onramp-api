"""문서 도메인 분류 — 페이지 단위 멀티라벨 계약 (P1, #49).

Step 1 범위: 공유 ontology 기반 프롬프트 + 출력 스키마 검증 + secondary 채택 규칙.
LLM 호출·캐시·rule fallback 배선은 Step 2(dry-run)에서. 여기서는 결정론 로직만 둔다.

계약: docs/Jihong/fixes/49_doc_domain_classifier.md
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from app.rag.domains import DOMAIN_KEYS, domain_definition_block

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
    """
    adopted = [classification.primary_domain]
    for ev in classification.domains[1:]:
        if len(adopted) - 1 >= MAX_SECONDARY:
            break
        if ev.confidence >= secondary_threshold and ev.evidence_headings:
            adopted.append(ev.domain)
    return adopted


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
