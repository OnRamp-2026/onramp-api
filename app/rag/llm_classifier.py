"""문서 단위 LLM 도메인 분류기 (적재 파이프라인 옵션).

룰 기반 분류(``classifier.py``)의 first-match 한계를 보완하기 위해, 문서 1건을 LLM에 보내
5개 도메인 중 하나로 분류한다. 마스킹된 본문만 보내므로 임베딩 파이프라인과 동일한 노출 수준.

설계: docs Baemin/05_llm_domain_classification.md
- 5개 도메인 유지 + tie-break 규칙(형식축 우선) + secondary/confidence
- 실패·비활성 시 호출부가 룰로 fallback 하도록 ``classify``는 None을 반환할 수 있다.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from app.rag.classifier import DOMAIN_RULES
from app.services.llm_selector import call_llm

logger = logging.getLogger(__name__)

# 룰과 동일한 5분류 체계 (단일 ontology). 키는 classifier.DOMAIN_RULES와 일치.
DOMAIN_DEFINITIONS: dict[str, str] = {
    "incident": "장애/인시던트 대응 — 장애 타임라인·영향·원인(root cause)·복구·재발방지(postmortem).",
    "api_reference": "API 명세/레퍼런스 — 엔드포인트·요청/응답·HTTP 상태코드·에러코드.",
    "meeting_note": "회의록 — 참석자·논의·결정사항·액션아이템·스프린트 미팅 기록.",
    "planning": "기획/설계 — 요구사항·PRD/RFC·정책·목표·범위·아키텍처 설계.",
    "manual": "운영 매뉴얼/런북 — 설치·절차·검증·롤백·트러블슈팅·운영 명령(kubectl/helm).",
}

# tie-break — 형식/주제 축 혼재로 인한 최대 오분류(회의록→incident, 튜토리얼→api)를 직접 겨냥.
_TIE_BREAK = (
    "판정 우선순위:\n"
    "- 회의록·기획 형식 문서는 내용이 장애/API를 다뤄도 형식(meeting_note/planning) 우선.\n"
    "- incident는 실제 장애 사후분석(postmortem·outage)에만 부여.\n"
    "- api_reference는 REST/HTTP API의 엔드포인트·요청/응답·상태코드 스펙이 주가 될 때만.\n"
    "  서버/미들웨어의 모듈·지시어(directive)·설정 항목 설명(예: Apache/nginx mod_* 문서,\n"
    "  설정 디렉티브 레퍼런스)은 api_reference가 아니라 manual로 분류한다.\n"
)

_SYSTEM_PROMPT = (
    "너는 사내 문서를 아래 5개 도메인 중 하나로 분류하는 분류기다.\n\n"
    + "\n".join(f"- {k}: {v}" for k, v in DOMAIN_DEFINITIONS.items())
    + "\n\n"
    + _TIE_BREAK
    + '\n반드시 JSON만 출력: '
    + '{"domain": "<위 5개 중 하나>", "secondary": "<두번째로 맞는 도메인 또는 \\"\\">", '
    + '"confidence": <0~1 float>}'
)

_MAX_CHARS = 3000  # 문서 앞부분만으로 도메인 판정엔 충분 — 비용/지연 절감


@dataclass(frozen=True)
class DomainResult:
    """LLM 문서 도메인 분류 결과."""

    domain: str
    secondary: str
    confidence: float


class DocumentDomainClassifier:
    """문서(제목+본문) → 단일 도메인. 실패 시 None 반환(호출부가 룰 fallback)."""

    def __init__(self, *, model: str = "", timeout: float = 20.0) -> None:
        self.model = model
        self.timeout = timeout

    async def classify(self, title: str, markdown: str) -> DomainResult | None:
        user_prompt = f"제목: {title}\n\n본문:\n{(markdown or '')[:_MAX_CHARS]}"
        try:
            raw = await call_llm(
                _SYSTEM_PROMPT,
                user_prompt,
                model=self.model,
                temperature=0.0,
                max_tokens=200,
                timeout=self.timeout,
                json_mode=True,
            )
            data = json.loads(raw)
        except Exception as exc:  # noqa: BLE001 — 분류는 best-effort, 실패는 룰 fallback
            logger.warning("LLM 도메인 분류 실패 — 룰 fallback: %s", exc)
            return None

        domain = str(data.get("domain", "")).strip()
        if domain not in DOMAIN_RULES:
            logger.warning("LLM이 알 수 없는 도메인 반환(%r) — 룰 fallback", domain)
            return None

        secondary = str(data.get("secondary", "")).strip()
        if secondary not in DOMAIN_RULES or secondary == domain:
            secondary = ""

        try:
            confidence = float(data.get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        confidence = min(1.0, max(0.0, confidence))

        return DomainResult(domain=domain, secondary=secondary, confidence=confidence)
