"""5도메인 단일 ontology — 라우터(질문 분류)와 문서 분류가 같은 정의를 공유한다.

정의가 두 곳에서 갈리면 soft 도메인 가산이 헛돈다(태깅 기준 ≠ 라우팅 기준 → #49 근본 원인).
같은 정의에서 관점만 달리해 각자 프롬프트를 생성한다:
    - 라우터: "이 질문이 어떤 종류의 근거를 요구하는가?"
    - 문서 분류: "이 문서가 어떤 종류의 질문에 근거를 제공하는가?"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# 정의(아래 DOMAIN_DEFINITIONS)가 바뀌면 올린다 → 문서 분류 캐시 무효화 키의 일부.
ONTOLOGY_VERSION = "2026-06-10"

Perspective = Literal["router", "document"]


@dataclass(frozen=True)
class DomainDef:
    key: str  # Domain enum 값과 일치
    label: str  # 한글 명칭
    definition: str  # 핵심 정의 (경계 보정 반영)
    boundary: str = ""  # 인접 도메인과의 경계 메모 (선택)


# 경계 보정: api_reference의 "사용법"은 manual과 겹쳐 제외, incident는 일반 점검과 구분.
DOMAIN_DEFINITIONS: tuple[DomainDef, ...] = (
    DomainDef(
        "incident",
        "장애대응",
        "실제 장애의 증상·영향·원인·복구·재발 방지",
        "일반 점검/트러블슈팅은 manual. 장애 증상+원인+복구 근거가 있을 때만.",
    ),
    DomainDef(
        "manual",
        "운영매뉴얼",
        "설치·설정·운영·점검의 절차와 작업 흐름",
        "'kubectl로 배포하는 전체 절차'는 manual.",
    ),
    DomainDef(
        "api_reference",
        "API명세",
        "API·명령·설정값의 정확한 문법·옵션·파라미터·반환값",
        "'사용법/절차'는 manual. 'kubectl get의 옵션·출력 필드'는 api_reference.",
    ),
    DomainDef("meeting_note", "회의록", "회의에서 논의하거나 결정한 내용"),
    DomainDef("planning", "기획서", "설계 목적·요구사항·정책·구조·트레이드오프"),
)

DOMAIN_KEYS: tuple[str, ...] = tuple(d.key for d in DOMAIN_DEFINITIONS)

_PERSPECTIVE_HEADER: dict[Perspective, str] = {
    "router": "질문이 '어떤 종류의 근거를 요구하는가'로 분류한다.",
    "document": "문서가 '어떤 종류의 질문에 근거를 제공하는가'로 분류한다.",
}


def domain_definition_block(perspective: Perspective) -> str:
    """ontology에서 프롬프트용 도메인 정의 블록을 생성한다. 관점 문구만 달라진다."""
    lines = [_PERSPECTIVE_HEADER[perspective]]
    for d in DOMAIN_DEFINITIONS:
        line = f"- {d.key} ({d.label}): {d.definition}"
        if d.boundary:
            line += f" [경계: {d.boundary}]"
        lines.append(line)
    return "\n".join(lines)
