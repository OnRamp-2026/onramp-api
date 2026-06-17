"""답변 포맷 정책 — 라우터 의도(domains)로 structured/freeform 결정 (#191).

포맷은 **라우터에서 의도-time에 1회 결정**해 state["answer_format"]에 박는다.
이렇게 해야 Trust 재검색이 domains를 변형(EXPAND_TOPICS 시 [] 로 도메인 해제)해도
포맷이 흔들리지 않는다 — domains는 '검색 필터'(가변), answer_format은 '의도'(불변)로 분리.
"""

from __future__ import annotations

from app.agents.state import Domain


def decide_answer_format(domains: list[Domain], structured_domains: set[str]) -> str:
    """domains가 structured 집합과 교집합이면 'structured', 아니면 'freeform'.

    domains가 비었거나(라우터 애매) 교집합이 없으면 freeform(안전한 기본).
    """
    return "structured" if any(d in structured_domains for d in domains) else "freeform"
