"""Router Agent 실 LLM 통합 테스트 (opt-in / 수동 실행용).

실제 OpenAI를 호출하므로 비용이 발생하고 OPENAI_API_KEY가 필요하다.
CI 자동 실행과 비용을 막기 위해 기본은 전체를 주석 처리해 둔다.

수동 검증 시:
    1) 아래 블록의 주석을 해제
    2) OPENAI_API_KEY 설정
    3) pytest tests/integration/test_router_live.py -v

주의: LLM 분류는 비결정적이고 모호한 질문은 흔들릴 수 있어, 명확한 케이스만 단언한다.
"""

# import os
#
# import pytest
#
# from app.agents.router.node import route_node
# from app.agents.state import Domain, UseCase
#
# pytestmark = pytest.mark.skipif(
#     not os.getenv("OPENAI_API_KEY"), reason="실 LLM 키(OPENAI_API_KEY) 필요"
# )
#
#
# @pytest.mark.parametrize(
#     "query, expected_domain",
#     [
#         ("EKS Pod가 CrashLoopBackOff 상태인데 어떻게 해결해?", Domain.INCIDENT),
#         ("DB 커넥션 풀 고갈로 서비스가 죽었을 때 대응 절차", Domain.INCIDENT),
#         ("결제 API 응답에 어떤 필드가 내려오는지 명세 알려줘", Domain.API_SPEC),
#         ("지난 스프린트 회고 회의에서 결정된 액션아이템 정리해줘", Domain.MEETING_NOTES),
#     ],
# )
# async def test_live_search_classification(query, expected_domain):
#     out = await route_node({"query": query})
#     assert out["use_case"] == UseCase.SEARCH
#     assert out["domain"] == expected_domain
#     assert out["refined_query"]  # 정제 쿼리가 생성됨
#
#
# async def test_live_unanswerable():
#     out = await route_node({"query": "오늘 점심 뭐 먹을까?"})
#     assert out["use_case"] == UseCase.UNANSWERABLE
#     assert out["refined_query"] == ""
#     assert out["answerability_reason"]
