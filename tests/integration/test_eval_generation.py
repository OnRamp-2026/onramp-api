"""생성 평가 스모크 (실 ragas + LLM). 의존성/키 없으면 skip — #C 비차단 검증용.

ragas 미설치, OPENAI_API_KEY 없음, 또는 Qdrant 미가동이면 skip.
1~2문항으로 Faithfulness/Answer Relevancy가 0~1 범위로 산출되는지만 확인한다(비용 최소).
"""

import pytest

from app.config import get_settings
from app.eval.generation_adapter import GenerationResult
from app.eval.ragas_judge import ragas_available, score_generation


@pytest.fixture
def _require_ragas_and_key():
    if not ragas_available():
        pytest.skip('ragas 미설치 (uv pip install -e ".[eval]")')
    if not get_settings().openai_api_key:
        pytest.skip("OPENAI_API_KEY 미설정 — LLM-judge 호출 불가")


@pytest.mark.asyncio
async def test_score_generation_smoke(_require_ragas_and_key) -> None:
    # 답변이 문맥에 충실한 케이스 — Faithfulness가 산출되는지(0~1)만 검증
    results = [
        GenerationResult(
            query="Redis 커넥션 풀 고갈은 어떻게 대응하나요?",
            answer_text="해결: Redis 커넥션 풀 최대치를 늘리고 누수 커넥션을 회수합니다.",
            retrieved_contexts=[
                "Redis 커넥션 풀이 고갈되면 max pool size를 늘리고, 반환되지 않는 커넥션 누수를 점검한다.",
            ],
            answerability_status="answerable",
            n_docs=1,
        ),
    ]
    scores = await score_generation(results)
    assert scores.n_evaluated == 1
    # judge가 None을 줄 수도 있으나, 값이 있으면 0~1 범위여야 한다
    if scores.faithfulness is not None:
        assert 0.0 <= scores.faithfulness <= 1.0
    if scores.answer_relevancy is not None:
        assert 0.0 <= scores.answer_relevancy <= 1.0
