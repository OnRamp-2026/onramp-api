"""RAGAS judge 단위 테스트 — ragas 미설치/설치 분기 + 집계 로직 (실 ragas/LLM 불필요)."""

import app.eval.ragas_judge as judge_mod
from app.config import Settings
from app.eval.generation_adapter import GenerationResult
from app.eval.ragas_judge import GenerationScores, _mean, ragas_available, resolve_judge_model, score_generation


def _result(query="q", answer="상황: x", contexts=("문맥",)) -> GenerationResult:
    return GenerationResult(
        query=query, answer_text=answer, retrieved_contexts=list(contexts), answerability_status="answerable", n_docs=1
    )


def test_mean_ignores_none_and_nan() -> None:
    assert _mean([1.0, 0.0]) == 0.5
    assert _mean([float("nan"), 1.0]) == 1.0
    assert _mean([]) is None
    assert _mean([float("nan")]) is None


def test_resolve_judge_model_prefers_gpt_else_default() -> None:
    assert resolve_judge_model(Settings(default_model="gpt-4o")) == "gpt-4o"
    assert resolve_judge_model(Settings(default_model="o3-mini")) == "o3-mini"
    # 비-OpenAI(또는 빈값) → 기본 judge 모델
    assert resolve_judge_model(Settings(default_model="claude-3")) == judge_mod.DEFAULT_JUDGE_MODEL
    assert resolve_judge_model(Settings(default_model="")) == judge_mod.DEFAULT_JUDGE_MODEL


def test_ragas_available_returns_bool() -> None:
    assert isinstance(ragas_available(), bool)


async def test_score_generation_all_skipped() -> None:
    # 평가가능한 샘플이 없으면 ragas를 부르지 않고 None/스킵 집계
    holds = [
        GenerationResult(query="q1", answer_text="", retrieved_contexts=["c"]),  # 답변 보류
        GenerationResult(query="q2", answer_text="a", retrieved_contexts=[]),  # 문맥 없음
    ]
    scores = await score_generation(holds)
    assert scores == GenerationScores(None, None, 0, 2)


async def test_score_generation_aggregates_with_stubbed_scorers(monkeypatch) -> None:
    # ragas 채점기를 monkeypatch해 집계 로직만 검증 (실 ragas/LLM 없이)
    monkeypatch.setattr(judge_mod, "_build_evaluator", lambda settings: (object(), object()))

    class _Faith:
        def __init__(self, **_):
            pass

        async def single_turn_ascore(self, _sample):
            return 0.8

    class _Rel:
        def __init__(self, **_):
            pass

        async def single_turn_ascore(self, _sample):
            return 0.6

    class _Sample:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    import sys
    import types

    # ragas.dataset_schema.SingleTurnSample / ragas.metrics.{Faithfulness,ResponseRelevancy} 가짜 주입
    schema_mod = types.ModuleType("ragas.dataset_schema")
    schema_mod.SingleTurnSample = _Sample
    metrics_mod = types.ModuleType("ragas.metrics")
    metrics_mod.Faithfulness = _Faith
    metrics_mod.ResponseRelevancy = _Rel
    monkeypatch.setitem(sys.modules, "ragas.dataset_schema", schema_mod)
    monkeypatch.setitem(sys.modules, "ragas.metrics", metrics_mod)

    scores = await score_generation([_result(), _result(query="q2")])
    assert scores.n_evaluated == 2
    assert scores.n_skipped == 0
    assert scores.faithfulness == 0.8
    assert scores.answer_relevancy == 0.6
    d = scores.as_dict()
    assert d["faithfulness"] == 0.8 and d["n_evaluated"] == 2


async def test_score_generation_skips_failed_samples(monkeypatch) -> None:
    monkeypatch.setattr(judge_mod, "_build_evaluator", lambda settings: (object(), object()))

    class _Boom:
        def __init__(self, **_):
            pass

        async def single_turn_ascore(self, _sample):
            raise RuntimeError("judge 호출 실패")

    class _Sample:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    import sys
    import types

    schema_mod = types.ModuleType("ragas.dataset_schema")
    schema_mod.SingleTurnSample = _Sample
    metrics_mod = types.ModuleType("ragas.metrics")
    metrics_mod.Faithfulness = _Boom
    metrics_mod.ResponseRelevancy = _Boom
    monkeypatch.setitem(sys.modules, "ragas.dataset_schema", schema_mod)
    monkeypatch.setitem(sys.modules, "ragas.metrics", metrics_mod)

    # 전부 실패해도 예외 없이 None 집계. 채점 실패 샘플은 n_evaluated에서 빠지고 n_skipped로 잡힌다.
    scores = await score_generation([_result()])
    assert scores.faithfulness is None
    assert scores.answer_relevancy is None
    assert scores.n_evaluated == 0
    assert scores.n_skipped == 1
