"""생성 평가를 Langfuse Dataset Run(Experiment)으로 — RAGAS faithfulness·answer_relevancy (#120).

골든셋(onramp-retrieval-golden) 위에서 **실 그래프**로 답변을 생성하고, RAGAS로 채점해
faithfulness·answer_relevancy를 **run-level 점수**로 Langfuse에 기록 → 생성 품질을 대시보드/Run으로 본다.

ragas 필요(`.[eval]` extra) → prod 이미지 아닌 **별도 eval 이미지**로 실행(Dockerfile.eval).
필요 env: LANGFUSE_*(활성) + OPENAI_API_KEY + Qdrant 접근.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from langfuse import Evaluation  # noqa: E402

from app.eval.generation_adapter import GenerationResult, generate_for_eval  # noqa: E402
from app.eval.ragas_judge import ragas_available, score_generation  # noqa: E402
from app.observability import get_langfuse_client  # noqa: E402

DATASET_NAME = "onramp-retrieval-golden"


async def _task(*, item, **kwargs):
    """item(query) → 실 그래프로 답변 생성 → RAGAS 입력 dict."""
    inp = item.input or {}
    exp = item.expected_output or {}
    res = await generate_for_eval(
        inp.get("query", ""),
        domain=inp.get("domain"),
        reference=exp.get("ground_truth_answer"),
    )
    return {
        "query": res.query,
        "answer_text": res.answer_text,
        "retrieved_contexts": res.retrieved_contexts,
        "reference": res.reference,
        "evaluable": res.is_evaluable,
    }


async def _ragas_run_evaluator(*, item_results, **kwargs):
    """모든 item 출력을 모아 RAGAS 배치 채점 → run-level faithfulness·answer_relevancy."""
    results: list[GenerationResult] = []
    for ir in item_results:
        out = getattr(ir, "output", None) or {}
        if not out.get("evaluable"):  # unanswerable·보류 제외
            continue
        results.append(
            GenerationResult(
                query=out["query"],
                answer_text=out["answer_text"],
                retrieved_contexts=out["retrieved_contexts"],
                reference=out.get("reference"),
            )
        )
    if not results:
        return []
    scores = await score_generation(results)
    evals = []
    if scores.faithfulness is not None:
        evals.append(Evaluation(name="faithfulness", value=scores.faithfulness))
    if scores.answer_relevancy is not None:
        evals.append(Evaluation(name="answer_relevancy", value=scores.answer_relevancy))
    return evals


def run(task=_task, run_evaluator=_ragas_run_evaluator, dataset_name: str = DATASET_NAME) -> int:
    if not ragas_available():
        print("ragas 미설치 — .[eval] extra 필요 (별도 eval 이미지로 실행)", file=sys.stderr)
        return 1
    client = get_langfuse_client()
    if client is None:
        print("Langfuse 비활성 — LANGFUSE_ENABLED=true + 키 필요", file=sys.stderr)
        return 1
    dataset = client.get_dataset(dataset_name)
    client.run_experiment(
        name="generation-eval",
        description="생성 품질 (RAGAS faithfulness·answer_relevancy)",
        data=dataset.items,
        task=task,
        run_evaluators=[run_evaluator],
    )
    client.flush()
    print(f"generation experiment 완료 → dataset '{dataset_name}' Run 기록")
    return 0


if __name__ == "__main__":
    sys.exit(run())
