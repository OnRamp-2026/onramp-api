"""검색 평가를 Langfuse Dataset Run(Experiment)으로 실행 (Epic #120 — 검색 품질 가시화).

골든셋(onramp-retrieval-golden) 위에서 우리 검색 경로(rerank)를 돌리고,
Hit Rate@k·MRR·Recall·nDCG를 **Langfuse Run 점수**로 기록한다 → 대시보드·Run 비교에서
검색 품질을 본다. unanswerable(정답셋 빈셋) 질문은 검색 지표에서 제외.

필요 env: LANGFUSE_*(활성) + OPENAI_API_KEY + Qdrant 접근. (nightly CronJob으로 실행)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from langfuse import Evaluation  # noqa: E402

from app.eval import metrics  # noqa: E402
from app.eval.retrieval_adapter import retrieve_for_eval  # noqa: E402
from app.observability import get_langfuse_client  # noqa: E402

DATASET_NAME = "onramp-retrieval-golden"
K = 5


async def _task(*, item, **kwargs):
    """데이터셋 item(query) → 우리 검색 경로의 ranked chunk_id 리스트."""
    query = (item.input or {}).get("query", "")
    result = await retrieve_for_eval(query, mode="rerank")
    return result.chunk_ids


def _relevant(expected_output) -> set[str]:
    return set((expected_output or {}).get("relevant_chunk_ids") or [])


def _make_evaluator(name, fn):
    def _ev(*, input, output, expected_output, metadata=None, **kwargs):
        relevant = _relevant(expected_output)
        if not relevant:  # unanswerable → 검색 지표 집계 제외
            return []
        return Evaluation(name=name, value=float(fn(output or [], relevant, K)))

    return _ev


EVALUATORS = [
    _make_evaluator(f"hit_rate@{K}", metrics.hit_rate_at_k),
    _make_evaluator(f"mrr@{K}", metrics.reciprocal_rank),
    _make_evaluator(f"recall@{K}", metrics.recall_at_k),
    _make_evaluator(f"ndcg@{K}", metrics.ndcg_at_k),
]


def run(task=_task, dataset_name: str = DATASET_NAME) -> int:
    client = get_langfuse_client()
    if client is None:
        print("Langfuse 비활성 — LANGFUSE_ENABLED=true + 키 필요", file=sys.stderr)
        return 1
    dataset = client.get_dataset(dataset_name)
    client.run_experiment(
        name="retrieval-eval",
        description="검색 품질 (Hit Rate@5·MRR·Recall·nDCG) — rerank 모드",
        data=dataset.items,
        task=task,
        evaluators=EVALUATORS,
    )
    client.flush()
    print(f"retrieval experiment 완료 → dataset '{dataset_name}' Run 기록")
    return 0


if __name__ == "__main__":
    sys.exit(run())
