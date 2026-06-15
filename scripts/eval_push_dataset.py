"""골든셋(queries.jsonl+qrels.jsonl)을 Langfuse Dataset으로 업로드한다 (#139, Epic #120 E6).

Dataset Run으로 모델/프롬프트 버전 간 검색·생성 품질을 비교할 기반.
qid를 item id로 써서 **멱등 업서트**(재실행해도 중복 안 생김).

사용:
    LANGFUSE_ENABLED=true LANGFUSE_HOST=... LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=... \\
        python scripts/eval_push_dataset.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.eval.dataset import load_golden_set  # noqa: E402
from app.observability import get_langfuse_client  # noqa: E402

DATASET_NAME = "onramp-retrieval-golden"


def push(queries: Path, qrels: Path, dataset_name: str = DATASET_NAME) -> int:
    """골든셋을 Langfuse Dataset에 업로드한다. 성공 0, 비활성/실패 1."""
    client = get_langfuse_client()
    if client is None:
        print(
            "Langfuse 비활성 — LANGFUSE_ENABLED=true + LANGFUSE_HOST/PUBLIC_KEY/SECRET_KEY 필요",
            file=sys.stderr,
        )
        return 1

    golden = load_golden_set(queries, qrels)
    client.create_dataset(
        name=dataset_name,
        description="OnRamp 검색/생성 평가 골든셋 (data/eval)",
        metadata={"source": "data/eval"},
    )
    for g in golden:
        client.create_dataset_item(
            dataset_name=dataset_name,
            id=g.qid,  # qid 멱등 업서트
            input={"query": g.query, "domain": g.domain},
            expected_output={
                "relevant_chunk_ids": list(g.relevant_chunk_ids),
                "is_answerable": g.is_answerable,
                "ground_truth_answer": g.ground_truth_answer,
                "gold_domains": list(g.gold_domains),
            },
            metadata={"router_domains": list(g.router_domains), "is_draft": g.is_draft},
        )
    client.flush()
    print(f"업로드 완료: {len(golden)}건 → Langfuse dataset '{dataset_name}'")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="골든셋 → Langfuse Dataset 업로드 (E6).")
    parser.add_argument("--queries", type=Path, default=ROOT_DIR / "data" / "eval" / "queries.jsonl")
    parser.add_argument("--qrels", type=Path, default=ROOT_DIR / "data" / "eval" / "qrels.jsonl")
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    args = parser.parse_args()
    sys.exit(push(args.queries, args.qrels, args.dataset_name))


if __name__ == "__main__":
    main()
