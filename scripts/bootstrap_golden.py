"""골든셋 후보 초안 생성기 — Qdrant 색인분에서 chunk를 샘플링해 LLM으로 질문을 만든다.

산출물은 `_draft: true` 마킹된 초안이며 **반드시 팀 검수 후 `_draft`를 제거**해 확정한다.
(질문 자연스러움·관련 chunk_id 정확성 확인, paraphrase로 다양화 → 문구 베끼기 누수 방지)

기본 출력은 `*.draft.jsonl`(실 골든셋을 덮어쓰지 않음). 검수 후 queries.jsonl/qrels.jsonl로 병합한다.
의존: 라이브 Qdrant + LLM(call_llm) — 소액 비용 발생.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings  # noqa: E402
from app.db.qdrant import get_qdrant  # noqa: E402
from app.services.llm_selector import call_llm  # noqa: E402

logger = logging.getLogger(__name__)

_GEN_SYSTEM = (
    "너는 사내 지식 검색 평가셋을 만드는 도우미다. 주어진 문서 조각을 보고, "
    "그 조각이 정답 근거가 되는 자연스러운 한국어 질문 1개를 만든다. "
    "문서 문구를 그대로 베끼지 말고 사용자가 실제로 물어볼 법하게 바꿔 표현한다. "
    '반드시 JSON 하나만 반환: {"query": "..."}'
)

# 범위 밖(답변 불가) 질문 시드 — Router 차단/Answerability 보류 측정용
_UNANSWERABLE_SEEDS = [
    "이번 주 점심 메뉴 추천해줘",
    "오늘 서울 날씨 어때?",
    "다음 분기 연봉 인상률은 얼마야?",
]


async def _gen_question(content: str, model: str) -> str | None:
    try:
        raw = await call_llm(_GEN_SYSTEM, f"문서 조각:\n{content[:1500]}", model=model, json_mode=True)
        query = json.loads(raw).get("query", "").strip()
        return query or None
    except Exception:
        logger.warning("질문 생성 실패 — 건너뜀", exc_info=True)
        return None


def _sample_chunks(limit: int, per_domain: int) -> list[dict]:
    client = get_qdrant()
    settings = get_settings()
    points, _ = client.scroll(
        collection_name=settings.qdrant_collection, with_payload=True, with_vectors=False, limit=limit
    )
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for point in points:
        payload = point.payload or {}
        if payload.get("chunk_id") and payload.get("content"):
            by_domain[payload.get("domain", "manual")].append(payload)

    sampled: list[dict] = []
    for domain, payloads in by_domain.items():
        k = min(per_domain, len(payloads))
        sampled.extend(random.sample(payloads, k))
        logger.info("domain=%s: %d개 중 %d개 샘플", domain, len(payloads), k)
    return sampled


async def run(args) -> None:
    sampled = _sample_chunks(args.limit, args.per_domain)
    if not sampled:
        logger.error("샘플 0건 — Qdrant 색인분이 비었는지 확인 (make up + 색인)")
        return

    queries: list[dict] = []
    qrels: list[dict] = []
    idx = 0
    for payload in sampled:
        query = await _gen_question(payload["content"], args.model)
        if not query:
            continue
        idx += 1
        qid = f"d{idx:03d}"
        queries.append(
            {"qid": qid, "query": query, "domain": payload.get("domain"), "is_answerable": True, "_draft": True}
        )
        qrels.append({"qid": qid, "relevant_chunk_ids": [payload["chunk_id"]]})

    for seed in _UNANSWERABLE_SEEDS:
        idx += 1
        qid = f"d{idx:03d}"
        queries.append({"qid": qid, "query": seed, "domain": None, "is_answerable": False, "_draft": True})
        qrels.append({"qid": qid, "relevant_chunk_ids": []})

    args.out_queries.parent.mkdir(parents=True, exist_ok=True)
    args.out_queries.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in queries) + "\n", encoding="utf-8")
    args.out_qrels.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in qrels) + "\n", encoding="utf-8")
    logger.info("초안 %d개 생성 → %s / %s", len(queries), args.out_queries, args.out_qrels)
    logger.info("⚠ 팀 검수 후 _draft 제거하고 queries.jsonl/qrels.jsonl로 병합하세요.")


def main() -> None:
    parser = argparse.ArgumentParser(description="골든셋 후보 초안 생성 (팀 검수용).")
    parser.add_argument("--limit", type=int, default=500, help="Qdrant scroll 상한")
    parser.add_argument("--per-domain", type=int, default=8, help="도메인별 샘플 수")
    parser.add_argument("--model", default="", help="질문 생성 LLM (빈값=config 기본)")
    parser.add_argument("--out-queries", type=Path, default=ROOT_DIR / "data" / "eval" / "queries.draft.jsonl")
    parser.add_argument("--out-qrels", type=Path, default=ROOT_DIR / "data" / "eval" / "qrels.draft.jsonl")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
