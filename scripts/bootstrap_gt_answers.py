"""GT 답변(ground_truth_answer) 후보 초안 생성기 — #67.

answerable 골든 질문마다, 그 질문의 정답 chunk(relevant_chunk_ids) 내용을 근거로
LLM이 간결한 참조 답변을 만든다. 산출물은 `_draft: true` 마킹된 초안이며
**반드시 팀 검수 후** queries.jsonl의 `ground_truth_answer`로 병합한다(문구 베끼기 누수 방지 위해 paraphrase).

출력: `data/eval/gt_answers.draft.jsonl` (*.draft.jsonl → gitignore). 실 골든셋을 덮어쓰지 않는다.
의존: 라이브 Qdrant + LLM(call_llm) — 소액 비용 발생.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings  # noqa: E402
from app.db.qdrant import get_qdrant  # noqa: E402
from app.eval.dataset import load_golden_set  # noqa: E402
from app.services.llm_selector import call_llm  # noqa: E402

logger = logging.getLogger(__name__)

_GEN_SYSTEM = (
    "너는 사내 지식 평가셋의 모범답안을 만드는 도우미다. 질문과 정답 근거 문서들을 보고, "
    "오직 근거에 있는 사실만으로 간결하고 정확한 한국어 모범답안을 작성한다. "
    "근거에 없는 내용은 추측하지 말고, 핵심만 3~5문장으로 요약한다. "
    "근거에 [MASKED_...] 같은 마스킹된 비밀값이 있으면 그 토큰을 그대로 쓰지 말고 "
    "일반적인 설명(예: '해당 키링 파일', 'signed-by 옵션')으로 대체한다. "
    '반드시 JSON 하나만 반환: {"ground_truth_answer": "..."}'
)


def build_content_map(points) -> dict[str, str]:
    """Qdrant 포인트들에서 chunk_id → content 매핑을 만든다 (순수 함수)."""
    content_map: dict[str, str] = {}
    for point in points:
        payload = getattr(point, "payload", None) or {}
        chunk_id = payload.get("chunk_id")
        content = payload.get("content")
        if chunk_id and content:
            content_map[chunk_id] = content
    return content_map


def collect_contexts(chunk_ids, content_map: dict[str, str]) -> list[str]:
    """정답 chunk_id들의 content를 모은다 (없는 id는 건너뜀, 순수 함수)."""
    return [content_map[cid] for cid in chunk_ids if cid in content_map]


def _scroll_all(limit: int) -> list:
    """Qdrant 전 페이지를 scroll로 누적한다 (next_page_offset 따라감)."""
    client = get_qdrant()
    settings = get_settings()
    all_points: list = []
    offset = None
    remaining = max(0, limit)
    while remaining > 0:
        batch, offset = client.scroll(
            collection_name=settings.qdrant_collection,
            with_payload=True,
            with_vectors=False,
            limit=remaining,
            offset=offset,
        )
        if not batch:
            break
        all_points.extend(batch)
        remaining -= len(batch)
        if offset is None:  # 마지막 페이지
            break
    return all_points


async def _gen_answer(query: str, contexts: list[str], model: str) -> str | None:
    joined = "\n\n".join(f"[근거 {i + 1}]\n{c[:1200]}" for i, c in enumerate(contexts))
    try:
        raw = await call_llm(_GEN_SYSTEM, f"질문: {query}\n\n{joined}", model=model, json_mode=True)
        answer = json.loads(raw).get("ground_truth_answer", "").strip()
        return answer or None
    except Exception:
        logger.warning("GT 답변 생성 실패 — 건너뜀 (query=%.40s)", query, exc_info=True)
        return None


async def run(args) -> None:
    golden = load_golden_set(args.queries, args.qrels)
    answerable = [g for g in golden if g.is_answerable and g.relevant_chunk_ids]
    if args.limit is not None:
        answerable = answerable[: max(0, args.limit)]

    content_map = build_content_map(_scroll_all(args.scroll_limit))
    if not content_map:
        logger.error("Qdrant 색인분이 비었습니다 (make up + 색인 필요)")
        return
    logger.info("골든 answerable %d개, 코퍼스 chunk %d개 로드", len(answerable), len(content_map))

    drafts: list[dict] = []
    for i, g in enumerate(answerable, start=1):
        contexts = collect_contexts(g.relevant_chunk_ids, content_map)
        if not contexts:
            logger.warning("[%d] qid=%s 정답 chunk 내용 0건 — 건너뜀", i, g.qid)
            continue
        logger.info("[%d/%d] GT 생성: qid=%s %.40s", i, len(answerable), g.qid, g.query)
        answer = await _gen_answer(g.query, contexts, args.model)
        if answer:
            drafts.append({"qid": g.qid, "ground_truth_answer": answer, "_draft": True})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in drafts) + "\n", encoding="utf-8")
    logger.info("GT 초안 %d개 생성 → %s", len(drafts), args.out)
    logger.info("⚠ 팀 검수 후 paraphrase하여 queries.jsonl의 ground_truth_answer로 병합하세요.")


def main() -> None:
    parser = argparse.ArgumentParser(description="GT 답변 후보 초안 생성 (#67, 팀 검수용).")
    parser.add_argument("--queries", type=Path, default=ROOT_DIR / "data" / "eval" / "queries.jsonl")
    parser.add_argument("--qrels", type=Path, default=ROOT_DIR / "data" / "eval" / "qrels.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="처리할 골든 문항 수(비용 절감)")
    parser.add_argument("--scroll-limit", type=int, default=2000, help="Qdrant scroll 상한")
    parser.add_argument("--model", default="", help="GT 생성 LLM (빈값=config 기본)")
    parser.add_argument("--out", type=Path, default=ROOT_DIR / "data" / "eval" / "gt_answers.draft.jsonl")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    if args.scroll_limit <= 0:
        parser.error("--scroll-limit 은 양의 정수여야 합니다")

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
