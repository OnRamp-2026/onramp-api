"""골든셋 후보 초안 생성기 — Qdrant 색인분에서 chunk를 샘플링해 LLM으로 질문을 만든다.

산출물은 `_draft: true` 마킹된 초안이며 **반드시 팀 검수 후 `_draft`를 제거**해 확정한다.
(질문 자연스러움·관련 chunk_id 정확성 확인, paraphrase로 다양화 → 문구 베끼기 누수 방지)

모드 (#78 — 전체 코퍼스 난이도 측정용 티어):
    single     기존 동작. 청크 1개 → 질문 1개 (qid d0xx) + 범위 밖 unanswerable 시드.
    multi-hop  같은 페이지의 인접 청크 2~3개를 종합해야 답할 수 있는 질문 (qid h0xx,
               멀티청크 qrels → Recall@k가 Hit Rate와 분리되는 진짜 재현율 측정).
    near-miss  도메인 안 주제지만 코퍼스가 답하지 않는 질문 (qid n0xx, unanswerable).
               '점심 메뉴'류 범위 밖보다 answerability 변별력이 훨씬 높다.
    confusable 벡터 이웃(다른 페이지 유사 청크)이 많은 타깃 청크를 골라, 타깃에만 있는
               정보를 묻는 질문 (qid c0xx) — 유사 문서 군집 속 정밀 변별 측정.

기본 출력은 `*.draft.jsonl`(실 골든셋을 덮어쓰지 않음, gitignore). 검수 후 병합한다.
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
from uuid import NAMESPACE_URL, uuid5

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

_GEN_SYSTEM_MULTI_HOP = (
    "너는 사내 지식 검색 평가셋을 만드는 도우미다. 같은 문서의 연속된 조각 여러 개를 보고, "
    "그 조각들의 정보를 **모두 종합해야** 답할 수 있는 자연스러운 한국어 질문 1개를 만든다. "
    "조각 하나만으로 답이 끝나는 질문은 금지. 문서 문구를 그대로 베끼지 말 것. "
    '반드시 JSON 하나만 반환: {"query": "..."}'
)

_GEN_SYSTEM_NEAR_MISS = (
    "너는 사내 지식 검색 평가셋을 만드는 도우미다. 주어진 문서 조각과 **같은 주제**지만, "
    "이 조각(및 일반적인 같은 문서)이 답하지 **않는** 인접 세부사항을 묻는 자연스러운 한국어 질문 "
    "1개를 만든다. 예: 문서가 설치 절차라면 '라이선스 비용'처럼 문서가 다루지 않을 내용. "
    "질문은 사내 기술 문서에 있을 법하게 들리되, 이 코퍼스에는 답이 없어야 한다. "
    '반드시 JSON 하나만 반환: {"query": "..."}'
)

_GEN_SYSTEM_CONFUSABLE = (
    "너는 사내 지식 검색 평가셋을 만드는 도우미다. [타깃 문서]와 [유사 문서들]을 보고, "
    "[타깃 문서]에만 있는 정보를 묻는 자연스러운 한국어 질문 1개를 만든다. "
    "[유사 문서들]로도 답할 수 있는 일반적인 질문은 금지 — 검색기가 유사 문서 사이에서 "
    "타깃을 정확히 골라내야만 답할 수 있어야 한다. 문서 문구를 그대로 베끼지 말 것. "
    '반드시 JSON 하나만 반환: {"query": "..."}'
)

# 범위 밖(답변 불가) 질문 시드 — Router 차단/Answerability 보류 측정용
_UNANSWERABLE_SEEDS = [
    "이번 주 점심 메뉴 추천해줘",
    "오늘 서울 날씨 어때?",
    "다음 분기 연봉 인상률은 얼마야?",
]

# qid 접두사 — data/eval/README.md 의 티어 관례
_QID_PREFIX = {"single": "d", "multi-hop": "h", "near-miss": "n", "confusable": "c"}


def _point_id(chunk_id: str) -> str:
    """indexer._point_id 와 동일 — chunk_id → Qdrant point UUID5 (멱등)."""
    return str(uuid5(NAMESPACE_URL, chunk_id))


async def _gen(system: str, user: str, model: str) -> str | None:
    try:
        raw = await call_llm(system, user, model=model, json_mode=True)
        query = json.loads(raw).get("query", "").strip()
        return query or None
    except Exception:
        logger.warning("질문 생성 실패 — 건너뜀", exc_info=True)
        return None


def _scroll_payloads(limit: int) -> list[dict]:
    """Qdrant에서 content 있는 청크 payload를 수집한다 (커서 페이지네이션)."""
    client = get_qdrant()
    settings = get_settings()
    payloads: list[dict] = []
    offset = None
    while len(payloads) < limit:
        points, offset = client.scroll(
            collection_name=settings.qdrant_collection,
            with_payload=True,
            with_vectors=False,
            limit=min(1000, limit - len(payloads)),
            offset=offset,
        )
        payloads.extend(
            p.payload for p in points if p.payload and p.payload.get("chunk_id") and p.payload.get("content")
        )
        if offset is None:
            break
    return payloads


def sample_per_domain(payloads: list[dict], per_domain: int) -> list[dict]:
    """도메인별 균등 샘플 (순수 함수)."""
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for payload in payloads:
        by_domain[payload.get("domain", "manual")].append(payload)
    sampled: list[dict] = []
    for domain, items in sorted(by_domain.items()):
        k = min(per_domain, len(items))
        sampled.extend(random.sample(items, k))
        logger.info("domain=%s: %d개 중 %d개 샘플", domain, len(items), k)
    return sampled


def group_adjacent_chunks(payloads: list[dict], *, span: int = 2, max_groups_per_page: int = 1) -> list[list[dict]]:
    """같은 페이지의 chunk_index 연속 청크를 span개씩 묶는다 (multi-hop 재료, 순수 함수).

    span 미만 청크 페이지는 제외. 페이지당 max_groups_per_page 그룹까지.
    """
    by_page: dict[str, list[dict]] = defaultdict(list)
    for payload in payloads:
        by_page[payload.get("page_id", "")].append(payload)
    groups: list[list[dict]] = []
    for _, chunks in sorted(by_page.items()):
        chunks = sorted(chunks, key=lambda c: c.get("chunk_index", 0))
        page_groups = 0
        for i in range(0, len(chunks) - span + 1, span):
            window = chunks[i : i + span]
            indexes = [c.get("chunk_index", -1) for c in window]
            if indexes != list(range(indexes[0], indexes[0] + span)):  # 비연속 → 스킵
                continue
            groups.append(window)
            page_groups += 1
            if page_groups >= max_groups_per_page:
                break
    return groups


def _neighbor_payloads(target: dict, *, limit: int = 4, min_score: float = 0.5) -> list[dict]:
    """타깃 청크의 벡터 이웃 중 **다른 페이지** 청크 payload (confusable 재료)."""
    client = get_qdrant()
    settings = get_settings()
    result = client.query_points(
        collection_name=settings.qdrant_collection,
        query=_point_id(target["chunk_id"]),  # point id → 추천(유사) 검색
        limit=limit * 3,
        with_payload=True,
    )
    neighbors = []
    for p in result.points:
        payload = p.payload or {}
        if p.score < min_score or payload.get("page_id") == target.get("page_id") or not payload.get("content"):
            continue
        neighbors.append(payload)
        if len(neighbors) >= limit:
            break
    return neighbors


def _record(qid: str, query: str, domain: str | None, *, answerable: bool, chunk_ids: list[str]) -> tuple[dict, dict]:
    q = {"qid": qid, "query": query, "domain": domain, "is_answerable": answerable, "_draft": True}
    return q, {"qid": qid, "relevant_chunk_ids": chunk_ids}


async def _build_single(sampled: list[dict], model: str, start: int) -> list[tuple[dict, dict]]:
    out = []
    idx = start
    for payload in sampled:
        query = await _gen(_GEN_SYSTEM, f"문서 조각:\n{payload['content'][:1500]}", model)
        if not query:
            continue
        idx += 1
        out.append(
            _record(f"d{idx:03d}", query, payload.get("domain"), answerable=True, chunk_ids=[payload["chunk_id"]])
        )
    for seed in _UNANSWERABLE_SEEDS:
        idx += 1
        out.append(_record(f"d{idx:03d}", seed, None, answerable=False, chunk_ids=[]))
    return out


async def _build_multi_hop(groups: list[list[dict]], model: str, start: int) -> list[tuple[dict, dict]]:
    out = []
    idx = start
    for group in groups:
        parts = "\n\n".join(f"[조각 {i + 1}]\n{c['content'][:900]}" for i, c in enumerate(group))
        query = await _gen(_GEN_SYSTEM_MULTI_HOP, f"같은 문서의 연속 조각들:\n{parts}", model)
        if not query:
            continue
        idx += 1
        out.append(
            _record(
                f"h{idx:03d}", query, group[0].get("domain"), answerable=True, chunk_ids=[c["chunk_id"] for c in group]
            )
        )
    return out


async def _build_near_miss(sampled: list[dict], model: str, start: int) -> list[tuple[dict, dict]]:
    out = []
    idx = start
    for payload in sampled:
        query = await _gen(_GEN_SYSTEM_NEAR_MISS, f"문서 조각:\n{payload['content'][:1500]}", model)
        if not query:
            continue
        idx += 1
        # near-miss는 unanswerable: 정답 청크 없음. domain은 질문이 속한 영역(라우터 입력 시뮬레이션).
        out.append(_record(f"n{idx:03d}", query, payload.get("domain"), answerable=False, chunk_ids=[]))
    return out


async def _build_confusable(sampled: list[dict], model: str, start: int, min_neighbors: int) -> list[tuple[dict, dict]]:
    out = []
    idx = start
    for payload in sampled:
        try:
            neighbors = _neighbor_payloads(payload)
        except Exception:
            logger.warning("이웃 조회 실패 — 건너뜀 (chunk_id=%s)", payload.get("chunk_id"), exc_info=True)
            continue
        if len(neighbors) < min_neighbors:  # 혼동 군집이 아님 → 스킵
            continue
        sim = "\n\n".join(
            f"[유사 문서 {i + 1}] {n.get('page_title', '')}\n{n['content'][:500]}" for i, n in enumerate(neighbors)
        )
        user = f"[타깃 문서] {payload.get('page_title', '')}\n{payload['content'][:1200]}\n\n{sim}"
        query = await _gen(_GEN_SYSTEM_CONFUSABLE, user, model)
        if not query:
            continue
        idx += 1
        out.append(
            _record(f"c{idx:03d}", query, payload.get("domain"), answerable=True, chunk_ids=[payload["chunk_id"]])
        )
    return out


async def run(args) -> None:
    payloads = _scroll_payloads(args.limit)
    if not payloads:
        logger.error("샘플 0건 — Qdrant 색인분이 비었는지 확인 (make up + 색인)")
        return
    logger.info("mode=%s — 코퍼스 청크 %d개 로드", args.mode, len(payloads))

    if args.mode == "single":
        records = await _build_single(sample_per_domain(payloads, args.per_domain), args.model, args.start_index)
    elif args.mode == "multi-hop":
        groups = group_adjacent_chunks(sample_per_domain(payloads, args.per_domain * 4), span=args.span)
        random.shuffle(groups)
        records = await _build_multi_hop(groups[: args.count], args.model, args.start_index)
    elif args.mode == "near-miss":
        records = await _build_near_miss(sample_per_domain(payloads, args.per_domain), args.model, args.start_index)
    else:  # confusable
        sampled = sample_per_domain(payloads, args.per_domain * 3)
        random.shuffle(sampled)
        records = await _build_confusable(sampled[: args.count * 3], args.model, args.start_index, args.min_neighbors)
        records = records[: args.count]

    queries = [q for q, _ in records]
    qrels = [r for _, r in records]
    args.out_queries.parent.mkdir(parents=True, exist_ok=True)
    args.out_queries.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in queries) + "\n", encoding="utf-8")
    args.out_qrels.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in qrels) + "\n", encoding="utf-8")
    logger.info("초안 %d개 생성 → %s / %s", len(queries), args.out_queries, args.out_qrels)
    logger.info("⚠ 팀 검수 후 _draft 제거하고 queries.jsonl/qrels.jsonl로 병합하세요.")
    logger.info(
        "⚠ 검수 보조: python scripts/pool_candidates.py --queries %s 로 top-10 후보를 뽑아 라벨 누락을 확인하세요.",
        args.out_queries,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="골든셋 후보 초안 생성 (팀 검수용).")
    parser.add_argument("--mode", choices=sorted(_QID_PREFIX), default="single")
    parser.add_argument("--limit", type=int, default=10000, help="Qdrant scroll 상한")
    parser.add_argument("--per-domain", type=int, default=8, help="도메인별 샘플 수")
    parser.add_argument("--count", type=int, default=12, help="multi-hop/confusable 목표 문항 수")
    parser.add_argument("--span", type=int, default=2, help="multi-hop 인접 청크 수 (2~3)")
    parser.add_argument("--min-neighbors", type=int, default=2, help="confusable 최소 이웃 수")
    parser.add_argument("--start-index", type=int, default=0, help="qid 시작 번호 오프셋 (기존 골든셋과 충돌 방지)")
    parser.add_argument("--model", default="", help="질문 생성 LLM (빈값=config 기본)")
    parser.add_argument("--out-queries", type=Path, default=None)
    parser.add_argument("--out-qrels", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    if args.out_queries is None:
        args.out_queries = ROOT_DIR / "data" / "eval" / f"queries.{args.mode}.draft.jsonl"
    if args.out_qrels is None:
        args.out_qrels = ROOT_DIR / "data" / "eval" / f"qrels.{args.mode}.draft.jsonl"

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
