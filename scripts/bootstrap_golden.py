"""골든셋 후보 초안 생성기 — Qdrant 색인분에서 chunk를 샘플링해 LLM으로 질문을 만든다.

산출물은 `_draft: true` 마킹된 초안이며 **반드시 팀 검수 후 `_draft`를 제거**해 확정한다.
(질문 자연스러움·관련 chunk_id 정확성 확인, paraphrase로 다양화 → 문구 베끼기 누수 방지)

샘플링 기준은 data/eval/GOLDENSET_CRITERIA.md (#206) — single은 도메인 균등이 아니라
**층화(floor)+코퍼스 비례**(domain_quota)로 뽑는다. 생성 프롬프트는 §5-A 개정 반영
(지시대명사 금지·식별자 보존·answer_span 자가검증).

모드 (#78 — 전체 코퍼스 난이도 측정용 티어):
    single       청크 1개 → 질문 1개 (qid d0xx, floor+비례 quota) + 범위 밖 unanswerable 시드.
    multi-domain 서로 다른 도메인 청크 2~3개를 종합해야 답할 수 있는 질문 (qid m0xx,
                 교차 도메인 벡터 이웃을 정답으로 묶음 → 도메인 교차 recall 측정).
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
from collections import Counter, defaultdict
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings  # noqa: E402
from app.db.qdrant import get_qdrant  # noqa: E402
from app.services.llm_selector import call_llm  # noqa: E402

logger = logging.getLogger(__name__)

# 공통 작성 규칙 — answerable 질문 생성 프롬프트에 공유 (data/eval/GOLDENSET_CRITERIA.md §5-A 공통 1·2·3)
_COMMON_RULES = (
    "규칙: (1) 질문은 원문 조각 없이 단독으로 성립해야 한다 — '이 문서/위 조각/해당 설정' 같은 "
    "지시 표현 금지. (2) 고유명사·명령어·설정 키·에러코드는 원문 그대로 쓰고, 주변 자연어 표현만 "
    "사용자가 실제로 물어볼 법하게 바꾼다. (3) 조각에 답할 실질 내용이 없으면(목차·코드뿐·마스킹값뿐) "
    "answerable_from_chunk 를 false 로, query 를 빈 문자열로 반환한다."
)

# answerable 모드 JSON 계약 — query + 근거 스팬 자가검증 (§5-A 공통 4)
_ANSWERABLE_JSON = (
    '반드시 JSON 하나만 반환: {"query": "...", "answer_span": "질문의 정답 근거가 되는 원문 문장", '
    '"answerable_from_chunk": true}. 조각만으로 답할 수 없으면 answerable_from_chunk 를 false 로.'
)

_GEN_SYSTEM = (
    "너는 사내 지식 검색 평가셋을 만드는 도우미다. 주어진 문서 조각을 보고, "
    "그 조각**만으로** 답이 완성되는 자연스러운 한국어 질문 1개를 만든다. "
    "예/아니오로 끝나는 질문은 피하고 정보를 구하는 질문으로 만든다. "
    f"{_COMMON_RULES} {_ANSWERABLE_JSON}"
)

_GEN_SYSTEM_MULTI_HOP = (
    "너는 사내 지식 검색 평가셋을 만드는 도우미다. 같은 문서의 연속된 조각 여러 개를 보고, "
    "그 조각들의 정보를 **모두 종합해야** 답할 수 있는 자연스러운 한국어 질문 1개를 만든다. "
    "조각1의 사실과 조각2의 사실을 각각 써야만 답이 완성돼야 하고, 어느 한 조각만으로 답되면 실패다. "
    "단 '그리고~또~'식 복합질문이 아니라 자연스러운 한 문장으로 묻는다. "
    f"{_COMMON_RULES} {_ANSWERABLE_JSON}"
)

_GEN_SYSTEM_MULTI_DOMAIN = (
    "너는 사내 지식 검색 평가셋을 만드는 도우미다. **서로 다른 영역**의 문서 조각 여러 개를 보고, "
    "그 조각들을 **모두 종합해야** 답할 수 있는 자연스러운 한국어 질문 1개를 만든다. "
    "장애 대응·온보딩처럼 근거가 여러 영역에 흩어진 실제 질문을 모사한다. "
    "어느 한 조각(한 영역)만으로 답되면 실패다. "
    f"{_COMMON_RULES} {_ANSWERABLE_JSON}"
)

_GEN_SYSTEM_NEAR_MISS = (
    "너는 사내 지식 검색 평가셋을 만드는 도우미다. 주어진 문서 조각과 **같은 주제**지만, "
    "이 코퍼스가 **구조적으로 담지 않는** 세부사항을 묻는 자연스러운 한국어 질문 1개를 만든다. "
    "특히 가격·라이선스 비용·SLA 수치·담당자 실명·내부 일정처럼 기술 문서가 보통 다루지 않는 "
    "정보 유형을 노린다. 질문은 사내 기술 문서에 있을 법하게 들리되, 이 코퍼스에는 답이 없어야 한다. "
    "질문은 원문 조각 없이 단독 성립해야 하며 지시 표현('이 문서/위 조각') 금지, "
    "고유명사·설정 키는 원문 그대로 쓴다. "
    '반드시 JSON 하나만 반환: {"query": "..."}'
)

_GEN_SYSTEM_CONFUSABLE = (
    "너는 사내 지식 검색 평가셋을 만드는 도우미다. [타깃 문서]와 [유사 문서들]을 보고, "
    "[타깃 문서]에 **실제로 적힌 내용만 근거로**, [유사 문서들]에는 없는 식별 가능한 디테일을 "
    "묻는 자연스러운 한국어 질문 1개를 만든다. "
    "[유사 문서들]로도 답할 수 있는 일반적인 질문은 금지 — 검색기가 유사 문서 사이에서 "
    "타깃을 정확히 골라내야만 답할 수 있어야 한다. "
    f"{_COMMON_RULES} {_ANSWERABLE_JSON}"
)

# 범위 밖(답변 불가) 질문 시드 — Router 차단/Answerability 보류 측정용 (§5-A 6: 풀 확장 + 그럴듯한 도메인 밖)
_UNANSWERABLE_SEEDS = [
    # 명백한 범위 밖 (생활/사적)
    "이번 주 점심 메뉴 추천해줘",
    "오늘 서울 날씨 어때?",
    "다음 분기 연봉 인상률은 얼마야?",
    "주말에 가볼 만한 여행지 알려줘",
    "회사 근처 맛집 추천해줘",
    # 그럴듯하지만 코퍼스(Apache/Datadog/사내 운영문서) 밖 — 라우터 차단 난이도 상향
    "AWS Lambda 요금제 어떻게 돼?",
    "GCP BigQuery 파티션 설정 방법 알려줘",
    "Salesforce 영업 파이프라인 리포트 만드는 법",
    "iOS 앱 App Store 심사 거절 사유 확인하는 법",
    "Figma에서 컴포넌트 variant 만드는 방법",
]

# qid 접두사 — data/eval/README.md 의 티어 관례
_QID_PREFIX = {"single": "d", "multi-domain": "m", "multi-hop": "h", "near-miss": "n", "confusable": "c"}


def _point_id(chunk_id: str) -> str:
    """indexer._point_id 와 동일 — chunk_id → Qdrant point UUID5 (멱등)."""
    return str(uuid5(NAMESPACE_URL, chunk_id))


async def _gen(system: str, user: str, model: str) -> dict | None:
    """LLM 호출 → 파싱된 JSON dict 반환 (실패 시 None)."""
    try:
        raw = await call_llm(system, user, model=model, json_mode=True)
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        logger.warning("질문 생성 실패 — 건너뜀", exc_info=True)
        return None


def _query_only(parsed: dict | None) -> str | None:
    """near-miss 등 단순 계약 — query 문자열만 추출."""
    if not parsed:
        return None
    return (parsed.get("query") or "").strip() or None


def _answerable_query(parsed: dict | None) -> tuple[str | None, str]:
    """answerable 모드 — 자가검증(answerable_from_chunk=false)이면 폐기 (§5-A 공통 4).

    (query, answer_span) 반환. answer_span = 정답 근거 문장 — splitter 무관 평가 기준(§1,
    청킹 방식 비교 시 chunk_id 대신 "검색 청크가 이 문장을 담았나"로 채점).
    """
    if not parsed or parsed.get("answerable_from_chunk") is False:
        return None, ""
    query = (parsed.get("query") or "").strip() or None
    answer_span = (parsed.get("answer_span") or "").strip()
    return query, answer_span


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
    """도메인별 균등 샘플 (순수 함수). multi-hop/near-miss/confusable 후보 풀 구성용."""
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for payload in payloads:
        by_domain[payload.get("domain", "manual")].append(payload)
    sampled: list[dict] = []
    for domain, items in sorted(by_domain.items()):
        k = min(per_domain, len(items))
        sampled.extend(random.sample(items, k))
        logger.info("domain=%s: %d개 중 %d개 샘플", domain, len(items), k)
    return sampled


def domain_quota(domain_counts: dict[str, int], total: int, floor: int) -> dict[str, int]:
    """층화(floor)+코퍼스 비례 배분 (순수 함수, GOLDENSET_CRITERIA.md §2).

    quota(d) = floor + 비례배분(잔여 = total - floor*N). 잔여는 코퍼스 비중대로
    나누되 최대잉여법(largest remainder)으로 합이 정확히 total이 되게 한다(total>=floor*N일 때).
    """
    domains = sorted(domain_counts)
    if not domains:
        return {}
    base = floor * len(domains)
    remaining = max(0, total - base)
    corpus_total = sum(domain_counts.values()) or 1
    raw = {d: remaining * domain_counts[d] / corpus_total for d in domains}
    alloc = {d: int(raw[d]) for d in domains}
    leftover = remaining - sum(alloc.values())
    # 잔여 정수분을 소수부 큰 도메인부터 1씩 — 동률은 도메인명 순으로 결정론적
    for d in sorted(domains, key=lambda x: (raw[x] - alloc[x], x), reverse=True)[:leftover]:
        alloc[d] += 1
    return {d: floor + alloc[d] for d in domains}


def sample_by_quota(payloads: list[dict], quota: dict[str, int]) -> list[dict]:
    """도메인별 quota 만큼 샘플 (순수 함수). 부족하면 가용분만큼만."""
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for payload in payloads:
        by_domain[payload.get("domain", "manual")].append(payload)
    sampled: list[dict] = []
    for domain in sorted(quota):
        items = by_domain.get(domain, [])
        k = min(quota[domain], len(items))
        if k < quota[domain]:
            logger.warning("domain=%s: quota %d개 중 %d개만 가용", domain, quota[domain], k)
        sampled.extend(random.sample(items, k))
        logger.info("domain=%s: %d개 중 %d개 샘플 (quota=%d)", domain, len(items), k, quota[domain])
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


def _neighbor_payloads(
    target: dict, *, limit: int = 4, min_score: float = 0.5, cross_domain: bool = False
) -> list[dict]:
    """타깃 청크의 벡터 이웃 중 **다른 페이지** 청크 payload.

    confusable 재료(기본) 또는 cross_domain=True 시 **다른 도메인** 이웃만 (multi-domain 재료).
    """
    client = get_qdrant()
    settings = get_settings()
    result = client.query_points(
        collection_name=settings.qdrant_collection,
        query=_point_id(target["chunk_id"]),  # point id → 추천(유사) 검색
        limit=limit * 4,
        with_payload=True,
    )
    neighbors = []
    for p in result.points:
        payload = p.payload or {}
        if p.score < min_score or payload.get("page_id") == target.get("page_id") or not payload.get("content"):
            continue
        if cross_domain and payload.get("domain") == target.get("domain"):  # 같은 도메인 → 스킵
            continue
        neighbors.append(payload)
        if len(neighbors) >= limit:
            break
    return neighbors


def _record(
    qid: str,
    query: str,
    domain: str | None,
    *,
    answerable: bool,
    chunk_ids: list[str],
    page_ids: list[str] | None = None,
    source_urls: list[str] | None = None,
    answer_span: str = "",
) -> tuple[dict, dict]:
    # page_ids·source_urls·answer_span = splitter 무관 정답 기준(§1). 청킹 방식 비교(Token/MD/Recursive vs
    # OnRamp)는 chunk_id가 scheme마다 달라 못 쓰므로 page/doc(URL)/span으로 채점한다. chunk_id는 OnRamp 내부용.
    # page_id가 곧 doc 식별자(1 page = 1 source_document)이고, source_url은 원문 추적·교차확인용.
    q: dict[str, object] = {"qid": qid, "query": query, "domain": domain, "is_answerable": answerable, "_draft": True}
    if page_ids:
        q["page_ids"] = [p for p in page_ids if p]
    if source_urls:
        q["source_urls"] = [u for u in source_urls if u]
    if answer_span:
        q["answer_span"] = answer_span
    return q, {"qid": qid, "relevant_chunk_ids": chunk_ids}


async def _build_single(sampled: list[dict], model: str, start: int, *, scope_out: int) -> list[tuple[dict, dict]]:
    out = []
    idx = start
    for payload in sampled:
        query, span = _answerable_query(await _gen(_GEN_SYSTEM, f"문서 조각:\n{payload['content'][:1500]}", model))
        if not query:
            continue
        idx += 1
        out.append(
            _record(
                f"d{idx:03d}",
                query,
                payload.get("domain"),
                answerable=True,
                chunk_ids=[payload["chunk_id"]],
                page_ids=[payload.get("page_id", "")],
                source_urls=[payload.get("source_url", "")],
                answer_span=span,
            )
        )
    for seed in _UNANSWERABLE_SEEDS[:scope_out]:  # 범위 밖 unanswerable (scope-out)
        idx += 1
        out.append(_record(f"d{idx:03d}", seed, None, answerable=False, chunk_ids=[]))
    return out


async def _build_multi_hop(groups: list[list[dict]], model: str, start: int) -> list[tuple[dict, dict]]:
    out = []
    idx = start
    for group in groups:
        parts = "\n\n".join(f"[조각 {i + 1}]\n{c['content'][:900]}" for i, c in enumerate(group))
        query, span = _answerable_query(await _gen(_GEN_SYSTEM_MULTI_HOP, f"같은 문서의 연속 조각들:\n{parts}", model))
        if not query:
            continue
        idx += 1
        out.append(
            _record(
                f"h{idx:03d}",
                query,
                group[0].get("domain"),
                answerable=True,
                chunk_ids=[c["chunk_id"] for c in group],
                page_ids=list(dict.fromkeys(c.get("page_id", "") for c in group)),
                source_urls=list(dict.fromkeys(c.get("source_url", "") for c in group)),
                answer_span=span,
            )
        )
    return out


async def _build_multi_domain(sampled: list[dict], model: str, start: int, count: int) -> list[tuple[dict, dict]]:
    """서로 다른 도메인 청크 2~3개를 정답으로 묶는 질문 (m0xx, 도메인 교차 recall 측정)."""
    out: list[tuple[dict, dict]] = []
    idx = start
    for payload in sampled:
        if len(out) >= count:
            break
        try:
            neighbors = _neighbor_payloads(payload, limit=2, cross_domain=True)
        except Exception:
            logger.warning("교차도메인 이웃 조회 실패 — 건너뜀 (chunk_id=%s)", payload.get("chunk_id"), exc_info=True)
            continue
        if not neighbors:  # 교차 도메인 이웃 없음 → 스킵
            continue
        group = [payload, *neighbors]
        parts = "\n\n".join(
            f"[조각 {i + 1}] ({c.get('domain', '')}) {c.get('page_title', '')}\n{c['content'][:800]}"
            for i, c in enumerate(group)
        )
        query, span = _answerable_query(
            await _gen(_GEN_SYSTEM_MULTI_DOMAIN, f"서로 다른 영역의 조각들:\n{parts}", model)
        )
        if not query:
            continue
        idx += 1
        out.append(
            _record(
                f"m{idx:03d}",
                query,
                payload.get("domain"),
                answerable=True,
                chunk_ids=[c["chunk_id"] for c in group],
                page_ids=list(dict.fromkeys(c.get("page_id", "") for c in group)),
                source_urls=list(dict.fromkeys(c.get("source_url", "") for c in group)),
                answer_span=span,
            )
        )
    return out


async def _build_near_miss(sampled: list[dict], model: str, start: int) -> list[tuple[dict, dict]]:
    out = []
    idx = start
    for payload in sampled:
        query = _query_only(await _gen(_GEN_SYSTEM_NEAR_MISS, f"문서 조각:\n{payload['content'][:1500]}", model))
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
        query, span = _answerable_query(await _gen(_GEN_SYSTEM_CONFUSABLE, user, model))
        if not query:
            continue
        idx += 1
        out.append(
            _record(
                f"c{idx:03d}",
                query,
                payload.get("domain"),
                answerable=True,
                chunk_ids=[payload["chunk_id"]],
                page_ids=[payload.get("page_id", "")],
                source_urls=[payload.get("source_url", "")],
                answer_span=span,
            )
        )
    return out


async def run(args) -> None:
    payloads = _scroll_payloads(args.limit)
    if not payloads:
        logger.error("샘플 0건 — Qdrant 색인분이 비었는지 확인 (make up + 색인)")
        return
    logger.info("mode=%s — 코퍼스 청크 %d개 로드", args.mode, len(payloads))

    if args.mode == "single":
        counts = Counter(p.get("domain", "manual") for p in payloads)
        quota = domain_quota(dict(counts), args.total_answerable, args.floor)
        logger.info("single quota (floor=%d, total=%d): %s", args.floor, args.total_answerable, quota)
        records = await _build_single(
            sample_by_quota(payloads, quota), args.model, args.start_index, scope_out=args.scope_out
        )
    elif args.mode == "multi-domain":
        sampled = sample_per_domain(payloads, args.per_domain * 3)
        random.shuffle(sampled)
        records = await _build_multi_domain(sampled, args.model, args.start_index, args.count)
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
    parser.add_argument(
        "--per-domain", type=int, default=8, help="도메인별 샘플 수 (multi-domain/hop/near-miss/confusable 후보 풀)"
    )
    parser.add_argument("--total-answerable", type=int, default=50, help="single 모드 answerable 목표 수 (§2)")
    parser.add_argument("--floor", type=int, default=8, help="single 모드 도메인별 최소 보장 문항 (§2)")
    parser.add_argument(
        "--scope-out", type=int, default=10, help="single 모드 범위 밖 unanswerable 수 (시드 풀 상한 내)"
    )
    parser.add_argument("--count", type=int, default=12, help="multi-domain/hop/confusable 목표 문항 수")
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
