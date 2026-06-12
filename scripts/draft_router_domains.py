"""사람 검수용 `router_domains` 초안(검수표) 생성기.

answerable 골든 질문마다 검수 행을 만든다. 라우터 예측 캐시가 있으면 그 예측을
**제안값**(suggestion_source="router_prediction")으로 채우되, 이는 정답이 아니라
검수 보조다. 최종 정답은 사람이 `reviewed_router_domains`에 채운(approved/edited) 값만
쓴다 — 자동 제안을 곧바로 queries.jsonl 정답으로 확정하지 않는다(자기 정답화 방지).

산출물: data/eval/reviews/router_domains_review.jsonl (Git 추적 — 합성 평가질문이라 평문 OK).
우선 검수 대상: 멀티(m0xx) · confusable(c0xx).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from app.config import get_settings
from app.eval.dataset import GoldenQuery, load_golden_set
from app.eval.router_cache import DEFAULT_CACHE_PATH, current_meta, is_fresh, load_cache, sha12

# 기본 경로는 repo 루트 기준으로 해결한다(CWD 비의존 — 다른 디렉터리에서 실행해도 동작).
_ROOT = Path(__file__).resolve().parents[1]
_QUERIES = _ROOT / "data/eval/queries.jsonl"
_QRELS = _ROOT / "data/eval/qrels.jsonl"
DEFAULT_REVIEW_PATH = _ROOT / "data/eval/reviews/router_domains_review.jsonl"
_DEFAULT_CACHE = str(_ROOT / DEFAULT_CACHE_PATH)


def _priority(qid: str) -> str:
    if qid.startswith("m"):
        return "multi"
    if qid.startswith("c"):
        return "confusable"
    return "normal"


# 사람이 채우는 검수 필드 — 재실행 시 기존값을 **보존**해야 한다(덮어쓰면 검수 유실).
_REVIEW_FIELDS = ("reviewed_router_domains", "review_status", "reviewer", "reviewed_at")
_REVIEW_DEFAULTS = {"reviewed_router_domains": None, "review_status": "pending", "reviewer": None, "reviewed_at": None}


def _raw_suggestion(rec: dict | None, *, query_sha: str, meta) -> list[str] | None:
    """신선한 캐시면 **raw**(게이팅 전) 예측을 제안값으로 반환. 저신뢰로 게이팅돼 비워지기 전,
    LLM이 실제로 분류한 도메인을 검수자가 보게 한다(A/B는 게이팅 후 predicted_domains를 쓴다)."""
    if rec is not None and is_fresh(rec, query_sha=query_sha, meta=meta):
        return list(rec.get("raw_predicted_domains") or [])
    return None


def _is_priority(qid: str) -> bool:
    return _priority(qid) in ("multi", "confusable")


def _row(g: GoldenQuery, cache: dict[str, dict], existing: dict | None, *, query_sha: str, meta, blind: bool) -> dict:
    suggestion = _raw_suggestion(cache.get(g.qid), query_sha=query_sha, meta=meta)
    # 앵커링 완화: --blind면 중요 문항(multi/confusable)은 제안을 **검수표 행에 넣지 않는다**
    # (같은 행에 두면 검수자가 바로 봐서 blind 효과가 없다 → 제안은 별도 sidecar 파일로 저장).
    if suggestion is None:
        proposed, source = None, "none"
    elif blind and _is_priority(g.qid):
        proposed, source = None, "blinded"
    else:
        proposed, source = suggestion, "router_prediction"
    row = {
        "qid": g.qid,
        "query": g.query,  # 평문 — 검수표에만 둔다(캐시엔 미저장)
        "query_sha": query_sha,  # 검수 staleness 판정용 — 질문이 바뀌면 옛 검수를 무효화
        "domain": g.domain,
        "gold_domains": list(g.gold_domains),
        "suggestion_source": source,  # 제안 출처(매 실행 갱신) — 자동 제안일 뿐 정답 아님
        "proposed_router_domains": proposed,  # 모델 raw 제안(매 실행 갱신, blind면 미노출)
        "priority": _priority(g.qid),
    }
    # 사람 검수 필드 보존: **query_sha가 일치할 때만**. 질문 문구가 바뀌면 옛 라벨이
    # 새 질문에 안 맞으므로 pending으로 초기화해 재검수를 강제한다(qid만 보고 보존하면
    # 질문이 바뀌어도 옛 approved가 따라붙는 stale 검수 버그).
    keep = bool(existing) and existing.get("query_sha") == query_sha
    for field in _REVIEW_FIELDS:
        row[field] = existing.get(field, _REVIEW_DEFAULTS[field]) if keep else _REVIEW_DEFAULTS[field]
    return row


def _load_existing(path: Path) -> dict[str, dict]:
    """기존 검수표를 qid→행으로 로드(없으면 빈 dict). 사람 검수 결과 보존에 쓴다."""
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line:
            row = json.loads(line)
            out[row["qid"]] = row
    return out


def _write_atomic(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="router_domains 사람 검수표 초안 생성")
    parser.add_argument("--cache", default=_DEFAULT_CACHE, help="라우터 예측 캐시 경로(제안값 출처)")
    parser.add_argument("--out", default=str(DEFAULT_REVIEW_PATH), help="검수표 출력 경로")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="예측 제안이 없어도 빈 초안(proposed=null)을 강제로 쓴다(권장 안 함)",
    )
    parser.add_argument(
        "--blind",
        action="store_true",
        help="앵커링 완화: 중요 문항(multi/confusable)의 모델 제안을 가려 독립 라벨링 유도(가린 값은 별도 *.blinded.jsonl에 저장)",
    )
    args = parser.parse_args()

    golden = load_golden_set(_QUERIES, _QRELS)
    cache = load_cache(args.cache)
    out_path = Path(args.out)
    existing = _load_existing(out_path)  # 기존 검수 결과 보존용
    meta = current_meta("", get_settings())

    answerable = [g for g in golden if g.is_answerable]
    order = {"multi": 0, "confusable": 1, "normal": 2}  # 우선순위 후 qid 정렬
    answerable.sort(key=lambda g: (order[_priority(g.qid)], g.qid))

    rows = [
        _row(g, cache, existing.get(g.qid), query_sha=sha12(g.query), meta=meta, blind=args.blind) for g in answerable
    ]
    # 캐시 기반 제안 보유(blind로 가린 것도 캐시는 있는 것) — 빈 초안 거부 판정용
    with_suggestion = sum(1 for r in rows if r["suggestion_source"] in ("router_prediction", "blinded"))
    preserved = sum(1 for r in rows if r["review_status"] != "pending")
    # 질문 문구가 바뀌어 옛 검수가 무효화된(non-pending이었는데 query_sha 불일치) 건 수
    invalidated = sum(
        1
        for g in answerable
        if (e := existing.get(g.qid))
        and e.get("review_status", "pending") != "pending"
        and e.get("query_sha") != sha12(g.query)
    )

    # 검수표는 라우터 예측 캐시 **이후**에 만들어야 한다. 제안이 0건이면 검수 가치가 없는
    # null 초안을 '정상 초안'처럼 남기지 않는다 — 명확히 막고 절차를 안내한다.
    if with_suggestion == 0 and not args.allow_missing:
        print(
            "✗ 예측 제안이 0건입니다 — 검수표를 만들지 않았습니다.\n"
            f"  신선한 예측 캐시가 없습니다(경로: {args.cache}).\n"
            "  먼저 실행: python scripts/eval_router_domains.py --build-cache\n"
            "  그 뒤 이 스크립트를 다시 실행하면 예측이 proposed_router_domains로 채워집니다.\n"
            "  (정말 빈 초안이 필요하면 --allow-missing)"
        )
        raise SystemExit(1)

    _write_atomic(rows, out_path)

    # --blind: 가린 제안은 **별도 sidecar 파일**로 저장(검수표엔 미노출, 검수 후 비교용)
    if args.blind:
        sidecar = [
            {"qid": g.qid, "query": g.query, "blinded_suggestion": s}
            for g in answerable
            if _is_priority(g.qid) and (s := _raw_suggestion(cache.get(g.qid), query_sha=sha12(g.query), meta=meta))
        ]
        if sidecar:
            sidecar_path = out_path.with_suffix(".blinded.jsonl")
            _write_atomic(sidecar, sidecar_path)
            print(f"  blind 제안 {len(sidecar)}건 → {sidecar_path} (검수표엔 미노출, 검수 후 비교용)")

    by_pri = {p: sum(1 for r in rows if r["priority"] == p) for p in ("multi", "confusable", "normal")}
    print(f"검수표 {len(rows)}건 작성 → {args.out}")
    print(f"  제안값(캐시) 있음: {with_suggestion} / 없음: {len(rows) - with_suggestion}")
    print(f"  기존 검수 보존(non-pending): {preserved}")
    if invalidated:
        print(f"  ⚠️  질문 변경으로 무효화된 옛 검수(→pending 재검수 필요): {invalidated}")
    print(f"  우선순위: multi={by_pri['multi']} confusable={by_pri['confusable']} normal={by_pri['normal']}")
    print("  reviewed_router_domains를 사람이 채운(approved/edited) 행만 정답 반영")


if __name__ == "__main__":
    main()
