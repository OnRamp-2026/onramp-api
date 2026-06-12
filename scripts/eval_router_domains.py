"""멀티도메인 라우터 평가 — 예측 캐시 생성 + 지표 리포트.

라우터를 골든 질문에 1회씩 돌려 예측을 캐시(.cache/onramp-eval/router_predictions.jsonl, 오프라인 전용)에
저장하고, 캐시의 predicted_domains를 골든셋의 사람 검수 `router_domains`(정답)와 비교해
멀티도메인 지표를 출력한다. 캐시가 신선하면 LLM을 재호출하지 않는다(--report는
LLM을 전혀 부르지 않고 캐시만으로 리포트).

전제: route_node와 동일한 classify_query 사용(LLM 호출 1회, Qdrant 불필요).
사용:
    python scripts/eval_router_domains.py                 # 신선캐시 재사용 + 부족분만 예측 → 리포트
    python scripts/eval_router_domains.py --build-cache   # 예측 캐시만 생성/갱신(리포트 생략)
    python scripts/eval_router_domains.py --report        # LLM 없이 캐시만으로 리포트
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime

from app.agents.router.node import classify_query
from app.agents.state import UseCase
from app.config import get_settings
from app.eval.dataset import GoldenQuery, load_golden_set
from app.eval.router_cache import (
    DEFAULT_CACHE_PATH,
    PredictionRecord,
    current_meta,
    git_commit_sha,
    is_fresh,
    load_cache,
    sha12,
    write_cache,
)
from app.eval.router_metrics import RouterPred, summarize

_REQUESTED_MODEL = ""  # 평가는 운영 기본 모델 사용 (config.default_model)
_DEFAULT_RESULT_PATH = "data/eval/results/router_domains_baseline.json"


def _record_from_diag(g: GoldenQuery, diag, meta, commit: str, now: str) -> PredictionRecord:
    gated = [d.value for d in diag.domains]
    raw = [d.value for d in diag.raw_domains]
    low_conf_empty = bool(diag.parse_ok and not diag.domains and diag.raw_domains)
    return PredictionRecord(
        qid=g.qid,
        query_sha=sha12(g.query),
        raw_predicted_domains=raw,
        predicted_domains=gated,
        confidence=diag.confidence,
        use_case=diag.use_case.value,
        parse_ok=diag.parse_ok,
        fallback_reason=diag.fallback_reason,
        low_conf_empty=low_conf_empty,
        requested_model=meta.requested_model,
        effective_provider=meta.effective_provider,
        llm_provider=meta.llm_provider,
        default_model=meta.default_model,
        prompt_sha=meta.prompt_sha,
        schema_version=meta.schema_version,
        commit_sha=commit,
        created_at=now,
    )


async def _build_cache(golden: list[GoldenQuery], cache_path, *, from_cache: bool) -> dict[str, dict]:
    """신선한 캐시는 재사용하고 부족/stale qid만 예측해 캐시를 갱신·저장한다."""
    settings = get_settings()
    meta = current_meta(_REQUESTED_MODEL, settings)
    existing = load_cache(cache_path)
    commit = git_commit_sha()
    now = datetime.now(UTC).isoformat()

    records_by_qid: dict[str, PredictionRecord] = {}
    n_new = n_reused = 0
    missing: list[str] = []
    for g in golden:
        qsha = sha12(g.query)
        cached = existing.get(g.qid)
        reused = _reuse(cached, query_sha=qsha, meta=meta) if cached is not None else None
        if reused is not None:
            records_by_qid[g.qid] = reused
            n_reused += 1
            continue
        if from_cache:
            missing.append(g.qid)
            continue
        diag = await classify_query(g.query, model=_REQUESTED_MODEL)
        records_by_qid[g.qid] = _record_from_diag(g, diag, meta, commit, now)
        n_new += 1

    by_qid = {qid: asdict(rec) for qid, rec in records_by_qid.items()}

    if from_cache:
        if missing:
            print(f"⚠️  --report: 신선 캐시 없는 qid {len(missing)}개 → 리포트에서 제외: {missing[:10]}")
        return by_qid

    all_records = [records_by_qid[g.qid] for g in golden if g.qid in records_by_qid]  # 골든 순서 보존
    write_cache(all_records, cache_path)
    print(f"캐시 저장: {len(all_records)}건 (신규 {n_new}, 재사용 {n_reused}) → {cache_path}")
    return by_qid


def _reuse(cached: dict, *, query_sha: str, meta) -> PredictionRecord | None:
    """신선한 캐시 레코드면 PredictionRecord로 복원, 아니면(구형·stale) None."""
    if not is_fresh(cached, query_sha=query_sha, meta=meta):
        return None
    try:
        return PredictionRecord(**{k: cached[k] for k in PredictionRecord.__dataclass_fields__})
    except KeyError:
        return None  # 필드 누락된 구형 레코드 → 재예측 대상


def _eval_samples(
    golden: list[GoldenQuery], by_qid: dict[str, dict]
) -> tuple[list[RouterPred], list[RouterPred], int, int]:
    """**명시적(사람 검수) router_domains** 보유 질문만으로 raw·effective 표본을 만든다.

    하위호환 fallback(domain 단일)은 정답이 아니므로 제외한다 — 그렇지 않으면 검수 전
    fallback을 검수 정답처럼 평가해 전부 단일 라벨로 보이는 잘못된 결과가 된다.

    - raw 표본: `raw_predicted_domains`(게이팅 전) → 라우터 **분류 능력 + calibration**
    - effective 표본: `predicted_domains`(게이팅 후) → **운영 결과**
    반환: (raw_samples, effective_samples, parse_failures, low_conf_empty).
    """
    raw_samples: list[RouterPred] = []
    eff_samples: list[RouterPred] = []
    parse_failures = low_conf_empty = 0
    for g in golden:
        if not g.has_explicit_router_domains:
            continue
        rec = by_qid.get(g.qid)
        if rec is None:
            continue
        parse_failures += int(not rec.get("parse_ok", True))
        low_conf_empty += int(rec.get("low_conf_empty", False))
        gold = tuple(g.router_domains)
        conf = rec.get("confidence")
        ok = bool(rec.get("parse_ok", False))
        lce = bool(rec.get("low_conf_empty", False))
        eff_samples.append(RouterPred(g.qid, gold, tuple(rec.get("predicted_domains") or []), conf, ok, lce))
        raw_samples.append(RouterPred(g.qid, gold, tuple(rec.get("raw_predicted_domains") or []), conf, ok, lce))
    return raw_samples, eff_samples, parse_failures, low_conf_empty


def _block_breakdown(golden: list[GoldenQuery], by_qid: dict[str, dict]) -> dict:
    """UNANSWERABLE 차단을 near-miss(n0xx)와 사외·일반으로 **분리** 집계 + answerable 오차단율.

    near-miss는 도메인 내 주제지만 코퍼스에 답이 없는 질문 — 질문만으론 차단 판단이 본질적으로
    어렵다. 사외·일반과 섞어 한 숫자로 내면 "라우터 실패율"로 오독되므로 분리한다.
    """

    def blocked(g: GoldenQuery) -> bool:
        return by_qid.get(g.qid, {}).get("use_case") == UseCase.UNANSWERABLE.value

    def rate(n: int, d: int) -> float | None:
        return round(n / d, 4) if d else None

    unans = [g for g in golden if not g.is_answerable and g.qid in by_qid]
    near = [g for g in unans if g.qid.startswith("n")]
    outs = [g for g in unans if not g.qid.startswith("n")]
    ans = [g for g in golden if g.is_answerable and g.qid in by_qid]
    nb = sum(blocked(g) for g in near)
    ob = sum(blocked(g) for g in outs)
    fb = sum(blocked(g) for g in ans)
    return {
        "total": {"blocked": nb + ob, "n": len(unans), "rate": rate(nb + ob, len(unans))},
        "near_miss_n0xx": {
            "blocked": nb,
            "n": len(near),
            "rate": rate(nb, len(near)),
            "note": "질문만으론 코퍼스 정답 부재를 알기 어려움 — 라우터 단계 차단의 본질적 한계",
        },
        "out_of_scope": {
            "blocked": ob,
            "n": len(outs),
            "rate": rate(ob, len(outs)),
            "note": "사외·일상·HR 등 — 라우터가 차단해야 하는 진짜 대상",
        },
        "answerable_false_block": {"false_blocked": fb, "n": len(ans), "rate": rate(fb, len(ans))},
    }


def _metrics_blocks(golden: list[GoldenQuery], by_qid: dict[str, dict]) -> tuple | None:
    """(raw_metrics, effective_dict) 반환. explicit 표본이 없으면 None.

    effective 블록은 calibration(ECE·confidence_bins)을 뺀다 — 게이팅 후 빈 예측을 오답
    처리하면 calibration이 왜곡되므로 raw 기준만 유효.
    """
    raw_s, eff_s, pf, lce = _eval_samples(golden, by_qid)
    if not eff_s:
        return None
    raw_m = summarize(raw_s, parse_failures=pf, low_conf_empty=lce)
    eff_d = summarize(eff_s, parse_failures=pf, low_conf_empty=lce).as_dict()
    for k in ("ece", "ece_n_used", "ece_n_excluded", "confidence_bins"):
        eff_d.pop(k, None)
    return raw_m, eff_d


def _build_result(golden: list[GoldenQuery], by_qid: dict[str, dict]) -> dict | None:
    """재현 가능한 baseline 결과 dict. explicit 표본이 없으면 None.

    재현 메타는 **캐시 stale 키와 동일한 필드**(requested_model·effective_provider·llm_provider·
    default_model·prompt_sha·schema_version)를 그대로 기록해 결과↔캐시 계약을 일치시킨다.
    """
    from app.agents.router.node import _CONFIDENCE_THRESHOLD

    blocks = _metrics_blocks(golden, by_qid)
    if blocks is None:
        return None
    raw_m, eff_d = blocks
    meta = current_meta(_REQUESTED_MODEL, get_settings())
    ans = [g for g in golden if g.is_answerable]
    with open("data/eval/queries.jsonl", "rb") as f:
        golden_sha = hashlib.sha256(f.read()).hexdigest()[:12]
    return {
        "eval_datetime": datetime.now(UTC).isoformat(),
        "reproduction": {
            "golden_sha": golden_sha,
            "code_commit_sha": git_commit_sha(),
            "requested_model": meta.requested_model,
            "effective_provider": meta.effective_provider,
            "llm_provider": meta.llm_provider,
            "default_model": meta.default_model,
            "prompt_sha": meta.prompt_sha,
            "schema_version": meta.schema_version,
            "confidence_threshold": _CONFIDENCE_THRESHOLD,
            "note": "캐시는 gitignore(.cache/onramp-eval/)·LLM 비결정. 완전 재현은 같은 조건으로 --build-cache 재생성 필요.",
        },
        "counts": {
            "golden_total": len(golden),
            "answerable": len(ans),
            "unanswerable": len([g for g in golden if not g.is_answerable]),
            "n_eval_explicit_router_domains": raw_m.n_eval,
        },
        "router_multidomain": {
            "raw_classification_and_calibration": raw_m.as_dict(),
            "effective_after_gate": eff_d,
        },
        "unanswerable_block": _block_breakdown(golden, by_qid),
    }


def _write_result(result: dict, path: str) -> None:
    """baseline 결과를 원자적으로 JSON 저장(.tmp→os.replace)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _provenance_counts(golden: list[GoldenQuery]) -> dict[str, int]:
    answerable = [g for g in golden if g.is_answerable]
    return {
        "answerable": len(answerable),
        "explicit": sum(1 for g in answerable if g.router_domains_source == "explicit"),
        "fallback": sum(1 for g in answerable if g.router_domains_source == "fallback"),
    }


_NO_EXPLICIT_MSG = (
    "\nℹ️  공식 지표를 출력하지 않습니다 — 사람 검수된 router_domains(explicit)가 0건입니다.\n"
    "    절차: ① eval_router_domains.py --build-cache (예측 캐시 생성)\n"
    "          ② draft_router_domains.py (캐시 예측을 검수표 proposed로)\n"
    "          ③ 사람이 reviewed_router_domains 확정 → queries.jsonl 반영\n"
    "          ④ eval_router_domains.py --report (이때부터 멀티도메인 지표가 의미를 가짐)"
)


def _print_report(golden: list[GoldenQuery], by_qid: dict[str, dict], write_path: str | None = None) -> None:
    prov = _provenance_counts(golden)
    print(
        f"\nrouter_domains 정답 출처: explicit(검수){prov['explicit']} · "
        f"fallback(미검수 단일){prov['fallback']} / answerable {prov['answerable']}"
    )
    result = _build_result(golden, by_qid)
    if result is None:
        print(_NO_EXPLICIT_MSG)
        return
    print("\n=== 멀티도메인 라우터 지표 (explicit router_domains만) ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        "\nℹ️  raw=게이팅 전(라우터 분류 능력·confidence calibration) · effective=게이팅 후(운영 결과).\n"
        "    ECE/confidence_bins는 raw 기준만 유효 — 게이팅 후 빈 예측을 오답 처리하면 calibration이 왜곡된다."
    )
    if write_path:
        _write_result(result, write_path)
        print(f"\n✅ baseline 결과 저장: {write_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="멀티도메인 라우터 평가 (예측 캐시 + 지표)")
    parser.add_argument("--build-cache", action="store_true", help="예측 캐시만 생성/갱신(LLM 호출), 지표 리포트 생략")
    parser.add_argument("--report", action="store_true", help="LLM 호출 없이 신선 캐시만으로 지표 리포트")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE_PATH), help="예측 캐시 경로")
    parser.add_argument(
        "--write-result",
        nargs="?",
        const=_DEFAULT_RESULT_PATH,
        default=None,
        help=f"baseline 결과 JSON을 결정론적으로 저장(기본 {_DEFAULT_RESULT_PATH})",
    )
    args = parser.parse_args()

    if args.build_cache and args.report:
        parser.error("--build-cache 와 --report 는 동시에 쓸 수 없습니다 (둘 다 빼면 생성+리포트)")
    if args.build_cache and args.write_result:
        parser.error("--build-cache 는 리포트를 생략하므로 --write-result 와 함께 쓸 수 없습니다")

    golden = load_golden_set()
    # --report: LLM 없이 캐시만. --build-cache: LLM로 캐시 생성(리포트 생략). 기본: 생성+리포트.
    by_qid = asyncio.run(_build_cache(golden, args.cache, from_cache=args.report))
    if not args.build_cache:
        _print_report(golden, by_qid, write_path=args.write_result)


if __name__ == "__main__":
    main()
