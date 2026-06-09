"""라우터 도메인 분류 진단 — 골든셋으로 confusion matrix / P·R·F1 / confidence calibration.

라우터 LLM을 골든셋 질의에 직접 돌려 RouterOutput(domain·confidence·use_case)을 받아:
  · 도메인 confusion matrix (gold × pred, None/파싱실패 포함)
  · 도메인별 precision/recall/F1 + macro-F1
  · confidence 구간별 실제 정확도 (calibration)
  · 오분류 목록 (gold → pred)
  · domain=None(저confidence/파싱실패) · UNANSWERABLE 비율
예측을 data/eval/router_predictions.jsonl 에 저장(모델·시각 메타 포함)해 A/B 재현에 재사용.

전제: OPENAI_API_KEY. 사용: python scripts/diagnose_router.py [--out data/eval/router_predictions.jsonl]
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from app.agents.router.prompts import ROUTER_SYSTEM_PROMPT
from app.agents.router.schema import RouterOutput
from app.agents.state import Domain, UseCase
from app.config import get_settings
from app.eval.dataset import load_golden_set
from app.services.llm_selector import call_llm

DOMAINS = [d.value for d in Domain]
_CONF_BUCKETS = [(0.9, 1.01, "0.9–1.0"), (0.7, 0.9, "0.7–0.9"), (0.5, 0.7, "0.5–0.7"), (0.0, 0.5, "0.0–0.5")]


async def _predict(query: str, model: str) -> dict:
    """라우터 LLM 1회 → raw 예측. 파싱 실패는 pred_domain=None, parse_ok=False."""
    try:
        raw = await call_llm(ROUTER_SYSTEM_PROMPT, query, model=model, json_mode=True)
    except Exception as exc:  # LLM 호출 실패
        return {"pred_domain": None, "confidence": 0.0, "use_case": None, "parse_ok": False, "error": str(exc)}
    try:
        out = RouterOutput.model_validate_json(raw)
    except ValidationError:
        return {"pred_domain": None, "confidence": 0.0, "use_case": None, "parse_ok": False}
    return {
        "pred_domain": out.domain.value,
        "confidence": out.confidence,
        "use_case": out.use_case.value,
        "parse_ok": True,
    }


async def collect(out_path: Path) -> list[dict]:
    golden = load_golden_set()
    settings = get_settings()
    model = settings.default_model
    rows: list[dict] = []
    for g in golden:
        pred = await _predict(g.query, model)
        rows.append(
            {
                "qid": g.qid,
                "query": g.query,
                "gold_domain": g.domain,
                "is_answerable": g.is_answerable,
                **pred,
            }
        )
    meta = {"_meta": True, "model": model, "ts": datetime.now(UTC).isoformat(), "n": len(rows)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"예측 {len(rows)}건 저장 → {out_path} (model={model})\n")
    return rows


def _confusion(rows: list[dict]) -> dict[str, dict[str, int]]:
    """answerable + gold 도메인 보유 행만. gold → pred(None 포함) 카운트."""
    labels = [*DOMAINS, None]
    matrix = {g: {p: 0 for p in labels} for g in DOMAINS}
    for r in rows:
        if r["is_answerable"] and r["gold_domain"] in DOMAINS:
            matrix[r["gold_domain"]][r["pred_domain"] if r["pred_domain"] in DOMAINS else None] += 1
    return matrix


def _print_confusion(matrix: dict[str, dict[str, int]]) -> None:
    preds = [*DOMAINS, None]
    head = "gold\\pred".ljust(14) + "".join((p or "None")[:9].ljust(11) for p in preds)
    print(head)
    for g in DOMAINS:
        row = g[:13].ljust(14) + "".join(str(matrix[g][p]).ljust(11) for p in preds)
        print(row)


def _prf(matrix: dict[str, dict[str, int]]) -> None:
    print("\n도메인별 precision / recall / F1")
    f1s = []
    for d in DOMAINS:
        tp = matrix[d][d]
        actual = sum(matrix[d].values())  # gold=d 총수 (recall 분모)
        predicted = sum(matrix[g][d] for g in DOMAINS)  # pred=d 총수 (precision 분모)
        recall = tp / actual if actual else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1s.append(f1)
        print(f"  {d:<14} P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}  (gold {actual}, pred {predicted})")
    print(f"  → macro-F1 = {sum(f1s) / len(f1s):.3f}")


def _calibration(rows: list[dict]) -> None:
    print("\nconfidence 구간별 실제 정확도 (calibration)")
    scored = [r for r in rows if r["is_answerable"] and r["gold_domain"] in DOMAINS]
    for lo, hi, label in _CONF_BUCKETS:
        bucket = [r for r in scored if lo <= r["confidence"] < hi]
        if not bucket:
            print(f"  {label}: n=0")
            continue
        correct = sum(r["pred_domain"] == r["gold_domain"] for r in bucket)
        print(f"  {label}: n={len(bucket):<3} 정확도={correct / len(bucket):.3f}")


def _misclassified(rows: list[dict]) -> None:
    print("\n오분류 (gold → pred)")
    miss = [r for r in rows if r["is_answerable"] and r["gold_domain"] in DOMAINS and r["pred_domain"] != r["gold_domain"]]
    for r in miss:
        print(f"  [{r['qid']}] {r['gold_domain']} → {r['pred_domain']}  conf={r['confidence']:.2f}  | {r['query'][:40]}")
    print(f"  (총 {len(miss)}건)")


def _rates(rows: list[dict]) -> None:
    answerable = [r for r in rows if r["is_answerable"] and r["gold_domain"] in DOMAINS]
    gated = sum(r["confidence"] < 0.5 for r in answerable)  # conf<0.5 → 검색에서 domain=None
    parse_fail = sum(not r["parse_ok"] for r in rows)
    wrong_block = sum(r["is_answerable"] and r["use_case"] == UseCase.UNANSWERABLE.value for r in rows)
    unans = [r for r in rows if not r["is_answerable"]]
    right_block = sum(r["use_case"] == UseCase.UNANSWERABLE.value for r in unans)
    print("\n기타")
    print(f"  answerable 중 conf<0.5(검색 시 무필터): {gated}/{len(answerable)}")
    print(f"  파싱 실패(domain=None): {parse_fail}/{len(rows)}")
    print(f"  answerable인데 UNANSWERABLE로 잘못 차단: {wrong_block}")
    if unans:
        print(f"  범위밖 정상 차단: {right_block}/{len(unans)}")


def report(rows: list[dict]) -> None:
    matrix = _confusion(rows)
    print("=== Confusion Matrix (gold × pred) ===")
    _print_confusion(matrix)
    _prf(matrix)
    _calibration(rows)
    _misclassified(rows)
    _rates(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="라우터 도메인 분류 진단")
    parser.add_argument("--out", default="data/eval/router_predictions.jsonl", help="예측 캐시 JSONL 경로")
    args = parser.parse_args()
    rows = asyncio.run(collect(Path(args.out)))
    report(rows)


if __name__ == "__main__":
    main()
