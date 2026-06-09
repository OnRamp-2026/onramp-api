"""라우터 도메인 분류 진단 — confusion matrix / P·R·F1 / calibration(ECE) / 실패원인.

라우터 LLM을 골든셋에 직접 돌려 RouterOutput(domain·confidence·use_case)을 수집하고
순수 계산은 app/eval/router_diagnostics.py(테스트 가능)로 위임한다. 예측을 JSONL로
캐시(모델·프롬프트·git SHA·골든셋 hash 메타)해 --from-cache로 LLM 재호출 없이 재현.

⚠️ baseline 주의: 현재 골든셋 gold_domain은 bootstrap_golden이 **문서 payload.domain
(규칙기반)** 을 복사한 값이라 *질문 의도* 라벨이 아니다. 정확도는 라우터오류+문서오태깅
+의도불일치가 섞인 값 → 의도 기준 재검수 전엔 라우터 정확도로 확정하지 말 것.

전제: OPENAI_API_KEY. 사용:
    python scripts/diagnose_router.py                 # LLM 호출 + 캐시 저장
    python scripts/diagnose_router.py --from-cache    # 캐시로 재현(LLM 없음)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from app.agents.router.prompts import ROUTER_SYSTEM_PROMPT
from app.agents.router.schema import RouterOutput
from app.config import get_settings
from app.eval import router_diagnostics as rd
from app.eval.dataset import DEFAULT_QRELS_PATH, DEFAULT_QUERIES_PATH, load_golden_set
from app.services.llm_selector import call_llm, resolve_provider

_DEFAULT_OUT = "data/eval/router_predictions.jsonl"
_TEMPERATURE = 0.0


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


async def _predict(query: str, model: str) -> dict:
    """라우터 LLM 1회 → 예측. 실패는 failure_type으로 구분(llm_call/json_decode/schema_validation)."""
    try:
        raw = await call_llm(ROUTER_SYSTEM_PROMPT, query, model=model, temperature=_TEMPERATURE, json_mode=True)
    except Exception as exc:
        return _fail("llm_call", str(exc), "")
    raw_hash = _sha(raw.encode("utf-8"))
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        return _fail("json_decode", str(exc), raw_hash)
    try:
        out = RouterOutput.model_validate(obj)
    except ValidationError as exc:
        return _fail("schema_validation", str(exc.errors()[:1]), raw_hash)
    return {
        "pred_domain": out.domain.value,
        "confidence": out.confidence,
        "use_case": out.use_case.value,
        "parse_ok": True,
        "failure_type": None,
        "raw_hash": raw_hash,
    }


def _fail(failure_type: str, error: str, raw_hash: str) -> dict:
    return {
        "pred_domain": None,
        "confidence": 0.0,
        "use_case": None,
        "parse_ok": False,
        "failure_type": failure_type,
        "error": error,
        "raw_hash": raw_hash,
    }


def _golden_hash() -> str:
    data = DEFAULT_QUERIES_PATH.read_bytes() + DEFAULT_QRELS_PATH.read_bytes()
    return _sha(data)


async def collect(out_path: Path) -> list[dict]:
    golden = load_golden_set()
    settings = get_settings()
    model = settings.default_model or "gpt-4o-mini"
    rows: list[dict] = []
    for g in golden:
        rows.append(
            {
                "qid": g.qid,
                "query": g.query,
                "gold_domain": g.domain,
                "is_answerable": g.is_answerable,
                **(await _predict(g.query, model)),
            }
        )
    meta = {
        "_meta": True,
        "model": model,
        "provider": resolve_provider(model, settings),
        "temperature": _TEMPERATURE,
        "prompt_hash": _sha(ROUTER_SYSTEM_PROMPT.encode("utf-8")),
        "golden_hash": _golden_hash(),
        "git_sha": _git_sha(),
        "ts": datetime.now(UTC).isoformat(),
        "n": len(rows),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"예측 {len(rows)}건 저장 → {out_path}  (model={model}, git={meta['git_sha'][:8]})\n")
    return rows


def load_cache(path: Path) -> tuple[list[dict], dict]:
    meta: dict = {}
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("_meta"):
            meta = obj
        else:
            rows.append(obj)
    print(f"캐시 로드 {len(rows)}건 ← {path}  (model={meta.get('model')}, git={str(meta.get('git_sha'))[:8]})\n")
    return rows, meta


def report(rows: list[dict]) -> None:
    domains = list(rd.DOMAINS)
    correct, total = rd.overall_accuracy(rows)
    print(f"전체 도메인 정확도(scored {total}): {correct}/{total} = {correct / total:.3f}" if total else "scored 0")

    matrix = rd.confusion_matrix(rows)
    print("\n=== Confusion Matrix (gold↓ × pred→) ===")
    print("gold\\pred".ljust(15) + "".join((d[:9]).ljust(11) for d in domains) + "None")
    for g in domains:
        print(g.ljust(15) + "".join(str(matrix[g][d]).ljust(11) for d in domains) + str(matrix[g][None]))

    per, macro_sup, macro_all = rd.per_domain_prf(matrix)
    supported = [d for d in domains if per[d]["support_gold"] > 0]
    uncovered = [d for d in domains if per[d]["support_gold"] == 0]
    print("\n도메인별 precision / recall / F1")
    for d in domains:
        p = per[d]
        print(
            f"  {d:<14} P={p['precision']:.2f} R={p['recall']:.2f} F1={p['f1']:.2f} (gold {int(p['support_gold'])}, pred {int(p['support_pred'])})"
        )
    print(f"  → macro-F1(지원 {len(supported)}도메인)={macro_sup:.3f}   macro-F1(전체 5도메인)={macro_all:.3f}")
    if uncovered:
        print(f"  ⚠️ 미커버 도메인(gold 0건): {uncovered} — '5도메인 baseline'으로 부르지 말 것")

    table, ece = rd.calibration(rows)
    print("\nconfidence 구간별 정확도 (calibration)")
    for t in table:
        print(f"  {t['bin']}: n={t['n']:<3} 정확도={t['accuracy']:.3f}  평균conf={t['mean_conf']:.3f}")
    print(f"  → ECE={ece:.3f}  (LLM 자기보고 confidence는 완전 확률 아님 → 참고 지표)")

    print("\n주요 혼동 방향 (gold → pred)")
    for (g, p), c in rd.confusion_pairs(rows)[:6]:
        print(f"  {g} → {p}: {c}건")

    fails = rd.failure_summary(rows)
    if fails:
        print(f"\n실패 원인: {dict(fails)}")

    print(
        "\n⚠️ gold_domain은 문서 규칙기반 domain을 복사한 값 → 의도 라벨 아님. 위 '오분류'는 의도 기준 재검수 대상(일부는 gold 오류)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="라우터 도메인 분류 진단")
    parser.add_argument("--out", default=_DEFAULT_OUT, help="예측 캐시 JSONL 경로")
    parser.add_argument("--from-cache", dest="from_cache", action="store_true", help="LLM 호출 없이 캐시로 재현")
    args = parser.parse_args()
    path = Path(args.out)
    rows = load_cache(path)[0] if args.from_cache else asyncio.run(collect(path))
    report(rows)


if __name__ == "__main__":
    main()
