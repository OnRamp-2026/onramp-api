"""LLMOps 알림 — Langfuse Metrics API를 폴링해 임계 초과 시 Slack 통지 (I3, Epic #120).

Langfuse OSS는 native 알림이 없어, 이 스크립트를 nightly/hourly CronJob으로 돌린다.
SLACK_ALERT_WEBHOOK 미설정이면 Slack 전송 대신 로그만 남긴다(웹훅 없이도 검증 가능).

env:
    LANGFUSE_HOST / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY  (필수)
    SLACK_ALERT_WEBHOOK                                        (선택 — 없으면 로그만)
    ALERT_COST_1H_USD       (기본 5.0)  최근 1h 총비용 임계
    ALERT_TRUST_MIN         (기본 0.6)  24h 평균 trust_score 하한
    ALERT_RERANK_FALLBACK   (기본 0.5)  1h rerank 폴백 비율 상한
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import httpx


@dataclass
class Thresholds:
    cost_1h_usd: float = 5.0
    trust_min: float = 0.6
    rerank_fallback_ratio: float = 0.5

    @classmethod
    def from_env(cls) -> Thresholds:
        return cls(
            cost_1h_usd=float(os.getenv("ALERT_COST_1H_USD", "5.0")),
            trust_min=float(os.getenv("ALERT_TRUST_MIN", "0.6")),
            rerank_fallback_ratio=float(os.getenv("ALERT_RERANK_FALLBACK", "0.5")),
        )


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def query_metric(client: httpx.Client, query: dict) -> list[dict]:
    """Langfuse Metrics API 1회 조회 → data 리스트."""
    resp = client.get("/api/public/metrics", params={"query": json.dumps(query)})
    resp.raise_for_status()
    data = resp.json().get("data", [])
    return data if isinstance(data, list) else []


def _first_num(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    val = rows[0].get(key)
    try:
        return float(val) if val is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _pick_num(rows: list[dict], name: str, key: str) -> float:
    """dimensions(name) groupBy 결과에서 특정 name 행의 수치를 꺼낸다. 없으면 0.0."""
    for r in rows:
        if r.get("name") == name and r.get(key) is not None:
            try:
                return float(r[key])
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def evaluate(client: httpx.Client, now: datetime, th: Thresholds) -> list[str]:
    """임계 비교 → 위반 메시지 리스트 (없으면 빈 리스트)."""
    h1, d1 = _iso(now - timedelta(hours=1)), _iso(now - timedelta(days=1))
    now_s = _iso(now)
    breaches: list[str] = []

    # 1) 최근 1h 총비용
    cost = _first_num(
        query_metric(
            client,
            {
                "view": "observations",
                "metrics": [{"measure": "totalCost", "aggregation": "sum"}],
                "fromTimestamp": h1,
                "toTimestamp": now_s,
            },
        ),
        "sum_totalCost",
    )
    if cost > th.cost_1h_usd:
        breaches.append(f"💸 최근 1h 비용 ${cost:.4f} > ${th.cost_1h_usd} (임계 초과)")

    # 2) 24h 평균 trust_score — name으로 groupBy 후 trust_score 행 추출
    #    (filters는 type 디스크리미네이터가 필요해 dimensions 방식이 단순·안정)
    trust = _pick_num(
        query_metric(
            client,
            {
                "view": "scores-numeric",
                "metrics": [{"measure": "value", "aggregation": "avg"}],
                "dimensions": [{"field": "name"}],
                "fromTimestamp": d1,
                "toTimestamp": now_s,
            },
        ),
        "trust_score",
        "avg_value",
    )
    if 0.0 < trust < th.trust_min:
        breaches.append(f"📉 24h 평균 trust_score {trust:.3f} < {th.trust_min} (품질 하락)")

    return breaches


def notify(webhook: str | None, breaches: list[str]) -> None:
    text = "🚨 *OnRamp LLMOps 알림*\n" + "\n".join(f"• {b}" for b in breaches)
    if not webhook:
        print("[alert] SLACK_ALERT_WEBHOOK 미설정 — 로그만 출력:\n" + text)
        return
    with httpx.Client(timeout=10.0) as c:
        c.post(webhook, json={"text": text}).raise_for_status()
    print(f"[alert] Slack 전송 완료 ({len(breaches)}건)")


def main() -> int:
    host = os.getenv("LANGFUSE_HOST", "")
    pk = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    sk = os.getenv("LANGFUSE_SECRET_KEY", "")
    if not (host and pk and sk):
        print("LANGFUSE_HOST/PUBLIC_KEY/SECRET_KEY 필요", file=sys.stderr)
        return 1

    th = Thresholds.from_env()
    with httpx.Client(base_url=host, auth=(pk, sk), timeout=15.0) as client:
        breaches = evaluate(client, datetime.now(UTC), th)

    if breaches:
        notify(os.getenv("SLACK_ALERT_WEBHOOK") or None, breaches)
    else:
        print("[alert] 임계 위반 없음 — 정상")
    return 0


if __name__ == "__main__":
    sys.exit(main())
