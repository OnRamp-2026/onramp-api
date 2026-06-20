from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import ceil
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.session import SessionUser
from app.config import Settings, get_settings
from app.db.models import ChatObservation

ALLOWED_PERIODS = ("1d", "7d", "15d", "30d", "3m", "6m", "1y")
ALLOWED_METRICS = ("token_cost", "traffic_usage", "response_quality", "average_cost", "search_quality")
PERIOD_WINDOWS = {
    "1d": timedelta(days=1),
    "7d": timedelta(days=7),
    "15d": timedelta(days=15),
    "30d": timedelta(days=30),
    "3m": timedelta(days=90),
    "6m": timedelta(days=180),
    "1y": timedelta(days=365),
}
PERIOD_LABELS = {
    "1d": "1일",
    "7d": "7일",
    "15d": "15일",
    "30d": "30일",
    "3m": "3개월",
    "6m": "6개월",
    "1y": "1년",
}
POINT_COUNTS = {"1d": 6, "7d": 6, "15d": 5, "30d": 5, "3m": 6, "6m": 6, "1y": 6}


@dataclass
class MonitoringScope:
    tenant_id: str | None
    scope_value: str
    scope_label: str
    all_scope: bool


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _normalize_datetime(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _ensure_period(period: str) -> timedelta:
    if period not in PERIOD_WINDOWS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="지원하지 않는 period 입니다.")
    return PERIOD_WINDOWS[period]


def resolve_scope(scope: str, user: SessionUser, settings: Settings) -> MonitoringScope:
    normalized = (scope or "").strip()
    if not normalized:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="scope가 필요합니다.")
    if normalized == "all":
        if not settings.monitoring_allow_all_scope_demo:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="전체 범위 조회가 허용되지 않았습니다.")
        return MonitoringScope(tenant_id=None, scope_value="all", scope_label="전체", all_scope=True)
    if normalized != user.tenant_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="다른 tenant 범위는 조회할 수 없습니다.")
    return MonitoringScope(
        tenant_id=normalized,
        scope_value=normalized,
        scope_label=f"{normalized} 테넌트",
        all_scope=False,
    )


async def get_overview(
    db: AsyncSession,
    *,
    user: SessionUser,
    scope: str,
    period: str,
) -> dict[str, Any]:
    settings = get_settings()
    monitoring_scope = resolve_scope(scope, user, settings)
    rows, previous_rows = await _load_rows(db, monitoring_scope=monitoring_scope, period=period)
    aggregate = _build_aggregate(rows, previous_rows=previous_rows, period=period, monitoring_scope=monitoring_scope)
    return {
        "scope": monitoring_scope.scope_value,
        "scopeLabel": monitoring_scope.scope_label,
        "period": period,
        "cards": _build_overview_cards(aggregate),
    }


async def get_detail(
    db: AsyncSession,
    *,
    user: SessionUser,
    scope: str,
    period: str,
    metric_id: str,
) -> dict[str, Any]:
    if metric_id not in ALLOWED_METRICS:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="알 수 없는 metric_id 입니다.")
    settings = get_settings()
    monitoring_scope = resolve_scope(scope, user, settings)
    rows, previous_rows = await _load_rows(db, monitoring_scope=monitoring_scope, period=period)
    aggregate = _build_aggregate(rows, previous_rows=previous_rows, period=period, monitoring_scope=monitoring_scope)
    detail_builders = {
        "token_cost": _build_token_cost_detail,
        "traffic_usage": _build_traffic_usage_detail,
        "response_quality": _build_response_quality_detail,
        "average_cost": _build_average_cost_detail,
        "search_quality": _build_search_quality_detail,
    }
    return detail_builders[metric_id](aggregate)


async def _load_rows(
    db: AsyncSession,
    *,
    monitoring_scope: MonitoringScope,
    period: str,
) -> tuple[list[ChatObservation], list[ChatObservation]]:
    window = _ensure_period(period)
    end = _utcnow()
    start = end - window
    previous_start = start - window
    stmt = select(ChatObservation).where(
        ChatObservation.created_at >= previous_start, ChatObservation.created_at <= end
    )
    if monitoring_scope.tenant_id is not None:
        stmt = stmt.where(ChatObservation.tenant_id == monitoring_scope.tenant_id)
    result = await db.scalars(stmt.order_by(ChatObservation.created_at.asc()))
    all_rows = list(result)
    current_rows = [row for row in all_rows if _normalize_datetime(row.created_at) >= start]
    previous_rows = [row for row in all_rows if previous_start <= _normalize_datetime(row.created_at) < start]
    return current_rows, previous_rows


def _build_aggregate(
    rows: list[ChatObservation],
    *,
    previous_rows: list[ChatObservation],
    period: str,
    monitoring_scope: MonitoringScope,
) -> dict[str, Any]:
    window = _ensure_period(period)
    end = _utcnow()
    start = end - window
    count = len(rows)
    total_tokens = sum(row.total_tokens for row in rows)
    total_cost = sum(row.estimated_cost_usd for row in rows)
    avg_cost = total_cost / count if count else 0.0
    total_duration_values = [row.duration_ms for row in rows]
    p50_latency = _percentile(total_duration_values, 0.50)
    p95_latency = _percentile(total_duration_values, 0.95)
    bucket_counts = _bucketize_counts(rows)
    success_count = bucket_counts["success"]
    requery_count = bucket_counts["requery"]
    failure_count = bucket_counts["failure"]
    success_rate = _safe_rate(success_count, count)
    requery_rate = _safe_rate(requery_count, count)
    failure_rate = _safe_rate(failure_count, count)
    current_request_series = _build_series(rows, start=start, end=end, period=period, value_getter=lambda row: 1)
    cost_series = _build_series(
        rows, start=start, end=end, period=period, value_getter=lambda row: row.estimated_cost_usd
    )
    avg_cost_series = _average_series(
        rows,
        start=start,
        end=end,
        period=period,
        numerator=lambda row: row.estimated_cost_usd,
        denominator=lambda row: 1,
    )
    latency_series = _average_series(
        rows,
        start=start,
        end=end,
        period=period,
        numerator=lambda row: row.duration_ms,
        denominator=lambda row: 1,
        percentile=0.95,
    )
    search_rate_series = _rate_series(
        rows, start=start, end=end, period=period, predicate=lambda row: row.result_bucket == "requery"
    )
    previous_total_cost = sum(row.estimated_cost_usd for row in previous_rows)
    previous_total_tokens = sum(row.total_tokens for row in previous_rows)
    previous_count = len(previous_rows)
    previous_avg_cost = previous_total_cost / previous_count if previous_count else 0.0
    tenant_costs = _group_by_tenant(rows, metric=lambda row: row.estimated_cost_usd)
    tenant_requests = _group_by_tenant(rows, metric=lambda row: 1)
    tenant_avg_costs = {
        tenant_id: tenant_costs.get(tenant_id, 0.0) / tenant_requests.get(tenant_id, 1) for tenant_id in tenant_requests
    }
    return {
        "rows": rows,
        "period": period,
        "periodLabel": PERIOD_LABELS[period],
        "scope": monitoring_scope,
        "count": count,
        "totalTokens": total_tokens,
        "totalCost": total_cost,
        "avgCost": avg_cost,
        "p50Latency": p50_latency,
        "p95Latency": p95_latency,
        "successCount": success_count,
        "requeryCount": requery_count,
        "failureCount": failure_count,
        "successRate": success_rate,
        "requeryRate": requery_rate,
        "failureRate": failure_rate,
        "requestSeries": current_request_series,
        "costSeries": cost_series,
        "avgCostSeries": avg_cost_series,
        "latencySeries": latency_series,
        "searchRateSeries": search_rate_series,
        "previousTotalCost": previous_total_cost,
        "previousTotalTokens": previous_total_tokens,
        "previousCount": previous_count,
        "previousAvgCost": previous_avg_cost,
        "tenantCosts": tenant_costs,
        "tenantRequests": tenant_requests,
        "tenantAvgCosts": tenant_avg_costs,
        "largestTenantByCost": _pick_top_item(tenant_costs),
        "highestAvgCostTenant": _pick_top_item(tenant_avg_costs),
        "latencyByBucket": {
            "all": total_duration_values,
            "success": [row.duration_ms for row in rows if row.result_bucket == "success"],
            "requery": [row.duration_ms for row in rows if row.result_bucket == "requery"],
            "failure": [row.duration_ms for row in rows if row.result_bucket == "failure"],
        },
        "tokenByBucket": {
            "success": sum(row.total_tokens for row in rows if row.result_bucket == "success"),
            "requery": sum(row.total_tokens for row in rows if row.result_bucket == "requery"),
            "failure": sum(row.total_tokens for row in rows if row.result_bucket == "failure"),
        },
        "costByBucket": {
            "success": sum(row.estimated_cost_usd for row in rows if row.result_bucket == "success"),
            "requery": sum(row.estimated_cost_usd for row in rows if row.result_bucket == "requery"),
            "failure": sum(row.estimated_cost_usd for row in rows if row.result_bucket == "failure"),
        },
    }


def _build_overview_cards(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    total_cost = aggregate["totalCost"]
    total_tokens = aggregate["totalTokens"]
    total_requests = aggregate["count"]
    top_tenant = aggregate["largestTenantByCost"]
    highest_avg_cost_tenant = aggregate["highestAvgCostTenant"]
    token_delta = _delta_text(total_tokens, aggregate["previousTotalTokens"], suffix="토큰")
    request_series = aggregate["requestSeries"]
    response_items = [
        {
            "label": "p50 응답 속도",
            "width": _latency_width(aggregate["p50Latency"], aggregate["p95Latency"]),
            "value": _format_duration(aggregate["p50Latency"]),
        },
        {
            "label": "p95 응답 속도",
            "width": _latency_width(aggregate["p95Latency"], aggregate["p95Latency"]),
            "value": _format_duration(aggregate["p95Latency"]),
            "tone": "warning",
        },
        {
            "label": "재질의 요청 p95",
            "width": _latency_width(
                _percentile(aggregate["latencyByBucket"]["requery"], 0.95),
                aggregate["p95Latency"] or 1,
            ),
            "value": _format_duration(_percentile(aggregate["latencyByBucket"]["requery"], 0.95)),
        },
    ]
    search_items = [
        {
            "label": "성공 연결률",
            "width": _format_percent_width(aggregate["successRate"]),
            "value": _format_percent(aggregate["successRate"]),
        },
        {
            "label": "재질의 비중",
            "width": _format_percent_width(aggregate["requeryRate"]),
            "value": _format_percent(aggregate["requeryRate"]),
            "tone": "warning",
        },
        {
            "label": "실패 비중",
            "width": _format_percent_width(aggregate["failureRate"]),
            "value": _format_percent(aggregate["failureRate"]),
        },
    ]
    return [
        {
            "id": "token_cost",
            "row": "top",
            "label": "토큰 사용량 / 비용",
            "title": f"최근 {aggregate['periodLabel']} 검색 LLM 토큰 사용량",
            "chart": "inline",
            "value": _format_usd(total_cost),
            "caption": f"총 {_format_tokens(total_tokens)} 사용, {token_delta}",
            "items": [
                {"label": "이번 기간 누적", "value": _format_tokens(total_tokens)},
                {"label": "일 평균 비용", "value": _format_usd(_daily_average(total_cost, aggregate["period"]))},
                {
                    "label": "가장 큰 테넌트",
                    "value": _format_top_tenant_value(top_tenant, total_cost),
                },
            ],
        },
        {
            "id": "traffic_usage",
            "row": "top",
            "label": "요청량 / 사용량 추이",
            "title": f"최근 {aggregate['periodLabel']} 서비스 사용량",
            "chart": "line",
            "value": f"{total_requests:,}건",
            "caption": _traffic_caption(request_series),
            "yTicks": _y_ticks_from_points(request_series),
            "points": request_series,
        },
        {
            "id": "response_quality",
            "row": "bottom",
            "label": "응답 품질",
            "title": "사용자 체감 응답 속도",
            "chart": "stacked",
            "value": _format_duration(aggregate["p95Latency"]),
            "caption": f"최근 {aggregate['periodLabel']} p95 기준, p50은 {_format_duration(aggregate['p50Latency'])} 입니다",
            "items": response_items,
        },
        {
            "id": "average_cost",
            "row": "bottom",
            "label": "평균 비용",
            "title": "요청당 평균 비용",
            "chart": "inline",
            "value": _format_usd(aggregate["avgCost"]),
            "caption": _average_cost_caption(aggregate, highest_avg_cost_tenant),
            "items": [
                {"label": "일 평균 비용", "value": _format_usd(_daily_average(total_cost, aggregate["period"]))},
                {"label": "전체 요청", "value": f"{total_requests:,}건"},
                {"label": "비용 집중 테넌트", "value": highest_avg_cost_tenant[0] if highest_avg_cost_tenant else "-"},
            ],
        },
        {
            "id": "search_quality",
            "row": "bottom",
            "label": "검색 품질 / 재질의율",
            "title": "검색 흐름 품질",
            "chart": "stacked",
            "value": _format_percent(aggregate["requeryRate"]),
            "caption": "요청 1건 기준으로 성공, 실패, 재질의 중 하나로 종료됩니다",
            "items": search_items,
        },
    ]


def _build_token_cost_detail(aggregate: dict[str, Any]) -> dict[str, Any]:
    total_cost = aggregate["totalCost"]
    total_tokens = aggregate["totalTokens"]
    avg_cost = aggregate["avgCost"]
    requery_token_share = _safe_rate(aggregate["tokenByBucket"]["requery"], total_tokens)
    return {
        "title": "토큰 사용량 / 비용 상세",
        "badge": "비용 해석",
        "description": "현재 1차 비용은 chat LLM 호출 토큰 기준 추정치이며 embedding/reranker 비용은 포함하지 않습니다.",
        "headline": "재질의 비중이 높을수록 총 토큰과 평균 비용이 함께 증가합니다.",
        "summaryMetrics": [
            {"label": "총 토큰", "value": _format_tokens(total_tokens)},
            {"label": "총 비용", "value": _format_usd(total_cost)},
            {"label": "요청당 평균 비용", "value": _format_usd(avg_cost)},
        ],
        "chart": {
            "label": "비용 추세",
            "title": f"최근 {aggregate['periodLabel']} 비용 변화",
            "type": "line",
            "yTicks": _y_ticks_from_points(aggregate["costSeries"]),
            "points": aggregate["costSeries"],
        },
        "breakdownTitle": "결과 버킷별 토큰 사용량",
        "breakdownItems": [
            _breakdown_item("성공 요청", aggregate["tokenByBucket"]["success"], total_tokens),
            _breakdown_item("재질의 요청", aggregate["tokenByBucket"]["requery"], total_tokens, tone="warning"),
            _breakdown_item("실패 요청", aggregate["tokenByBucket"]["failure"], total_tokens),
        ],
        "items": [
            {
                "tag": "토큰",
                "tone": "healthy",
                "title": f"전체 토큰 중 재질의 비중은 {_format_percent(requery_token_share)} 입니다",
                "detail": "내부 재검색/재시도 루프가 많이 도는 tenant일수록 비용 압력이 먼저 커집니다.",
            },
            {
                "tag": "범위",
                "tone": "neutral",
                "title": "현재 비용 계산은 /v1/chat LLM 호출만 반영합니다",
                "detail": "embedding, reranker, 워크플로우 계열 비용은 이번 라운드에서 제외되어 있습니다.",
            },
            {
                "tag": "운영",
                "tone": "warning",
                "title": "평균 비용은 절대 청구액이 아니라 방향성을 읽는 용도입니다",
                "detail": "비용 증가 tenant를 빠르게 찾고, 이후 실제 billing 기준과 맞춰가는 1차 관측값입니다.",
            },
        ],
    }


def _build_traffic_usage_detail(aggregate: dict[str, Any]) -> dict[str, Any]:
    total_requests = aggregate["count"]
    peak_point = max(aggregate["requestSeries"], key=lambda item: item["value"], default={"label": "-", "value": 0})
    return {
        "title": "요청량 / 사용량 추이 상세",
        "badge": "트래픽 관측",
        "description": "운영자는 먼저 총 요청량과 피크 구간을 보고, 이상 징후가 있을 때만 세부 흐름을 확인합니다.",
        "headline": f"가장 높은 요청 구간은 {peak_point['label']}이며 {peak_point['value']:,}건이 관측되었습니다.",
        "summaryMetrics": [
            {"label": "총 요청", "value": f"{total_requests:,}건"},
            {"label": "성공 요청", "value": f"{aggregate['successCount']:,}건"},
            {"label": "재질의 요청", "value": f"{aggregate['requeryCount']:,}건"},
        ],
        "chart": {
            "label": "요청량 추세",
            "title": f"최근 {aggregate['periodLabel']} 요청량",
            "type": "line",
            "yTicks": _y_ticks_from_points(aggregate["requestSeries"]),
            "points": aggregate["requestSeries"],
        },
        "breakdownTitle": "결과 버킷별 요청 분포",
        "breakdownItems": [
            _breakdown_item("성공 요청", aggregate["successCount"], total_requests),
            _breakdown_item("재질의 요청", aggregate["requeryCount"], total_requests, tone="warning"),
            _breakdown_item("실패 요청", aggregate["failureCount"], total_requests),
        ],
        "items": [
            {
                "tag": "피크",
                "tone": "neutral",
                "title": f"피크 구간은 {peak_point['label']} 입니다",
                "detail": "이 구간의 요청 증가가 p95 지연과 비용 상승으로 바로 이어지는지 함께 봐야 합니다.",
            },
            {
                "tag": "품질",
                "tone": "healthy",
                "title": "성공 요청량만 따로 보면 실제 서비스 체감 사용량을 더 잘 읽을 수 있습니다",
                "detail": "총 요청량이 같아도 실패/재질의 비중이 높으면 운영 부담은 더 커집니다.",
            },
            {
                "tag": "대시보드",
                "tone": "neutral",
                "title": "메인 화면은 요청량 총합만 보여주고 상세에서 흐름을 나눠봅니다",
                "detail": "과한 세부 그래프를 메인에 올리지 않고, 클릭 후 분해해서 보는 구조를 유지합니다.",
            },
        ],
    }


def _build_response_quality_detail(aggregate: dict[str, Any]) -> dict[str, Any]:
    requery_p95 = _percentile(aggregate["latencyByBucket"]["requery"], 0.95)
    failure_p95 = _percentile(aggregate["latencyByBucket"]["failure"], 0.95)
    return {
        "title": "응답 품질 상세",
        "badge": "Latency 중심",
        "description": "이 상세 화면은 성공률 대신 latency만 봅니다. 운영자는 어디서 응답이 길어지는지 먼저 확인합니다.",
        "headline": f"최근 {aggregate['periodLabel']} 기준 p95 응답 속도는 {_format_duration(aggregate['p95Latency'])} 입니다.",
        "summaryMetrics": [
            {"label": "p50 응답 속도", "value": _format_duration(aggregate["p50Latency"])},
            {"label": "p95 응답 속도", "value": _format_duration(aggregate["p95Latency"])},
            {"label": "재질의 요청 p95", "value": _format_duration(requery_p95)},
        ],
        "chart": {
            "label": "Latency 추세",
            "title": f"최근 {aggregate['periodLabel']} p95 응답 속도",
            "type": "line",
            "yTicks": _y_ticks_from_points(aggregate["latencySeries"]),
            "points": aggregate["latencySeries"],
        },
        "breakdownTitle": "응답 속도 분해",
        "breakdownItems": [
            _latency_breakdown_item("전체 p50", aggregate["p50Latency"], aggregate["p95Latency"]),
            _latency_breakdown_item("전체 p95", aggregate["p95Latency"], aggregate["p95Latency"], tone="warning"),
            _latency_breakdown_item("재질의 요청 p95", requery_p95, aggregate["p95Latency"]),
            _latency_breakdown_item("실패 요청 p95", failure_p95, aggregate["p95Latency"]),
        ],
        "items": [
            {
                "tag": "Latency",
                "tone": "warning",
                "title": "재질의가 발생한 요청은 일반 요청보다 응답이 길어질 가능성이 높습니다",
                "detail": "응답 속도 저하는 검색 성공률과 함께 비용까지 건드리므로 운영 우선순위가 높습니다.",
            },
            {
                "tag": "품질",
                "tone": "neutral",
                "title": "이번 1차 상세에서는 성공률을 제외했습니다",
                "detail": "응답 품질은 체감 속도 중심으로 보고, 성공/실패는 검색 품질 상세에서 따로 확인합니다.",
            },
            {
                "tag": "운영",
                "tone": "healthy",
                "title": "p50과 p95를 같이 보면 평균이 아니라 꼬리 지연을 읽을 수 있습니다",
                "detail": "메인 카드 한 숫자만으로는 숨는 지연 분산을 상세 화면에서 드러냅니다.",
            },
        ],
    }


def _build_average_cost_detail(aggregate: dict[str, Any]) -> dict[str, Any]:
    is_all = aggregate["scope"].all_scope
    if is_all:
        breakdown_items = [
            _breakdown_item(tenant_id, value, max(aggregate["tenantCosts"].values(), default=0.0), raw_value=True)
            for tenant_id, value in sorted(aggregate["tenantCosts"].items(), key=lambda item: item[1], reverse=True)[:3]
        ]
        notes = [
            {
                "tag": "전체",
                "tone": "neutral",
                "title": "전체 범위에서는 어떤 tenant가 비용 압력을 만드는지 먼저 봅니다",
                "detail": "플랫폼 운영자 관점에서는 고비용 tenant를 빠르게 찾는 것이 우선입니다.",
            },
            {
                "tag": "평균",
                "tone": "healthy",
                "title": "요청당 평균 비용은 트래픽 총량보다 효율 저하를 먼저 드러냅니다",
                "detail": "요청 수가 같아도 재질의 비중이 늘면 평균 비용이 먼저 상승합니다.",
            },
            {
                "tag": "테넌트",
                "tone": "warning",
                "title": "scope=all 과 tenant 상세는 같은 지표를 다른 관점으로 읽습니다",
                "detail": "전체는 비교, tenant는 해당 서비스 자체의 압력을 보는 용도로 나뉩니다.",
            },
        ]
    else:
        breakdown_items = [
            _breakdown_item("성공 요청", aggregate["costByBucket"]["success"], aggregate["totalCost"], raw_value=True),
            _breakdown_item(
                "재질의 요청",
                aggregate["costByBucket"]["requery"],
                aggregate["totalCost"],
                raw_value=True,
                tone="warning",
            ),
            _breakdown_item("실패 요청", aggregate["costByBucket"]["failure"], aggregate["totalCost"], raw_value=True),
        ]
        notes = [
            {
                "tag": "tenant",
                "tone": "neutral",
                "title": "tenant 상세에서는 외부 비교보다 내부 비용 구조를 봅니다",
                "detail": "이 tenant 안에서 어떤 유형의 요청이 평균 비용을 끌어올리는지가 더 중요합니다.",
            },
            {
                "tag": "재질의",
                "tone": "warning",
                "title": "재질의 요청이 많으면 요청당 평균 비용이 빠르게 상승합니다",
                "detail": "검색 성공 연결이 흔들리는지 함께 점검해야 합니다.",
            },
            {
                "tag": "활용",
                "tone": "healthy",
                "title": "평균 비용은 경보보다 해석 지표에 가깝습니다",
                "detail": "이상 징후를 찾은 뒤 실제 토큰/트래픽 상세로 들어가는 진입점입니다.",
            },
        ]
    return {
        "title": "평균 비용 상세",
        "badge": "방향성 지표",
        "description": "비용 예측 모델 대신 현재는 요청당 평균 비용을 보여줍니다. 실제 구현 전 운영 방향을 읽기 위한 1차 지표입니다.",
        "headline": f"최근 {aggregate['periodLabel']} 요청당 평균 비용은 {_format_usd(aggregate['avgCost'])} 입니다.",
        "summaryMetrics": [
            {"label": "요청당 평균 비용", "value": _format_usd(aggregate["avgCost"])},
            {"label": "총 비용", "value": _format_usd(aggregate["totalCost"])},
            {"label": "총 요청", "value": f"{aggregate['count']:,}건"},
        ],
        "chart": {
            "label": "평균 비용 추세",
            "title": f"최근 {aggregate['periodLabel']} 평균 비용 변화",
            "type": "line",
            "yTicks": _y_ticks_from_points(aggregate["avgCostSeries"]),
            "points": aggregate["avgCostSeries"],
        },
        "breakdownTitle": "평균 비용 분해",
        "breakdownItems": breakdown_items,
        "items": notes,
    }


def _build_search_quality_detail(aggregate: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": "검색 품질 / 재질의율 상세",
        "badge": "요청 단위 판정",
        "description": "현재 재질의는 실제 사용자 후속 질문이 아니라 내부 재검색/재시도 루프가 발생한 요청을 뜻합니다.",
        "headline": f"검색 성공 연결률은 {_format_percent(aggregate['successRate'])}, 재질의율은 {_format_percent(aggregate['requeryRate'])} 입니다.",
        "summaryMetrics": [
            {"label": "성공 연결률", "value": _format_percent(aggregate["successRate"])},
            {"label": "재질의율", "value": _format_percent(aggregate["requeryRate"])},
            {"label": "실패율", "value": _format_percent(aggregate["failureRate"])},
        ],
        "chart": {
            "label": "재질의 추세",
            "title": f"최근 {aggregate['periodLabel']} 재질의율 변화",
            "type": "line",
            "yTicks": _y_ticks_from_points(aggregate["searchRateSeries"]),
            "points": aggregate["searchRateSeries"],
        },
        "breakdownTitle": "요청 결과 분포",
        "breakdownItems": [
            _breakdown_item("성공 연결", aggregate["successCount"], aggregate["count"]),
            _breakdown_item("재질의", aggregate["requeryCount"], aggregate["count"], tone="warning"),
            _breakdown_item("실패", aggregate["failureCount"], aggregate["count"]),
        ],
        "items": [
            {
                "tag": "정의",
                "tone": "neutral",
                "title": "성공 / 실패 / 재질의는 요청 1건 단위로 판정합니다",
                "detail": "사용자 세션 묶음이 아니라 API 요청 한 건이 어떻게 끝났는지로 보는 1차 정의입니다.",
            },
            {
                "tag": "품질",
                "tone": "warning",
                "title": "재질의율이 오르면 비용과 latency가 함께 흔들릴 가능성이 큽니다",
                "detail": "검색 성공 연결이 떨어지는지, 특정 tenant에서만 심한지 같이 확인해야 합니다.",
            },
            {
                "tag": "운영",
                "tone": "healthy",
                "title": "실패율보다 재질의율이 먼저 오르는지 보는 것이 더 빠른 초기 신호가 됩니다",
                "detail": "완전 실패 전에 검색 품질 저하를 잡을 수 있기 때문입니다.",
            },
        ],
    }


def _build_series(
    rows: list[ChatObservation],
    *,
    start: datetime,
    end: datetime,
    period: str,
    value_getter,
) -> list[dict[str, Any]]:
    grouped = _init_buckets(start=start, end=end, period=period)
    for row in rows:
        index = _bucket_index(_normalize_datetime(row.created_at), start=start, end=end, count=len(grouped))
        grouped[index]["value"] += float(value_getter(row))
    return _finalize_points(grouped)


def _average_series(
    rows: list[ChatObservation],
    *,
    start: datetime,
    end: datetime,
    period: str,
    numerator,
    denominator,
    percentile: float | None = None,
) -> list[dict[str, Any]]:
    grouped = _init_buckets(start=start, end=end, period=period)
    if percentile is None:
        for row in rows:
            index = _bucket_index(_normalize_datetime(row.created_at), start=start, end=end, count=len(grouped))
            grouped[index]["value"] += float(numerator(row))
            grouped[index]["denominator"] += float(denominator(row))
        points = []
        for bucket in grouped:
            if bucket["denominator"] > 0:
                value = bucket["value"] / bucket["denominator"]
            else:
                value = 0.0
            points.append({"label": bucket["label"], "value": round(value, 3)})
        return points
    for row in rows:
        index = _bucket_index(_normalize_datetime(row.created_at), start=start, end=end, count=len(grouped))
        grouped[index]["samples"].append(float(numerator(row)))
    return [
        {"label": bucket["label"], "value": round(_percentile(bucket["samples"], percentile), 3)} for bucket in grouped
    ]


def _rate_series(
    rows: list[ChatObservation],
    *,
    start: datetime,
    end: datetime,
    period: str,
    predicate,
) -> list[dict[str, Any]]:
    grouped = _init_buckets(start=start, end=end, period=period)
    for row in rows:
        index = _bucket_index(_normalize_datetime(row.created_at), start=start, end=end, count=len(grouped))
        grouped[index]["denominator"] += 1
        if predicate(row):
            grouped[index]["value"] += 100.0
    return [
        {
            "label": bucket["label"],
            "value": round(bucket["value"] / bucket["denominator"], 1) if bucket["denominator"] else 0.0,
        }
        for bucket in grouped
    ]


def _init_buckets(*, start: datetime, end: datetime, period: str) -> list[dict[str, Any]]:
    count = POINT_COUNTS[period]
    total_seconds = max((end - start).total_seconds(), 1.0)
    step = total_seconds / count
    buckets = []
    for index in range(count):
        bucket_start = start + timedelta(seconds=step * index)
        buckets.append(
            {
                "label": _bucket_label(bucket_start, period),
                "value": 0.0,
                "denominator": 0.0,
                "samples": [],
            }
        )
    return buckets


def _bucket_index(value: datetime, *, start: datetime, end: datetime, count: int) -> int:
    total = max((end - start).total_seconds(), 1.0)
    elapsed = max(0.0, min((value - start).total_seconds(), total))
    index = int((elapsed / total) * count)
    return min(index, count - 1)


def _finalize_points(grouped: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points = []
    for bucket in grouped:
        value = bucket["value"]
        rounded = round(value, 3)
        if abs(rounded - round(rounded)) < 0.001:
            rounded = int(round(rounded))
        points.append({"label": bucket["label"], "value": rounded})
    return points


def _bucket_label(dt: datetime, period: str) -> str:
    if period == "1d":
        return dt.astimezone(UTC).strftime("%H:%M")
    if period in {"7d", "15d", "30d"}:
        return dt.astimezone(UTC).strftime("%m/%d")
    return dt.astimezone(UTC).strftime("%y/%m")


def _bucketize_counts(rows: list[ChatObservation]) -> dict[str, int]:
    counts = {"success": 0, "requery": 0, "failure": 0}
    for row in rows:
        counts[row.result_bucket] = counts.get(row.result_bucket, 0) + 1
    return counts


def _group_by_tenant(rows: list[ChatObservation], *, metric) -> dict[str, float]:
    grouped: dict[str, float] = defaultdict(float)
    for row in rows:
        grouped[row.tenant_id] += float(metric(row))
    return dict(grouped)


def _pick_top_item(values: dict[str, float]) -> tuple[str, float] | None:
    if not values:
        return None
    return max(values.items(), key=lambda item: item[1])


def _percentile(values: list[int | float], pct: float) -> float:
    cleaned = sorted(float(value) for value in values if value is not None)
    if not cleaned:
        return 0.0
    if len(cleaned) == 1:
        return cleaned[0]
    rank = pct * (len(cleaned) - 1)
    low = int(rank)
    high = min(low + 1, len(cleaned) - 1)
    fraction = rank - low
    return cleaned[low] + (cleaned[high] - cleaned[low]) * fraction


def _safe_rate(value: float, total: float) -> float:
    return (value / total * 100.0) if total else 0.0


def _format_usd(value: float) -> str:
    return f"${value:,.2f}"


def _format_tokens(value: int) -> str:
    return f"{value:,} 토큰"


def _format_duration(value: float) -> str:
    if value <= 0:
        return "0ms"
    if value < 1000:
        return f"{int(round(value))}ms"
    return f"{value / 1000:.1f}초"


def _format_percent(value: float) -> str:
    if abs(value - round(value)) < 0.05:
        return f"{int(round(value))}%"
    return f"{value:.1f}%"


def _format_percent_width(value: float) -> str:
    return f"{max(6.0, min(100.0, value)):.0f}%"


def _y_ticks_from_points(points: list[dict[str, Any]]) -> list[float | int]:
    max_value = max((float(point["value"]) for point in points), default=0.0)
    if max_value <= 0:
        return [0, 1, 2, 3]
    step = max_value / 3
    ticks = [0.0, step, step * 2, step * 3]
    normalized: list[float | int] = []
    for tick in ticks:
        if max_value >= 100:
            normalized.append(int(ceil(tick / 10.0) * 10))
        elif max_value >= 10:
            normalized.append(int(ceil(tick)))
        else:
            normalized.append(round(tick, 2))
    deduped = []
    for value in normalized:
        if value not in deduped:
            deduped.append(value)
    while len(deduped) < 4:
        deduped.append(deduped[-1] + 1 if deduped else 1)
    return deduped[:4]


def _delta_text(current: float, previous: float, *, suffix: str) -> str:
    if previous <= 0:
        return "이전 집계가 없어 변화율을 계산하지 않았습니다"
    delta = ((current - previous) / previous) * 100.0
    direction = "증가" if delta >= 0 else "감소"
    return f"직전 동일 기간 대비 {abs(delta):.1f}% {direction}"


def _daily_average(value: float, period: str) -> float:
    days = max(PERIOD_WINDOWS[period].days, 1)
    return value / days


def _format_top_tenant_value(top_tenant: tuple[str, float] | None, total_cost: float) -> str:
    if top_tenant is None or total_cost <= 0:
        return "-"
    tenant_id, cost = top_tenant
    return f"{tenant_id} {_format_percent(_safe_rate(cost, total_cost))}"


def _traffic_caption(points: list[dict[str, Any]]) -> str:
    peak_point = max(points, key=lambda item: item["value"], default={"label": "-", "value": 0})
    return f"{peak_point['label']} 구간에 요청이 가장 많이 집중되었습니다"


def _average_cost_caption(aggregate: dict[str, Any], highest_avg_cost_tenant: tuple[str, float] | None) -> str:
    if aggregate["scope"].all_scope and highest_avg_cost_tenant is not None:
        return f"{highest_avg_cost_tenant[0]} 가 요청당 평균 비용 기준으로 가장 높은 편입니다"
    delta = _delta_text(aggregate["avgCost"], aggregate["previousAvgCost"], suffix="평균 비용")
    return f"현재 선택 범위 기준 {delta}"


def _breakdown_item(
    label: str, value: float, total: float, *, tone: str | None = None, raw_value: bool = False
) -> dict[str, Any]:
    if raw_value:
        ratio = _safe_rate(value, total)
        item_value = _format_usd(value)
    else:
        ratio = _safe_rate(value, total)
        item_value = _format_percent(ratio)
    item = {"label": label, "width": _format_percent_width(ratio), "value": item_value}
    if tone is not None:
        item["tone"] = tone
    return item


def _latency_width(value: float, baseline: float) -> str:
    if baseline <= 0:
        return "6%"
    return f"{max(6.0, min(100.0, value / baseline * 100.0)):.0f}%"


def _latency_breakdown_item(label: str, value: float, baseline: float, *, tone: str | None = None) -> dict[str, Any]:
    item = {"label": label, "width": _latency_width(value, baseline), "value": _format_duration(value)}
    if tone is not None:
        item["tone"] = tone
    return item
