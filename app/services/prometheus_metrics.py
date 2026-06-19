from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from redis.asyncio import Redis
from redis.exceptions import ResponseError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import EventOutbox, ReportJob, ReportJobStatus
from app.queue.constants import (
    REPORT_EVENT_GROUP,
    STT_COMPLETED_STREAM,
    STT_PROGRESS_STREAM,
    STT_TRANSCRIPT_COMPLETED_STREAM,
    TRANSCRIPT_OBSERVER_GROUP,
    WORKFLOW_UPDATER_GROUP,
)


@dataclass(frozen=True)
class StreamGroupMetric:
    stream: str
    group: str
    pending: int
    lag: int


@dataclass(frozen=True)
class WorkerMetricSnapshot:
    collected_at: datetime
    report_jobs_queued: int
    report_jobs_processing: int
    event_outbox_pending: int
    stream_lengths: dict[str, int]
    stream_groups: list[StreamGroupMetric]


_STREAM_GROUPS: tuple[tuple[str, str], ...] = (
    (STT_PROGRESS_STREAM, WORKFLOW_UPDATER_GROUP),
    (STT_TRANSCRIPT_COMPLETED_STREAM, TRANSCRIPT_OBSERVER_GROUP),
    (STT_COMPLETED_STREAM, REPORT_EVENT_GROUP),
)


async def collect_worker_metric_snapshot(session: AsyncSession, redis: Redis) -> WorkerMetricSnapshot:
    queued_stmt = select(func.count()).select_from(ReportJob).where(ReportJob.status == ReportJobStatus.queued)
    processing_stmt = select(func.count()).select_from(ReportJob).where(ReportJob.status == ReportJobStatus.processing)
    outbox_stmt = (
        select(func.count())
        .select_from(EventOutbox)
        .where(EventOutbox.published_at.is_(None), EventOutbox.available_at <= datetime.now(UTC))
    )

    report_jobs_queued = int((await session.scalar(queued_stmt)) or 0)
    report_jobs_processing = int((await session.scalar(processing_stmt)) or 0)
    event_outbox_pending = int((await session.scalar(outbox_stmt)) or 0)

    stream_lengths: dict[str, int] = {}
    stream_groups: list[StreamGroupMetric] = []

    for stream, group in _STREAM_GROUPS:
        stream_lengths[stream] = await _safe_xlen(redis, stream)
        pending, lag = await _load_group_metrics(redis, stream, group)
        stream_groups.append(StreamGroupMetric(stream=stream, group=group, pending=pending, lag=lag))

    return WorkerMetricSnapshot(
        collected_at=datetime.now(UTC),
        report_jobs_queued=report_jobs_queued,
        report_jobs_processing=report_jobs_processing,
        event_outbox_pending=event_outbox_pending,
        stream_lengths=stream_lengths,
        stream_groups=stream_groups,
    )


def render_worker_metrics(snapshot: WorkerMetricSnapshot) -> str:
    lines = [
        "# HELP onramp_report_jobs Number of report jobs by status.",
        "# TYPE onramp_report_jobs gauge",
        f'onramp_report_jobs{{status="queued"}} {snapshot.report_jobs_queued}',
        f'onramp_report_jobs{{status="processing"}} {snapshot.report_jobs_processing}',
        "# HELP onramp_event_outbox_pending Number of unpublished outbox events ready to send.",
        "# TYPE onramp_event_outbox_pending gauge",
        f"onramp_event_outbox_pending {snapshot.event_outbox_pending}",
        "# HELP onramp_redis_stream_length Current Redis stream length.",
        "# TYPE onramp_redis_stream_length gauge",
    ]

    for stream, length in sorted(snapshot.stream_lengths.items()):
        lines.append(f'onramp_redis_stream_length{{stream="{stream}"}} {length}')

    lines.extend(
        [
            "# HELP onramp_redis_stream_group_pending Pending messages for a Redis stream consumer group.",
            "# TYPE onramp_redis_stream_group_pending gauge",
        ]
    )
    for metric in snapshot.stream_groups:
        lines.append(
            f'onramp_redis_stream_group_pending{{stream="{metric.stream}",group="{metric.group}"}} {metric.pending}'
        )

    lines.extend(
        [
            "# HELP onramp_redis_stream_group_lag Lag for a Redis stream consumer group.",
            "# TYPE onramp_redis_stream_group_lag gauge",
        ]
    )
    for metric in snapshot.stream_groups:
        lines.append(f'onramp_redis_stream_group_lag{{stream="{metric.stream}",group="{metric.group}"}} {metric.lag}')

    lines.extend(
        [
            "# HELP onramp_metrics_collected_at_seconds Unix timestamp when worker metrics were collected.",
            "# TYPE onramp_metrics_collected_at_seconds gauge",
            f"onramp_metrics_collected_at_seconds {snapshot.collected_at.timestamp():.0f}",
        ]
    )
    return "\n".join(lines) + "\n"


async def _safe_xlen(redis: Redis, stream: str) -> int:
    try:
        return int(await redis.xlen(stream))
    except ResponseError as exc:
        if "no such key" in str(exc).lower():
            return 0
        raise


async def _load_group_metrics(redis: Redis, stream: str, group: str) -> tuple[int, int]:
    try:
        groups = await redis.xinfo_groups(stream)
    except ResponseError as exc:
        if "no such key" in str(exc).lower():
            return 0, 0
        raise

    for item in groups:
        if item.get("name") == group:
            pending = int(item.get("pending", 0) or 0)
            lag = int(item.get("lag", 0) or 0)
            return pending, lag
    return 0, 0
