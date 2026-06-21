from __future__ import annotations

from sqlalchemy import case, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from app.db.models import Report, ReportStatus, TranscriptionWorkflow, WorkflowStatus
from app.models.asset_history import (
    AssetHistoryCounts,
    AssetHistoryItem,
    AssetHistoryListResponse,
    AssetHistoryProgress,
    AssetHistorySource,
    AssetHistoryStatus,
)
from app.models.response import FiveElementsResponse

_FAILED = {
    WorkflowStatus.transcription_failed,
    WorkflowStatus.correction_failed,
    WorkflowStatus.report_failed,
    WorkflowStatus.cancelled,
}


def _status_expression() -> ColumnElement[str]:
    return case(
        (TranscriptionWorkflow.status == WorkflowStatus.deleting, "deleting"),
        (TranscriptionWorkflow.status.in_(_FAILED), "failed"),
        (
            or_(
                Report.status == ReportStatus.published,
                TranscriptionWorkflow.status == WorkflowStatus.published,
            ),
            "completed",
        ),
        (Report.status == ReportStatus.publishing, "processing"),
        (Report.id.is_not(None), "draft"),
        else_="processing",
    )


def _status(workflow: TranscriptionWorkflow, report: Report | None) -> AssetHistoryStatus:
    if workflow.status == WorkflowStatus.deleting:
        return "deleting"
    if workflow.status in _FAILED:
        return "failed"
    if report is not None and report.status == ReportStatus.published:
        return "completed"
    if workflow.status == WorkflowStatus.published:
        return "completed"
    if report is not None and report.status == ReportStatus.publishing:
        return "processing"
    if report is not None:
        return "draft"
    return "processing"


def _report_body(report: Report | None) -> FiveElementsResponse | None:
    if report is None:
        return None
    return FiveElementsResponse(
        situation=report.situation,
        cause=report.cause,
        evidence=report.evidence,
        solution=report.solution,
        infra_context=report.infra_context,
    )


def _item(workflow: TranscriptionWorkflow, report: Report | None) -> AssetHistoryItem:
    processed = workflow.completed_chunks + workflow.failed_chunks
    percent = round((processed / workflow.total_chunks) * 100, 2) if workflow.total_chunks else 0.0
    return AssetHistoryItem(
        asset_id=str(workflow.transcription_id),
        transcription_id=str(workflow.transcription_id),
        report_id=str(report.id) if report is not None else None,
        title=report.title if report is not None else workflow.title,
        category=report.category if report is not None else workflow.category,
        status=_status(workflow, report),
        workflow_status=workflow.status.value,
        confluence_url=report.confluence_url if report is not None else "",
        created_at=workflow.created_at.isoformat(),
        updated_at=max(
            workflow.updated_at, report.updated_at if report is not None else workflow.updated_at
        ).isoformat(),
        source=AssetHistorySource(
            filename=workflow.source_filename,
            content_type=workflow.source_content_type,
            size_bytes=workflow.source_size_bytes,
        ),
        progress=AssetHistoryProgress(
            total_chunks=workflow.total_chunks,
            completed_chunks=workflow.completed_chunks,
            failed_chunks=workflow.failed_chunks,
            percent=percent,
        ),
        report=_report_body(report),
    )


async def list_assets(
    session: AsyncSession,
    *,
    tenant_id: str,
    user_id: str,
    status: AssetHistoryStatus | None = None,
    limit: int = 50,
) -> AssetHistoryListResponse:
    status_expression = _status_expression()
    join_condition = (Report.source_transcription_id == TranscriptionWorkflow.transcription_id) & (
        Report.tenant_id == TranscriptionWorkflow.tenant_id
    )
    owner_condition = (
        TranscriptionWorkflow.tenant_id == tenant_id,
        TranscriptionWorkflow.created_by_user_id == user_id,
    )
    total, processing, draft, deleting, completed, failed = (
        await session.execute(
            select(
                func.count(),
                func.sum(case((status_expression == "processing", 1), else_=0)),
                func.sum(case((status_expression == "draft", 1), else_=0)),
                func.sum(case((status_expression == "deleting", 1), else_=0)),
                func.sum(case((status_expression == "completed", 1), else_=0)),
                func.sum(case((status_expression == "failed", 1), else_=0)),
            )
            .select_from(TranscriptionWorkflow)
            .outerjoin(Report, join_condition)
            .where(*owner_condition)
        )
    ).one()
    counts = AssetHistoryCounts(
        all=total,
        processing=processing or 0,
        draft=draft or 0,
        deleting=deleting or 0,
        completed=completed or 0,
        failed=failed or 0,
    )

    items_statement = select(TranscriptionWorkflow, Report).outerjoin(Report, join_condition).where(*owner_condition)
    if status is not None:
        items_statement = items_statement.where(status_expression == status)
    rows = (await session.execute(items_statement.order_by(TranscriptionWorkflow.updated_at.desc()).limit(limit))).all()
    return AssetHistoryListResponse(
        items=[_item(workflow, report) for workflow, report in rows],
        counts=counts,
    )
