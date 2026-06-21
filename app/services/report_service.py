from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.confluence import ConfluenceClient
from app.db.models import Report, ReportStatus, TranscriptionWorkflow, WorkflowStatus, utcnow
from app.middleware.error_handler import OnRampError
from app.models.request import AssetUpdateRequest
from app.models.response import AssetApproveResponse, AssetResponse, FiveElementsResponse
from app.services.asset_service import _five_elements_to_wiki


async def get_report(
    session: AsyncSession,
    *,
    tenant_id: str,
    report_id: UUID,
    user_id: str | None = None,
) -> Report:
    statement = select(Report).where(
        Report.id == report_id,
        Report.tenant_id == tenant_id,
    )
    if user_id is not None:
        statement = statement.join(
            TranscriptionWorkflow,
            (TranscriptionWorkflow.transcription_id == Report.source_transcription_id)
            & (TranscriptionWorkflow.tenant_id == Report.tenant_id),
        ).where(TranscriptionWorkflow.created_by_user_id == user_id)
    report = await session.scalar(statement)
    if report is None:
        raise OnRampError("보고서를 찾을 수 없습니다", status_code=404)
    return report


async def update_report(
    session: AsyncSession,
    *,
    tenant_id: str,
    user_id: str | None = None,
    report_id: UUID,
    update: AssetUpdateRequest,
) -> Report:
    report = await get_report(session, tenant_id=tenant_id, user_id=user_id, report_id=report_id)
    if report.status != ReportStatus.draft:
        raise OnRampError("등록 중이거나 이미 등록된 보고서는 수정할 수 없습니다", status_code=409)
    for field in ("title", "category", "situation", "cause", "evidence", "solution", "infra_context"):
        value = getattr(update, field)
        if value is not None:
            setattr(report, field, value)
    report.updated_at = utcnow()
    await session.flush()
    return report


async def approve_report(
    session: AsyncSession,
    *,
    tenant_id: str,
    user_id: str | None = None,
    report_id: UUID,
    confluence: ConfluenceClient | None = None,
) -> AssetApproveResponse:
    statement = (
        select(Report)
        .where(
            Report.id == report_id,
            Report.tenant_id == tenant_id,
        )
        .with_for_update()
    )
    if user_id is not None:
        statement = statement.join(
            TranscriptionWorkflow,
            (TranscriptionWorkflow.transcription_id == Report.source_transcription_id)
            & (TranscriptionWorkflow.tenant_id == Report.tenant_id),
        ).where(TranscriptionWorkflow.created_by_user_id == user_id)
    report = await session.scalar(statement)
    if report is None:
        raise OnRampError("보고서를 찾을 수 없습니다", status_code=404)
    if report.status == ReportStatus.published:
        raise OnRampError("이미 등록된 보고서입니다", status_code=409)
    if report.status == ReportStatus.publishing:
        raise OnRampError("보고서 등록이 진행 중이거나 확인이 필요합니다", status_code=409)

    report.status = ReportStatus.publishing
    report.updated_at = utcnow()
    await session.commit()

    try:
        page = await (confluence or ConfluenceClient()).create_page(
            title=report.title,
            html=_five_elements_to_wiki(_five_elements(report), report.category),
        )
    except Exception:
        persisted = await session.get(Report, report_id, with_for_update=True)
        if persisted is not None and persisted.status == ReportStatus.publishing:
            persisted.status = ReportStatus.draft
            persisted.updated_at = utcnow()
            await session.commit()
        raise

    persisted = await session.get(Report, report_id, with_for_update=True)
    if persisted is None:
        raise OnRampError("보고서를 찾을 수 없습니다", status_code=404)
    report = persisted
    report.status = ReportStatus.published
    report.confluence_page_id = page.page_id
    report.confluence_url = page.url
    report.updated_at = utcnow()
    workflow = await session.scalar(
        select(TranscriptionWorkflow).where(
            TranscriptionWorkflow.tenant_id == tenant_id,
            TranscriptionWorkflow.transcription_id == report.source_transcription_id,
        )
    )
    if workflow is not None:
        workflow.status = WorkflowStatus.published
        workflow.updated_at = utcnow()
    await session.flush()
    return AssetApproveResponse(
        report_id=str(report.id),
        status=ReportStatus.published.value,
        confluence_url=report.confluence_url,
    )


def report_response(report: Report) -> AssetResponse:
    return AssetResponse(
        report_id=str(report.id),
        title=report.title,
        report=_five_elements(report),
        category=report.category,
        status=report.status.value,
        confluence_url=report.confluence_url,
        created_at=report.created_at.isoformat(),
        updated_at=report.updated_at.isoformat(),
    )


def _five_elements(report: Report) -> FiveElementsResponse:
    return FiveElementsResponse(
        situation=report.situation,
        cause=report.cause,
        evidence=report.evidence,
        solution=report.solution,
        infra_context=report.infra_context,
    )
