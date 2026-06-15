from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter

from app.api.deps import CurrentTenant, DatabaseSession
from app.models.request import AssetUpdateRequest
from app.models.response import AssetApproveResponse, AssetResponse
from app.services import report_service

router = APIRouter(prefix="/reports")


@router.get("/{report_id}", response_model=AssetResponse)
async def get_report(
    report_id: UUID,
    session: DatabaseSession,
    tenant_id: CurrentTenant,
) -> AssetResponse:
    report = await report_service.get_report(session, tenant_id=tenant_id, report_id=report_id)
    return report_service.report_response(report)


@router.patch("/{report_id}", response_model=AssetResponse)
async def update_report(
    report_id: UUID,
    update: AssetUpdateRequest,
    session: DatabaseSession,
    tenant_id: CurrentTenant,
) -> AssetResponse:
    report = await report_service.update_report(
        session,
        tenant_id=tenant_id,
        report_id=report_id,
        update=update,
    )
    await session.commit()
    return report_service.report_response(report)


@router.post("/{report_id}/approve", response_model=AssetApproveResponse)
async def approve_report(
    report_id: UUID,
    session: DatabaseSession,
    tenant_id: CurrentTenant,
) -> AssetApproveResponse:
    response = await report_service.approve_report(
        session,
        tenant_id=tenant_id,
        report_id=report_id,
    )
    await session.commit()
    return response
