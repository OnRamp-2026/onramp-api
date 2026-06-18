"""자산화(/v1/asset) HITL 엔드포인트 — 녹취 → 5요소 보고서 → 수정 → Confluence 등록."""

from fastapi import APIRouter

from app.api.deps import CurrentUser
from app.models.request import AssetRequest, AssetUpdateRequest
from app.models.response import AssetApproveResponse, AssetResponse
from app.services import asset_service

router = APIRouter()


@router.post("/asset", response_model=AssetResponse)
async def create_asset(request: AssetRequest, user: CurrentUser) -> AssetResponse:
    """녹취 텍스트 → 5요소 보고서 초안 생성 (status=draft)."""
    return await asset_service.create_report(request)


@router.get("/asset/{report_id}", response_model=AssetResponse)
async def get_asset(report_id: str, user: CurrentUser) -> AssetResponse:
    """초안 조회 (프론트 수정 UI 표시용)."""
    return asset_service.get_report(report_id)


@router.patch("/asset/{report_id}", response_model=AssetResponse)
async def update_asset(report_id: str, update: AssetUpdateRequest, user: CurrentUser) -> AssetResponse:
    """HITL — 사람이 수정한 내용 반영 (partial update)."""
    return asset_service.update_report(report_id, update)


@router.post("/asset/{report_id}/approve", response_model=AssetApproveResponse)
async def approve_asset(report_id: str, user: CurrentUser) -> AssetApproveResponse:
    """수정 완료 → Confluence 등록 (status=published)."""
    return await asset_service.approve_report(report_id)
