from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from app.models.response import FiveElementsResponse

AssetHistoryStatus = Literal["processing", "draft", "completed", "failed"]


class AssetHistorySource(BaseModel):
    filename: str
    content_type: str
    size_bytes: int


class AssetHistoryProgress(BaseModel):
    total_chunks: int
    completed_chunks: int
    failed_chunks: int
    percent: float


class AssetHistoryItem(BaseModel):
    asset_id: str
    transcription_id: str
    report_id: str | None
    title: str
    category: str
    status: AssetHistoryStatus
    workflow_status: str
    confluence_url: str
    created_at: str
    updated_at: str
    source: AssetHistorySource
    progress: AssetHistoryProgress
    report: FiveElementsResponse | None


class AssetHistoryCounts(BaseModel):
    all: int = 0
    processing: int = 0
    draft: int = 0
    completed: int = 0
    failed: int = 0


class AssetHistoryListResponse(BaseModel):
    items: list[AssetHistoryItem]
    counts: AssetHistoryCounts
