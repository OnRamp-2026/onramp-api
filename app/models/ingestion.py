from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

IngestionMode = Literal["incremental", "full_scan"]


class IngestionRunCreate(BaseModel):
    mode: IngestionMode


class IngestionRunResponse(BaseModel):
    run_id: UUID
    tenant_id: str
    mode: str
    trigger: str
    status: str
    stage: str
    pages_discovered: int
    pages_processed: int
    pages_indexed: int
    pages_skipped: int
    pages_failed: int
    chunks_indexed: int
    chunks_deleted: int
    started_at: datetime
    finished_at: datetime | None
    error_message: str | None


class IngestionRunListResponse(BaseModel):
    runs: list[IngestionRunResponse] = Field(default_factory=list)
