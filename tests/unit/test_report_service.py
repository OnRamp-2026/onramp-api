from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.base import Base
from app.db.models import Report, ReportStatus
from app.middleware.error_handler import OnRampError
from app.models.request import AssetUpdateRequest
from app.services.report_service import get_report, report_response, update_report


@pytest.fixture
async def session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


async def create_report(session_factory: async_sessionmaker[AsyncSession]) -> Report:
    report = Report(
        tenant_id="tenant-a",
        source_transcription_id=uuid4(),
        title="기존 제목",
        category="회의록",
        situation="상황",
        cause="원인",
        evidence="근거",
        solution="해결",
        infra_context="환경",
        status=ReportStatus.draft,
        raw_text_sha256="a" * 64,
        corrected_text_sha256="b" * 64,
        dictionary_version="2026-06-14",
        result_object_key="tenants/tenant-a/result.json",
    )
    async with session_factory() as session:
        session.add(report)
        await session.commit()
    return report


@pytest.mark.asyncio
async def test_report_is_tenant_scoped_and_patchable(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    report = await create_report(session_factory)
    async with session_factory() as session:
        updated = await update_report(
            session,
            tenant_id="tenant-a",
            report_id=report.id,
            update=AssetUpdateRequest(title="수정 제목", situation="수정 상황"),
        )
        await session.commit()
        response = report_response(updated)
    assert response.title == "수정 제목"
    assert response.report.situation == "수정 상황"
    assert response.report.cause == "원인"

    async with session_factory() as session:
        with pytest.raises(OnRampError):
            await get_report(session, tenant_id="tenant-b", report_id=report.id)
