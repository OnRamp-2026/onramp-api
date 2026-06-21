from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.base import Base
from app.db.models import Report, ReportStatus, TranscriptionWorkflow, WorkflowStatus
from app.services.asset_history_service import list_assets


@pytest.fixture
async def session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


def workflow(*, user_id: str, status: WorkflowStatus, title: str) -> TranscriptionWorkflow:
    return TranscriptionWorkflow(
        transcription_id=uuid4(),
        tenant_id="tenant-a",
        created_by_user_id=user_id,
        idempotency_key=None,
        status=status,
        source_object_key=f"tenants/tenant-a/{title}.m4a",
        source_filename=f"{title}.m4a",
        source_content_type="audio/mp4",
        source_size_bytes=1024,
        title=title,
        language="ko-KR",
        category="장애대응",
        total_chunks=10,
        completed_chunks=4,
        failed_chunks=0,
    )


def report_for(item: TranscriptionWorkflow, status: ReportStatus) -> Report:
    return Report(
        tenant_id=item.tenant_id,
        source_transcription_id=item.transcription_id,
        title=item.title,
        category=item.category,
        situation="상황",
        cause="원인",
        evidence="근거",
        solution="해결",
        infra_context="환경",
        status=status,
        raw_text_sha256="a" * 64,
        corrected_text_sha256="b" * 64,
        dictionary_version="2026-06-21",
        result_object_key="result.json",
    )


@pytest.mark.asyncio
async def test_list_assets_returns_only_current_user_and_combines_workflow_report_status(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    processing = workflow(user_id="user-a", status=WorkflowStatus.transcribing, title="처리 중")
    draft = workflow(user_id="user-a", status=WorkflowStatus.draft, title="초안")
    completed = workflow(user_id="user-a", status=WorkflowStatus.published, title="완료")
    other = workflow(user_id="user-b", status=WorkflowStatus.transcribing, title="다른 사용자")

    async with session_factory() as session:
        session.add_all(
            [
                processing,
                draft,
                completed,
                other,
                report_for(draft, ReportStatus.draft),
                report_for(completed, ReportStatus.published),
            ]
        )
        await session.commit()

    async with session_factory() as session:
        result = await list_assets(session, tenant_id="tenant-a", user_id="user-a")

    assert {item.title for item in result.items} == {"처리 중", "초안", "완료"}
    assert {item.title: item.status for item in result.items} == {
        "처리 중": "processing",
        "초안": "draft",
        "완료": "completed",
    }
    assert result.counts.model_dump() == {
        "all": 3,
        "processing": 1,
        "draft": 1,
        "completed": 1,
        "failed": 0,
    }
    assert next(item for item in result.items if item.title == "처리 중").report is None
    assert next(item for item in result.items if item.title == "초안").report is not None


@pytest.mark.asyncio
async def test_list_assets_filters_status_after_counting_all_items(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    processing = workflow(user_id="user-a", status=WorkflowStatus.correcting, title="처리 중")
    failed = workflow(user_id="user-a", status=WorkflowStatus.report_failed, title="실패")
    async with session_factory() as session:
        session.add_all([processing, failed])
        await session.commit()

    async with session_factory() as session:
        result = await list_assets(
            session,
            tenant_id="tenant-a",
            user_id="user-a",
            status="failed",
        )

    assert [item.title for item in result.items] == ["실패"]
    assert result.counts.all == 2
    assert result.counts.processing == 1
    assert result.counts.failed == 1


@pytest.mark.asyncio
async def test_list_assets_treats_confluence_publishing_as_processing(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    publishing = workflow(user_id="user-a", status=WorkflowStatus.draft, title="등록 중")
    async with session_factory() as session:
        session.add_all([publishing, report_for(publishing, ReportStatus.publishing)])
        await session.commit()

    async with session_factory() as session:
        result = await list_assets(session, tenant_id="tenant-a", user_id="user-a")

    assert result.items[0].status == "processing"
