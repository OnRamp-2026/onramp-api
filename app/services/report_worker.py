from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import (
    Report,
    ReportJob,
    ReportJobStatus,
    ReportStatus,
    TranscriptionWorkflow,
    WorkflowStatus,
    utcnow,
)
from app.models.response import FiveElementsResponse
from app.services.asset_service import generate_report_content
from app.services.stt_result_client import SttResult, SttResultClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeneratedReport:
    title: str
    report: FiveElementsResponse


ReportGenerator = Callable[[str, str, str], Awaitable[GeneratedReport]]


async def default_report_generator(transcript: str, category: str, title: str) -> GeneratedReport:
    generated = await generate_report_content(transcript, category, title)
    return GeneratedReport(title=generated.title, report=generated.report)


class ReportIntegrityError(ValueError):
    pass


class ReportWorker:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        stt_client: SttResultClient,
        generator: ReportGenerator = default_report_generator,
    ) -> None:
        self.session_factory = session_factory
        self.stt_client = stt_client
        self.generator = generator

    async def process_next(self) -> bool:
        async with self.session_factory() as session:
            job = await session.scalar(
                select(ReportJob)
                .where(ReportJob.status == ReportJobStatus.queued)
                .order_by(ReportJob.created_at)
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            if job is None:
                return False
            workflow = await self._workflow(session, job.source_transcription_id)
            job.status = ReportJobStatus.processing
            workflow.status = WorkflowStatus.report_processing
            await session.commit()

        try:
            result = await self.stt_client.get_result(job.source_transcription_id)
            self._validate_result(job, result)
            generated = await self.generator(result.corrected.text, workflow.category, workflow.title)
        except Exception as exc:
            logger.exception("report generation failed", extra={"report_job_id": str(job.id)})
            async with self.session_factory() as session, session.begin():
                persisted_job = await session.get(ReportJob, job.id)
                persisted_workflow = await self._workflow(session, job.source_transcription_id)
                if persisted_job is not None:
                    persisted_job.status = ReportJobStatus.failed
                    persisted_job.last_error = str(exc)[:2000]
                    persisted_job.updated_at = utcnow()
                persisted_workflow.status = WorkflowStatus.report_failed
                persisted_workflow.updated_at = utcnow()
            return True

        async with self.session_factory() as session, session.begin():
            persisted_job = await session.get(ReportJob, job.id)
            persisted_workflow = await self._workflow(session, job.source_transcription_id)
            existing = await session.scalar(
                select(Report).where(
                    Report.tenant_id == job.tenant_id,
                    Report.source_transcription_id == job.source_transcription_id,
                )
            )
            report = existing or Report(
                tenant_id=job.tenant_id,
                source_transcription_id=job.source_transcription_id,
                title=generated.title,
                category=persisted_workflow.category,
                situation=generated.report.situation,
                cause=generated.report.cause,
                evidence=generated.report.evidence,
                solution=generated.report.solution,
                infra_context=generated.report.infra_context,
                status=ReportStatus.draft,
                raw_text_sha256=job.raw_text_sha256,
                corrected_text_sha256=job.corrected_text_sha256,
                dictionary_version=job.dictionary_version,
                result_object_key=job.result_object_key,
            )
            if existing is None:
                session.add(report)
                await session.flush()
            if persisted_job is not None:
                persisted_job.status = ReportJobStatus.completed
                persisted_job.last_error = None
                persisted_job.updated_at = utcnow()
            persisted_workflow.status = WorkflowStatus.draft
            persisted_workflow.report_id = report.id
            persisted_workflow.updated_at = utcnow()
        return True

    @staticmethod
    async def _workflow(session: AsyncSession, transcription_id: UUID) -> TranscriptionWorkflow:
        workflow = await session.scalar(
            select(TranscriptionWorkflow).where(TranscriptionWorkflow.transcription_id == transcription_id)
        )
        if workflow is None:
            raise ValueError("transcription workflow not found")
        return workflow

    @staticmethod
    def _validate_result(job: ReportJob, result: SttResult) -> None:
        if result.transcription_id != job.source_transcription_id:
            raise ReportIntegrityError("transcription id mismatch")
        if result.tenant_id != job.tenant_id:
            raise ReportIntegrityError("tenant mismatch")
        if result.raw.text_sha256 != job.raw_text_sha256:
            raise ReportIntegrityError("raw checksum mismatch")
        if result.corrected.text_sha256 != job.corrected_text_sha256:
            raise ReportIntegrityError("corrected checksum mismatch")
        if result.dictionary_version != job.dictionary_version:
            raise ReportIntegrityError("dictionary version mismatch")
