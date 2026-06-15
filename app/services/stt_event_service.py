from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import (
    EventInbox,
    ReportJob,
    ReportJobStatus,
    TranscriptionWorkflow,
    WorkflowStatus,
    utcnow,
)
from app.queue.constants import (
    COMPLETED_EVENT_TYPE,
    PROGRESS_EVENT_TYPE,
    REPORT_EVENT_GROUP,
    TRANSCRIPT_COMPLETED_EVENT_TYPE,
    TRANSCRIPT_OBSERVER_GROUP,
    WORKFLOW_UPDATER_GROUP,
)
from app.queue.events import ProgressUpdated, StreamEnvelope, TranscriptCompleted, TranscriptionCompleted


class SttEventService:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self.session_factory = session_factory

    async def process(self, envelope: StreamEnvelope) -> None:
        if envelope.event_type == PROGRESS_EVENT_TYPE:
            await self._process_progress(envelope)
        elif envelope.event_type == TRANSCRIPT_COMPLETED_EVENT_TYPE:
            await self._process_transcript_completed(envelope)
        elif envelope.event_type == COMPLETED_EVENT_TYPE:
            await self._process_completed(envelope)
        else:
            raise UnrecoverableSttEventError(f"unsupported STT event type: {envelope.event_type}")

    async def _process_progress(self, envelope: StreamEnvelope) -> None:
        payload = ProgressUpdated.model_validate(envelope.payload)
        async with self.session_factory() as session, session.begin():
            if await self._is_processed(session, TRANSCRIPT_OBSERVER_GROUP, envelope.event_id):
                return
            workflow = await self._workflow(session, payload.transcription_id)
            self._verify_tenant(workflow, payload.tenant_id)
            workflow.total_chunks = payload.total_chunks
            workflow.completed_chunks = payload.completed_chunks
            workflow.failed_chunks = payload.failed_chunks
            if not self._report_stage_started(workflow):
                workflow.status = self._workflow_status(payload.status, workflow)
                workflow.updated_at = payload.occurred_at
            self._mark_processed(session, TRANSCRIPT_OBSERVER_GROUP, envelope.event_id, str(workflow.id))

    async def _process_transcript_completed(self, envelope: StreamEnvelope) -> None:
        payload = TranscriptCompleted.model_validate(envelope.payload)
        async with self.session_factory() as session, session.begin():
            if await self._is_processed(session, WORKFLOW_UPDATER_GROUP, envelope.event_id):
                return
            workflow = await self._workflow(session, payload.transcription_id)
            self._verify_tenant(workflow, payload.tenant_id)
            if not self._report_stage_started(workflow):
                workflow.status = WorkflowStatus.transcript_completed
                workflow.transcript_completed_received_at = utcnow()
                workflow.updated_at = utcnow()
            self._mark_processed(session, WORKFLOW_UPDATER_GROUP, envelope.event_id, str(workflow.id))

    async def _process_completed(self, envelope: StreamEnvelope) -> None:
        payload = TranscriptionCompleted.model_validate(envelope.payload)
        async with self.session_factory() as session, session.begin():
            if await self._is_processed(session, REPORT_EVENT_GROUP, envelope.event_id):
                return
            workflow = await self._workflow(session, payload.transcription_id)
            self._verify_tenant(workflow, payload.tenant_id)
            existing = await session.scalar(
                select(ReportJob).where(
                    ReportJob.tenant_id == payload.tenant_id,
                    ReportJob.source_transcription_id == payload.transcription_id,
                )
            )
            if existing is None:
                existing = ReportJob(
                    tenant_id=payload.tenant_id,
                    source_transcription_id=payload.transcription_id,
                    status=ReportJobStatus.queued,
                    raw_text_sha256=payload.raw_text_sha256,
                    corrected_text_sha256=payload.corrected_text_sha256,
                    dictionary_version=payload.dictionary_version,
                    result_object_key=payload.result_object_key,
                )
                session.add(existing)
                await session.flush()
            if not self._report_stage_started(workflow):
                workflow.status = WorkflowStatus.report_queued
                workflow.updated_at = utcnow()
            self._mark_processed(session, REPORT_EVENT_GROUP, envelope.event_id, str(existing.id))

    @staticmethod
    async def _workflow(session: AsyncSession, transcription_id) -> TranscriptionWorkflow:  # type: ignore[no-untyped-def]
        workflow = await session.scalar(
            select(TranscriptionWorkflow)
            .where(TranscriptionWorkflow.transcription_id == transcription_id)
            .with_for_update()
        )
        if workflow is None:
            raise ValueError("transcription workflow not found")
        return workflow

    @staticmethod
    def _verify_tenant(workflow: TranscriptionWorkflow, tenant_id: str) -> None:
        if workflow.tenant_id != tenant_id:
            raise UnrecoverableSttEventError("tenant mismatch")

    @staticmethod
    async def _is_processed(session: AsyncSession, consumer_group: str, event_id: str) -> bool:
        return await session.get(EventInbox, (consumer_group, event_id)) is not None

    @staticmethod
    def _mark_processed(
        session: AsyncSession,
        consumer_group: str,
        event_id: str,
        result_reference: str,
    ) -> None:
        session.add(
            EventInbox(
                consumer_group=consumer_group,
                event_id=event_id,
                result_reference=result_reference,
            )
        )

    @staticmethod
    def _workflow_status(status: str, workflow: TranscriptionWorkflow) -> WorkflowStatus:
        if status == "failed":
            return (
                WorkflowStatus.correction_failed
                if workflow.transcript_completed_received_at is not None
                else WorkflowStatus.transcription_failed
            )
        try:
            return WorkflowStatus(status)
        except ValueError:
            return workflow.status

    @staticmethod
    def _report_stage_started(workflow: TranscriptionWorkflow) -> bool:
        return workflow.status in {
            WorkflowStatus.report_queued,
            WorkflowStatus.report_processing,
            WorkflowStatus.draft,
            WorkflowStatus.published,
            WorkflowStatus.report_failed,
            WorkflowStatus.cancelled,
        }


class UnrecoverableSttEventError(ValueError):
    pass
