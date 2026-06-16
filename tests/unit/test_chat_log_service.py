"""chat_service의 운영/eval 로그 저장 동작을 검증한다 (#172)."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.agents.state import AnswerabilityStatus, Domain, FiveElements, SourceDocument, UseCase
from app.db.base import Base
from app.db.models import ChatLog
from app.models.request import ChatRequest


@pytest.fixture
async def session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        await engine.dispose()


def _graph_state() -> dict:
    return {
        "answer": FiveElements(
            situation="상황",
            cause="원인",
            evidence="근거",
            solution="해결",
            infra_context="맥락",
        ),
        "sources": [
            SourceDocument(
                title="장애 대응 가이드",
                url="http://example.test/runbook",
                space_key="OnRamp",
                content_snippet="CrashLoopBackOff 대응 절차",
                score=0.91,
                site="kubernetes",
                product_version="v1.33",
            )
        ],
        "answerability_status": AnswerabilityStatus.ANSWERABLE,
        "answerability_reason": "근거 충분",
        "domain": Domain.INCIDENT,
        "use_case": UseCase.SEARCH,
        "model": "gpt-4o",
    }


@pytest.mark.asyncio
async def test_chat_persists_chat_log_when_session_is_provided(monkeypatch, session_factory):
    import app.services.chat_service as cs

    async def fake_ainvoke(state, config=None):
        return _graph_state()

    monkeypatch.setattr(cs.compiled_graph, "ainvoke", fake_ainvoke)
    monkeypatch.setattr("app.observability.langfuse.get_callback_handler", lambda: None)

    async with session_factory() as session:
        response = await cs.chat(
            ChatRequest(query="EKS 장애 대응", model="gpt-4o"),
            tenant_id="tenant-a",
            session=session,
        )

        row = (await session.execute(select(ChatLog))).scalar_one()

    assert response.answerability_status == "answerable"
    assert row.tenant_id == "tenant-a"
    assert row.query == "EKS 장애 대응"
    assert row.domain == "incident"
    assert row.use_case == "검색"
    assert row.answerability_status == "answerable"
    assert row.answerability_reason == "근거 충분"
    assert row.model_used == "gpt-4o"
    assert row.source_count == 1
    assert row.sources is not None
    assert row.sources[0]["title"] == "장애 대응 가이드"
    assert row.latency_ms is not None and row.latency_ms >= 0


@pytest.mark.asyncio
async def test_chat_log_failure_does_not_block_response(monkeypatch):
    import app.services.chat_service as cs

    async def fake_ainvoke(state, config=None):
        return _graph_state()

    class FailingSession:
        def add(self, obj):
            raise RuntimeError("db down")

        async def rollback(self):
            self.rolled_back = True

    failing_session = FailingSession()
    monkeypatch.setattr(cs.compiled_graph, "ainvoke", fake_ainvoke)
    monkeypatch.setattr("app.observability.langfuse.get_callback_handler", lambda: None)

    response = await cs.chat(
        ChatRequest(query="EKS 장애 대응"),
        tenant_id="tenant-a",
        session=failing_session,
    )

    assert response.answerability_status == "answerable"
    assert failing_session.rolled_back is True
