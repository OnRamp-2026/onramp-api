"""POST /v1/chat/feedback 엔드포인트 — user_feedback score 기록 (#133)."""

import pytest

from app.models.request import FeedbackRequest


@pytest.mark.asyncio
async def test_chat_feedback_records_score(monkeypatch):
    import app.api.v1.chat as chat_api

    captured: dict = {}

    def fake_create(**kw):
        captured.update(kw)
        return True

    monkeypatch.setattr(chat_api, "create_trace_score", fake_create)

    res = await chat_api.chat_feedback(FeedbackRequest(trace_id="t1", value=1.0, comment="good"))

    assert res == {"recorded": True}
    assert captured["trace_id"] == "t1"
    assert captured["name"] == "user_feedback"
    assert captured["value"] == 1.0
    assert captured["comment"] == "good"


@pytest.mark.asyncio
async def test_chat_feedback_noop_returns_false(monkeypatch):
    import app.api.v1.chat as chat_api

    monkeypatch.setattr(chat_api, "create_trace_score", lambda **kw: False)
    res = await chat_api.chat_feedback(FeedbackRequest(trace_id="t1", value=0.0))
    assert res == {"recorded": False}


def test_feedback_request_value_range():
    from pydantic import ValidationError

    FeedbackRequest(trace_id="t", value=0.0)
    FeedbackRequest(trace_id="t", value=1.0)
    with pytest.raises(ValidationError):
        FeedbackRequest(trace_id="t", value=1.5)
    with pytest.raises(ValidationError):
        FeedbackRequest(trace_id="", value=1.0)
