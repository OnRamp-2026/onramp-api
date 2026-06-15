"""chat_service가 Langfuse run config를 그래프에 전달하는지 검증 (#124).

그래프/핸들러를 monkeypatch해 네트워크·LLM 없이 config forwarding만 본다.
"""

import pytest

from app.models.request import ChatRequest


@pytest.mark.asyncio
async def test_chat_forwards_langfuse_config_when_enabled(monkeypatch):
    import app.services.chat_service as cs
    from app.middleware.request_id import request_id_var

    captured: dict = {}

    async def fake_ainvoke(state, config=None):
        captured["config"] = config
        return {}

    monkeypatch.setattr(cs.compiled_graph, "ainvoke", fake_ainvoke)
    handler = object()
    monkeypatch.setattr("app.observability.langfuse.get_callback_handler", lambda: handler)

    request_id_var.set("rid-123")
    await cs.chat(ChatRequest(query="안녕", model="gpt-4o"))

    cfg = captured["config"]
    assert cfg is not None
    assert cfg["callbacks"] == [handler]
    assert cfg["metadata"]["request_id"] == "rid-123"
    assert cfg["metadata"]["langfuse_session_id"] == "rid-123"
    assert "gpt-4o" in cfg["metadata"]["langfuse_tags"]


@pytest.mark.asyncio
async def test_chat_passes_no_config_when_disabled(monkeypatch):
    import app.services.chat_service as cs

    captured: dict = {}

    async def fake_ainvoke(state, config=None):
        captured["config"] = config
        return {}

    monkeypatch.setattr(cs.compiled_graph, "ainvoke", fake_ainvoke)
    monkeypatch.setattr("app.observability.langfuse.get_callback_handler", lambda: None)

    await cs.chat(ChatRequest(query="안녕"))

    # 비활성 → config=None (기존 동작과 100% 동일)
    assert captured["config"] is None
