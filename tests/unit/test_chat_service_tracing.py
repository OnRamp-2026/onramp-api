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


@pytest.mark.asyncio
async def test_chat_wraps_invoke_in_root_span_when_enabled(monkeypatch):
    """langfuse 활성 시 chat()이 그래프 invoke를 루트 span으로 감싸고 output을 기록한다."""
    import app.observability.langfuse as lf
    import app.services.chat_service as cs

    async def fake_ainvoke(state, config=None):
        return {}

    monkeypatch.setattr(cs.compiled_graph, "ainvoke", fake_ainvoke)

    events: list = []

    class FakeRoot:
        def update(self, **kw):
            events.append(("update", kw))

    class FakeCM:
        def __enter__(self):
            events.append(("enter", None))
            return FakeRoot()

        def __exit__(self, *a):
            events.append(("exit", a[0]))
            return False

    class FakeClient:
        def start_as_current_observation(self, **kw):
            events.append(("start", kw))
            return FakeCM()

    monkeypatch.setattr(lf, "get_langfuse_client", lambda: FakeClient())
    monkeypatch.setattr(lf, "get_callback_handler", lambda: None)

    await cs.chat(ChatRequest(query="안녕"))

    kinds = [e[0] for e in events]
    assert kinds[0] == "start" and kinds[1] == "enter" and kinds[-1] == "exit"
    assert events[0][1]["as_type"] == "span"
    assert "update" in kinds  # output 기록


@pytest.mark.asyncio
async def test_chat_records_trust_score_overall(monkeypatch):
    """state['trust_score'](TrustScore 객체)의 overall이 trust_score score로 부착된다 (#137)."""
    import app.observability.langfuse as lf
    import app.services.chat_service as cs

    class _TS:
        overall = 0.83

    async def fake_ainvoke(state, config=None):
        return {"trust_score": _TS()}

    monkeypatch.setattr(cs.compiled_graph, "ainvoke", fake_ainvoke)

    scored: dict = {}

    class FakeRoot:
        def update(self, **kw):
            pass

    class FakeCM:
        def __enter__(self):
            return FakeRoot()

        def __exit__(self, *a):
            return False

    class FakeClient:
        def start_as_current_observation(self, **kw):
            return FakeCM()

        def score_current_trace(self, **kw):
            scored.update(kw)

        def get_current_trace_id(self):
            return "tid-1"

    monkeypatch.setattr(lf, "get_langfuse_client", lambda: FakeClient())
    monkeypatch.setattr(lf, "get_callback_handler", lambda: None)

    resp = await cs.chat(ChatRequest(query="안녕"))

    assert scored.get("name") == "trust_score"
    assert scored.get("value") == 0.83
    assert resp.trace_id == "tid-1"
