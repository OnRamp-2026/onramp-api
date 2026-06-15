"""Langfuse 관측 설정·팩토리 검증 (#121).

핵심 계약: disabled(기본)면 키 없이도 기동하고 팩토리는 no-op(None)이다.
"""

import sys
import types

import pytest
from pydantic import SecretStr, ValidationError

from app.config import Settings


def test_langfuse_disabled_by_default():
    assert Settings().langfuse_enabled is False


def test_langfuse_enabled_requires_all_keys():
    # 켜놓고 키/host 누락 → fail-fast
    with pytest.raises(ValidationError):
        Settings(langfuse_enabled=True)
    with pytest.raises(ValidationError):
        # host 누락
        Settings(
            langfuse_enabled=True,
            langfuse_public_key="pk-lf-x",
            langfuse_secret_key=SecretStr("sk-lf-x"),
        )


def test_langfuse_enabled_ok_with_all_keys():
    s = Settings(
        langfuse_enabled=True,
        langfuse_public_key="pk-lf-x",
        langfuse_secret_key=SecretStr("sk-lf-x"),
        langfuse_host="http://langfuse.local",
    )
    assert s.langfuse_enabled is True


def test_langfuse_secret_key_masked_in_repr():
    s = Settings(langfuse_secret_key=SecretStr("sk-lf-super-secret"))
    assert "sk-lf-super-secret" not in repr(s)


def test_factory_is_noop_when_disabled(monkeypatch):
    import app.observability.langfuse as lf

    lf.get_langfuse_client.cache_clear()
    lf.get_callback_handler.cache_clear()
    monkeypatch.setattr(lf, "get_settings", lambda: Settings(langfuse_enabled=False))

    assert lf.get_langfuse_client() is None
    assert lf.get_callback_handler() is None
    assert lf.is_enabled() is False

    lf.get_langfuse_client.cache_clear()
    lf.get_callback_handler.cache_clear()


def test_factory_returns_instances_when_enabled(monkeypatch):
    """enabled 경로: 가짜 langfuse 모듈을 주입해 client/handler가 실제로 생성되는지 검증.

    실제 SDK·네트워크 없이 monkeypatch로만 확인 (E2 계측 연동 회귀 방지).
    """
    import app.observability.langfuse as lf

    lf.get_langfuse_client.cache_clear()
    lf.get_callback_handler.cache_clear()

    enabled = Settings(
        langfuse_enabled=True,
        langfuse_public_key="pk-lf-x",
        langfuse_secret_key=SecretStr("sk-lf-x"),
        langfuse_host="http://langfuse.local",
    )
    monkeypatch.setattr(lf, "get_settings", lambda: enabled)

    # 가짜 langfuse / langfuse.langchain 모듈 주입 (lazy import가 이걸 집어든다)
    captured: dict = {}

    class FakeLangfuse:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeHandler:
        pass

    fake_langfuse = types.ModuleType("langfuse")
    fake_langfuse.Langfuse = FakeLangfuse  # type: ignore[attr-defined]
    fake_langchain = types.ModuleType("langfuse.langchain")
    fake_langchain.CallbackHandler = FakeHandler  # type: ignore[attr-defined]
    fake_langfuse.langchain = fake_langchain  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langfuse", fake_langfuse)
    monkeypatch.setitem(sys.modules, "langfuse.langchain", fake_langchain)

    assert isinstance(lf.get_langfuse_client(), FakeLangfuse)
    assert isinstance(lf.get_callback_handler(), FakeHandler)
    assert lf.is_enabled() is True
    # 시크릿은 평문으로 풀려(get_secret_value) 전달돼야 한다
    assert captured == {
        "public_key": "pk-lf-x",
        "secret_key": "sk-lf-x",
        "host": "http://langfuse.local",
    }

    lf.get_langfuse_client.cache_clear()
    lf.get_callback_handler.cache_clear()


def test_run_config_empty_when_disabled(monkeypatch):
    import app.observability.langfuse as lf

    monkeypatch.setattr(lf, "get_callback_handler", lambda: None)
    assert lf.langfuse_run_config(request_id="r", model="m") == {}


def test_run_config_has_callbacks_and_metadata_when_enabled(monkeypatch):
    import app.observability.langfuse as lf

    handler = object()
    monkeypatch.setattr(lf, "get_callback_handler", lambda: handler)

    cfg = lf.langfuse_run_config(request_id="rid", tenant="tenant1", model="gpt-4o", tags=["incident"])

    assert cfg["callbacks"] == [handler]
    md = cfg["metadata"]
    assert md["langfuse_user_id"] == "tenant1"
    assert md["langfuse_session_id"] == "rid"  # conversation 부재 → request_id 대체
    assert md["request_id"] == "rid"
    assert md["langfuse_tags"] == ["gpt-4o", "incident"]


def test_run_config_session_id_overrides_request_id(monkeypatch):
    import app.observability.langfuse as lf

    monkeypatch.setattr(lf, "get_callback_handler", lambda: object())
    cfg = lf.langfuse_run_config(request_id="rid", session_id="conv-9")
    assert cfg["metadata"]["langfuse_session_id"] == "conv-9"
