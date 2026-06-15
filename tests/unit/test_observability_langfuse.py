"""Langfuse 관측 설정·팩토리 검증 (#121).

핵심 계약: disabled(기본)면 키 없이도 기동하고 팩토리는 no-op(None)이다.
"""

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
