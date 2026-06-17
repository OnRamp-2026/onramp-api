"""DocumentDomainClassifier 단위 테스트 — call_llm을 모킹해 파싱·검증·fallback을 검사."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from app.rag import llm_classifier
from app.rag.llm_classifier import DocumentDomainClassifier


def _patch_call_llm(monkeypatch: pytest.MonkeyPatch, impl: Callable[..., object]) -> None:
    async def fake_call_llm(*_args: object, **_kwargs: object) -> object:
        return impl()

    monkeypatch.setattr(llm_classifier, "call_llm", fake_call_llm)


async def test_classify_returns_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_call_llm(monkeypatch, lambda: '{"domain": "meeting_note", "secondary": "planning", "confidence": 0.9}')

    result = await DocumentDomainClassifier().classify("스프린트 회의", "결정사항과 액션아이템 논의")

    assert result is not None
    assert result.domain == "meeting_note"
    assert result.secondary == "planning"
    assert result.confidence == 0.9


async def test_classify_unknown_domain_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_call_llm(monkeypatch, lambda: '{"domain": "weather", "confidence": 1.0}')

    assert await DocumentDomainClassifier().classify("t", "b") is None


async def test_classify_llm_error_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> object:
        raise RuntimeError("LLM down")

    _patch_call_llm(monkeypatch, boom)

    assert await DocumentDomainClassifier().classify("t", "b") is None


async def test_classify_invalid_json_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_call_llm(monkeypatch, lambda: "not json at all")

    assert await DocumentDomainClassifier().classify("t", "b") is None


async def test_classify_non_object_json_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    # 유효 JSON이지만 객체가 아닌 경우(list/str) data.get가 터지지 않고 룰 fallback(None)이어야 한다.
    _patch_call_llm(monkeypatch, lambda: '["manual"]')

    assert await DocumentDomainClassifier().classify("t", "b") is None


async def test_classify_drops_secondary_equal_to_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_call_llm(monkeypatch, lambda: '{"domain": "manual", "secondary": "manual", "confidence": 0.5}')

    result = await DocumentDomainClassifier().classify("t", "b")

    assert result is not None
    assert result.secondary == ""  # primary와 같으면 버린다


async def test_classify_clamps_and_defaults_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_call_llm(monkeypatch, lambda: '{"domain": "incident", "secondary": "bogus", "confidence": 9}')

    result = await DocumentDomainClassifier().classify("t", "b")

    assert result is not None
    assert result.confidence == 1.0  # 0~1로 클램프
    assert result.secondary == ""  # 화이트리스트 밖 secondary는 버린다
