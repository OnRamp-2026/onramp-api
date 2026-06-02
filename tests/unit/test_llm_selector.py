"""LLM Selector 단위 테스트 (실제 LLM 호출 없이 provider 라우팅·에러 검증)."""

import pytest

from app.config import Settings
from app.middleware.error_handler import LLMError
from app.services import llm_selector
from app.services.llm_selector import call_llm, resolve_provider


class _FakeCompletions:
    def __init__(self, content: str = "응답") -> None:
        self.content = content
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        message = type("M", (), {"content": self.content})()
        choice = type("C", (), {"message": message})()
        return type("R", (), {"choices": [choice]})()


class _FakeClient:
    def __init__(self, content: str = "응답") -> None:
        self.chat = type("Chat", (), {"completions": _FakeCompletions(content)})()


@pytest.fixture(autouse=True)
def _reset_clients():
    llm_selector.reset_clients()
    yield
    llm_selector.reset_clients()


# ── resolve_provider (model 이름 → provider) ──
def test_resolve_provider_by_model_name():
    s = Settings(llm_provider="")
    assert resolve_provider("gpt-4o-mini", s) == "openai"
    assert resolve_provider("azure-gpt4", s) == "azure"
    assert resolve_provider("mistral-7b", s) == "self_hosted"


def test_resolve_provider_from_config_when_model_empty():
    assert resolve_provider("", Settings(llm_provider="azure")) == "azure"


def test_resolve_provider_default_openai():
    # model·config 모두 비면 openai 기본
    assert resolve_provider("", Settings(llm_provider="")) == "openai"


# ── 에러 경로 ──
@pytest.mark.asyncio
async def test_call_llm_openai_no_key_raises():
    s = Settings(llm_provider="openai", openai_api_key="")
    with pytest.raises(LLMError):  # 에러 정규화 회귀를 잡도록 LLMError로 고정
        await call_llm("sys", "user", model="gpt-4o-mini", settings=s)


@pytest.mark.asyncio
async def test_call_llm_azure_no_config_raises():
    s = Settings(llm_provider="azure", azure_openai_endpoint="", azure_openai_api_key="")
    with pytest.raises(LLMError):
        await call_llm("sys", "user", model="azure-gpt4", settings=s)


@pytest.mark.asyncio
async def test_call_llm_self_hosted_no_url_raises():
    s = Settings(llm_provider="self_hosted", self_hosted_llm_url="")
    with pytest.raises(LLMError):
        await call_llm("sys", "user", model="mistral-7b", settings=s)


# ── 정상 호출 (openai client mock) ──
@pytest.mark.asyncio
async def test_call_llm_openai_success_and_json_mode(monkeypatch):
    fake = _FakeClient("hello")
    monkeypatch.setattr(llm_selector, "_get_openai_client", lambda settings: fake)
    s = Settings(llm_provider="openai", openai_api_key="sk-test")

    out = await call_llm("sys", "user", model="gpt-4o-mini", json_mode=True, settings=s)

    assert out == "hello"
    call = fake.chat.completions.calls[0]
    assert call["model"] == "gpt-4o-mini"
    assert call["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_call_llm_empty_response_raises(monkeypatch):
    monkeypatch.setattr(llm_selector, "_get_openai_client", lambda settings: _FakeClient(""))
    s = Settings(llm_provider="openai", openai_api_key="sk-test")
    with pytest.raises(LLMError):
        await call_llm("sys", "user", model="gpt-4o-mini", settings=s)


@pytest.mark.asyncio
async def test_call_llm_azure_success_maps_deployment(monkeypatch):
    fake = _FakeClient("azure 응답")
    monkeypatch.setattr(llm_selector, "_get_azure_client", lambda settings: fake)
    s = Settings(
        llm_provider="azure",
        azure_openai_endpoint="https://x.openai.azure.com",
        azure_openai_api_key="key",
    )
    out = await call_llm("sys", "user", model="Azure-gpt4", settings=s)
    assert out == "azure 응답"
    # "Azure-" 접두사가 대소문자 무시로 제거된 deployment 이름이 전달돼야 한다
    assert fake.chat.completions.calls[0]["model"] == "gpt4"


@pytest.mark.asyncio
async def test_call_llm_self_hosted_success(monkeypatch):
    captured: dict = {}

    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "local 응답"}}]}

    class _HttpClient:
        def __init__(self, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> bool:
            return False

        async def post(self, url, json=None):
            captured["url"] = url
            captured["body"] = json
            return _Resp()

    monkeypatch.setattr(llm_selector.httpx, "AsyncClient", _HttpClient)
    s = Settings(llm_provider="self_hosted", self_hosted_llm_url="http://local:8000/v1")
    out = await call_llm("sys", "user", model="mistral-7b", settings=s)
    assert out == "local 응답"
    assert captured["url"] == "http://local:8000/v1/chat/completions"
    assert captured["body"]["model"] == "mistral-7b"
