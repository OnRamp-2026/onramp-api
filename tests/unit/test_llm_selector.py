"""LLM Selector 단위 테스트 (실제 LLM 호출 없이 provider 라우팅·에러 검증)."""

import pytest

from app.config import Settings
from app.middleware.error_handler import LLMError
from app.services import llm_selector
from app.services.llm_selector import call_llm, resolve_provider


class _FakeCompletions:
    """chat.completions stub — create 호출 인자를 기록하고 고정 응답을 반환."""

    def __init__(self, content: str = "응답") -> None:
        self.content = content
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        """create 호출을 기록하고 고정 content를 담은 응답을 반환한다."""
        self.calls.append(kwargs)
        message = type("M", (), {"content": self.content})()
        choice = type("C", (), {"message": message})()
        return type("R", (), {"choices": [choice]})()


class _FakeClient:
    """AsyncOpenAI/AsyncAzureOpenAI 대체 stub."""

    def __init__(self, content: str = "응답") -> None:
        self.chat = type("Chat", (), {"completions": _FakeCompletions(content)})()


@pytest.fixture(autouse=True)
def _reset_clients():
    """클라이언트 싱글톤을 테스트마다 초기화."""
    llm_selector.reset_clients()
    yield
    llm_selector.reset_clients()


# ── resolve_provider ──
def test_resolve_provider_by_model_name():
    """config 미설정 시 model 이름으로 provider 추론."""
    s = Settings(llm_provider="")
    assert resolve_provider("gpt-4o-mini", s) == "openai"
    assert resolve_provider("azure-gpt4", s) == "azure"
    assert resolve_provider("mistral-7b", s) == "self_hosted"


def test_resolve_provider_from_config_when_model_empty():
    """model이 비면 config.llm_provider 사용."""
    assert resolve_provider("", Settings(llm_provider="azure")) == "azure"


def test_resolve_provider_default_openai():
    """model·config 모두 비면 openai 기본."""
    assert resolve_provider("", Settings(llm_provider="")) == "openai"


def test_resolve_provider_config_takes_precedence_over_model_name():
    """LLM_PROVIDER가 설정되면 model 이름이 openai여도 config가 이긴다.

    (LLM_PROVIDER=azure, DEFAULT_MODEL=gpt-4o 조합이 openai로 새던 버그 방지)
    """
    s = Settings(llm_provider="azure", default_model="gpt-4o")
    assert resolve_provider("gpt-4o", s) == "azure"
    assert resolve_provider("", s) == "azure"


def test_resolve_provider_normalizes_config_value():
    """대소문자/공백 섞인 config 값도 정규화돼 분기와 매칭된다."""
    assert resolve_provider("", Settings(llm_provider=" Azure ")) == "azure"
    assert resolve_provider("", Settings(llm_provider="OpenAI")) == "openai"


# ── 에러 경로 ──
@pytest.mark.asyncio
async def test_call_llm_openai_no_key_raises():
    """OpenAI 키 누락 → LLMError (에러 정규화 회귀 고정)."""
    s = Settings(llm_provider="openai", openai_api_key="")
    with pytest.raises(LLMError):
        await call_llm("sys", "user", model="gpt-4o-mini", settings=s)


@pytest.mark.asyncio
async def test_call_llm_azure_no_config_raises():
    """Azure endpoint/key 누락 → LLMError."""
    s = Settings(llm_provider="azure", azure_openai_endpoint="", azure_openai_api_key="")
    with pytest.raises(LLMError):
        await call_llm("sys", "user", model="azure-gpt4", settings=s)


@pytest.mark.asyncio
async def test_call_llm_self_hosted_no_url_raises():
    """Self-hosted URL 누락 → LLMError."""
    s = Settings(llm_provider="self_hosted", self_hosted_llm_url="")
    with pytest.raises(LLMError):
        await call_llm("sys", "user", model="mistral-7b", settings=s)


# ── 정상 호출 ──
@pytest.mark.asyncio
async def test_call_llm_openai_success_and_json_mode(monkeypatch):
    """openai 정상 호출 + json_mode → response_format 전달."""
    fake = _FakeClient("hello")
    monkeypatch.setattr(llm_selector, "_get_openai_client", lambda settings: fake)
    s = Settings(llm_provider="openai", openai_api_key="sk-test")

    out = await call_llm("sys", "user", model="gpt-4o-mini", json_mode=True, settings=s)

    assert out == "hello"
    call = fake.chat.completions.calls[0]
    assert call["model"] == "gpt-4o-mini"
    assert call["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_call_llm_openai_reasoning_model_omits_temperature(monkeypatch):
    """o1/o3 reasoning 모델은 temperature 미전달 + max_completion_tokens 사용."""
    fake = _FakeClient("reasoning")
    monkeypatch.setattr(llm_selector, "_get_openai_client", lambda settings: fake)
    s = Settings(llm_provider="openai", openai_api_key="sk-test")

    out = await call_llm("sys", "user", model="o1-mini", max_tokens=100, settings=s)

    assert out == "reasoning"
    call = fake.chat.completions.calls[0]
    assert "temperature" not in call
    assert call.get("max_completion_tokens") == 100
    assert "max_tokens" not in call


@pytest.mark.asyncio
async def test_call_llm_empty_response_raises(monkeypatch):
    """빈 응답 → LLMError."""
    monkeypatch.setattr(llm_selector, "_get_openai_client", lambda settings: _FakeClient(""))
    s = Settings(llm_provider="openai", openai_api_key="sk-test")
    with pytest.raises(LLMError):
        await call_llm("sys", "user", model="gpt-4o-mini", settings=s)


@pytest.mark.asyncio
async def test_call_llm_azure_success_maps_deployment(monkeypatch):
    """azure 정상 호출 — "azure-" 접두사 제거된 이름이 deployment(model)로 전달."""
    fake = _FakeClient("azure 응답")
    monkeypatch.setattr(llm_selector, "_get_azure_client", lambda settings: fake)
    s = Settings(
        llm_provider="azure",
        azure_openai_endpoint="https://x.openai.azure.com",
        azure_openai_api_key="key",
    )
    out = await call_llm("sys", "user", model="Azure-gpt4", settings=s)
    assert out == "azure 응답"
    assert fake.chat.completions.calls[0]["model"] == "gpt4"


@pytest.mark.asyncio
async def test_call_llm_self_hosted_success(monkeypatch):
    """self-hosted 정상 호출 — /chat/completions url·model·응답 파싱 검증."""
    captured: dict = {}

    class _Resp:
        """httpx 응답 stub."""

        def raise_for_status(self) -> None:
            """성공 응답 가정."""
            return None

        def json(self) -> dict:
            """OpenAI 호환 응답 본문."""
            return {"choices": [{"message": {"content": "local 응답"}}]}

    class _HttpClient:
        """httpx.AsyncClient stub (async context manager)."""

        def __init__(self, **kwargs) -> None:
            pass

        async def __aenter__(self):
            """컨텍스트 진입 — self 반환."""
            return self

        async def __aexit__(self, *args) -> None:
            """컨텍스트 종료 — 예외 억제하지 않음."""
            return None

        async def post(self, url, json=None):
            """요청 url/body를 기록하고 stub 응답을 반환."""
            captured["url"] = url
            captured["body"] = json
            return _Resp()

    monkeypatch.setattr(llm_selector.httpx, "AsyncClient", _HttpClient)
    s = Settings(llm_provider="self_hosted", self_hosted_llm_url="http://local:8000/v1")
    out = await call_llm("sys", "user", model="mistral-7b", settings=s)
    assert out == "local 응답"
    assert captured["url"] == "http://local:8000/v1/chat/completions"
    assert captured["body"]["model"] == "mistral-7b"
