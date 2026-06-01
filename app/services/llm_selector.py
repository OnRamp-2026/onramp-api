"""LLM 호출 — provider 추상화(OpenAI P0 / azure·self-hosted P1).

전체 Sovereign LLM 선택 기능(#7)은 별도다. 여기서는 Agent들이 공통으로 의존하는
최소 ``call_llm`` 인터페이스만 제공한다 (embedder의 provider 패턴과 동일한 형태).
"""

from __future__ import annotations

from openai import AsyncOpenAI

from app.config import Settings, get_settings

_DEFAULT_MODEL = "gpt-4o-mini"
_OPENAI_PROVIDERS = {"", "openai"}  # OpenAI chat completions 경로 (기본값 "" 포함)
_client: AsyncOpenAI | None = None


def _resolve_model(model: str, settings: Settings) -> str:
    return model or settings.default_model or _DEFAULT_MODEL


def _get_client(settings: Settings) -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


async def call_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str = "",
    temperature: float = 0.0,
    timeout: float = 30.0,
    json_mode: bool = False,
    settings: Settings | None = None,
) -> str:
    """system+user 프롬프트로 LLM을 1회 호출하고 응답 텍스트를 반환한다.

    P0는 OpenAI chat completions. provider=self_hosted는 P1(#7).
    """
    settings = settings or get_settings()
    provider = settings.llm_provider
    if provider in ("self_hosted", "azure"):
        raise NotImplementedError(f"{provider} LLM provider는 P1 (#7)")
    if provider not in _OPENAI_PROVIDERS:  # 오타·미지원 provider는 fail-fast
        raise ValueError(f"지원하지 않는 llm_provider: {provider!r} (지원: openai)")

    kwargs: dict = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    resp = await _get_client(settings).chat.completions.create(
        model=_resolve_model(model, settings),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        timeout=timeout,
        **kwargs,
    )
    if not resp.choices:  # 비정상 응답은 예외로 승격 → 호출부 fallback의 error 기록 경로를 탄다
        raise RuntimeError("LLM 응답에 choices가 없습니다")
    return resp.choices[0].message.content or ""


def reset_client() -> None:
    """테스트용 클라이언트 싱글톤 초기화."""
    global _client
    _client = None
