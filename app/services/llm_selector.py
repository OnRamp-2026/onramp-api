"""LLM Selector — Sovereign provider(openai / azure / self_hosted) 선택 + 호출.

call_llm 하나로 모든 Agent와 asset_service가 LLM을 호출한다. provider는 model 이름
우선, 없으면 config.llm_provider, 그것도 없으면 openai 기본으로 라우팅한다.
반환은 항상 응답 텍스트(str). JSON 파싱은 호출부(Agent/Service)가 한다.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from openai import AsyncAzureOpenAI, AsyncOpenAI

from app.config import Settings, get_settings
from app.middleware.error_handler import LLMError

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4o-mini"
_AZURE_API_VERSION = "2024-06-01"
_OPENAI_PREFIXES = ("gpt-", "o1", "o3", "chatgpt")
_AZURE_PREFIX = "azure-"

_openai_client: AsyncOpenAI | None = None
_azure_client: AsyncAzureOpenAI | None = None


def resolve_provider(model: str, settings: Settings) -> str:
    """provider 결정 — config.llm_provider가 authoritative.

    llm_provider가 설정돼 있으면 그대로 쓰고, model은 해당 provider에 넘길 모델/deployment
    이름으로만 사용한다 (default_model이 provider 선택을 좌우하지 않도록). llm_provider가
    비어 있을 때만 model 이름으로 추론하고, 그것도 없으면 openai 기본.
    """
    provider = settings.llm_provider.strip().lower()
    if provider:
        return provider
    name = model.strip().lower()
    if name.startswith(_AZURE_PREFIX):
        return "azure"
    if name.startswith(_OPENAI_PREFIXES):
        return "openai"
    if name:
        return "self_hosted"
    return "openai"


async def call_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str = "",
    temperature: float = 0.0,
    max_tokens: int | None = None,
    timeout: float = 30.0,
    json_mode: bool = False,
    settings: Settings | None = None,
) -> str:
    """system+user 프롬프트로 LLM을 1회 호출하고 응답 텍스트를 반환한다."""
    settings = settings or get_settings()
    provider = resolve_provider(model, settings)
    args = (system_prompt, user_prompt, model, temperature, max_tokens, timeout, json_mode, settings)

    try:
        if provider == "openai":
            content = await _call_openai(*args)
        elif provider == "azure":
            content = await _call_azure(*args)
        elif provider == "self_hosted":
            content = await _call_self_hosted(*args)
        else:
            raise LLMError(f"지원하지 않는 llm_provider: {provider!r}")
    except LLMError:
        raise
    except Exception as exc:  # openai/httpx 등 업스트림 실패 → 502
        logger.warning("LLM 호출 실패 (provider=%s)", provider, exc_info=True)
        raise LLMError("LLM 호출에 실패했습니다") from exc

    if not content:
        raise LLMError("LLM 응답이 비어있습니다")
    return content


def _extra_kwargs(max_tokens: int | None, json_mode: bool) -> dict[str, Any]:
    """json_mode·max_tokens를 chat.completions create kwargs로 변환 (azure/self_hosted 공용)."""
    kwargs: dict[str, Any] = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return kwargs


def _content(choices: list[Any]) -> str:
    """choices에서 첫 메시지 텍스트를 추출한다 (없으면 LLMError)."""
    if not choices:
        raise LLMError("LLM 응답에 choices가 없습니다")
    return choices[0].message.content or ""


def _get_openai_client(settings: Settings) -> AsyncOpenAI:
    """OpenAI 비동기 클라이언트 싱글톤."""
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _openai_client


def _get_azure_client(settings: Settings) -> AsyncAzureOpenAI:
    """Azure OpenAI 비동기 클라이언트 싱글톤."""
    global _azure_client
    if _azure_client is None:
        _azure_client = AsyncAzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=_AZURE_API_VERSION,
        )
    return _azure_client


async def _call_openai(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int | None,
    timeout: float,
    json_mode: bool,
    settings: Settings,
) -> str:
    """OpenAI chat.completions 호출 (o1/o3 reasoning 모델은 temperature 생략·max_completion_tokens)."""
    if not settings.openai_api_key:
        raise LLMError("OpenAI API 키가 설정되지 않았습니다")
    model_name = model or settings.default_model or _DEFAULT_MODEL
    create_kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "timeout": timeout,
    }
    if json_mode:
        create_kwargs["response_format"] = {"type": "json_object"}
    if model_name.lower().startswith(("o1", "o3")):
        # reasoning 모델: temperature 미지원, max_tokens 대신 max_completion_tokens
        if max_tokens is not None:
            create_kwargs["max_completion_tokens"] = max_tokens
    else:
        create_kwargs["temperature"] = temperature
        if max_tokens is not None:
            create_kwargs["max_tokens"] = max_tokens
    resp = await _get_openai_client(settings).chat.completions.create(**create_kwargs)
    return _content(resp.choices)


async def _call_azure(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int | None,
    timeout: float,
    json_mode: bool,
    settings: Settings,
) -> str:
    """Azure OpenAI 호출 ("azure-" 접두사를 뗀 이름을 deployment로 사용)."""
    if not settings.azure_openai_endpoint or not settings.azure_openai_api_key:
        raise LLMError("Azure OpenAI 설정(endpoint/key)이 없습니다")
    # "azure-" 접두사를 대소문자 무시로 제거하되 deployment 이름의 원본 케이스는 보존
    stripped = model.strip()
    base = stripped[len(_AZURE_PREFIX) :] if stripped.lower().startswith(_AZURE_PREFIX) else stripped
    deployment = base or settings.default_model or _DEFAULT_MODEL
    resp = await _get_azure_client(settings).chat.completions.create(
        model=deployment,  # Azure는 deployment 이름
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        timeout=timeout,
        **_extra_kwargs(max_tokens, json_mode),
    )
    return _content(resp.choices)


async def _call_self_hosted(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int | None,
    timeout: float,
    json_mode: bool,
    settings: Settings,
) -> str:
    """Self-hosted OpenAI 호환 서버(/chat/completions)를 httpx로 호출."""
    if not settings.self_hosted_llm_url:
        raise LLMError("Self-hosted LLM URL이 설정되지 않았습니다")
    body: dict[str, Any] = {
        "model": model or settings.self_hosted_model_name or _DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        **_extra_kwargs(max_tokens, json_mode),
    }
    url = f"{settings.self_hosted_llm_url.rstrip('/')}/chat/completions"
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=body)
        resp.raise_for_status()
        payload = resp.json()
    choices = payload.get("choices") or []
    if not choices:
        raise LLMError("Self-hosted LLM 응답에 choices가 없습니다")
    return choices[0].get("message", {}).get("content", "") or ""


def reset_clients() -> None:
    """테스트용 클라이언트 싱글톤 초기화."""
    global _openai_client, _azure_client
    _openai_client = None
    _azure_client = None
