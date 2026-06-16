"""Langfuse 클라이언트/콜백 팩토리 (LLMOps 관측 토대, #121).

설계 원칙:
- **kill-switch 우선**: `langfuse_enabled=false`(기본)면 모든 팩토리가 None을 반환한다.
  키가 없거나 SDK가 안 깔려도 앱은 정상 기동한다 (관측은 부가 기능, 절대 응답 경로를 막지 않는다).
- **lazy import**: `langfuse` 패키지를 모듈 최상단에서 import하지 않는다.
  enabled일 때만 import해, 미설치 환경(예: 일부 워커/테스트)에서도 안전하다.
- 실제 그래프 계측(CallbackHandler 주입)은 후속(E2, #120)에서 이 팩토리를 사용한다.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal

import structlog

from app.config import get_settings

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

logger = structlog.get_logger()


def is_enabled() -> bool:
    """관측이 켜져 있고 클라이언트가 실제로 준비됐는지."""
    return get_langfuse_client() is not None


@lru_cache(maxsize=1)
def get_langfuse_client() -> Langfuse | None:
    """Langfuse 클라이언트 싱글톤. disabled/미설치/초기화 실패 시 None.

    None 반환은 정상 동작이다 — 호출부는 None을 no-op으로 다뤄야 한다.
    """
    settings = get_settings()
    if not settings.langfuse_enabled:
        return None

    try:
        from langfuse import Langfuse
    except ImportError:
        logger.warning("langfuse_sdk_missing", hint="pip install langfuse — 관측 비활성으로 동작")
        return None

    try:
        client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            host=settings.langfuse_host,
        )
    except Exception as exc:  # 초기화 실패가 절대 앱 기동/응답을 막지 않게 흡수
        logger.warning("langfuse_init_failed", error=str(exc))
        return None

    logger.info("langfuse_enabled", host=settings.langfuse_host)
    return client


@lru_cache(maxsize=1)
def get_callback_handler() -> CallbackHandler | None:
    """LangChain/LangGraph용 Langfuse CallbackHandler. disabled/미설치 시 None.

    핸들러는 전역 클라이언트를 사용하므로, 먼저 `get_langfuse_client()`로 초기화한다.
    """
    if get_langfuse_client() is None:
        return None

    try:
        from langfuse.langchain import CallbackHandler
    except ImportError:
        logger.warning("langfuse_langchain_missing", hint="langfuse[langchain] 필요 — 관측 비활성")
        return None

    return CallbackHandler()


def langfuse_run_config(
    *,
    request_id: str = "",
    tenant: str | None = None,
    session_id: str | None = None,
    model: str = "",
    tags: list[str] | None = None,
) -> RunnableConfig:
    """LangGraph(ainvoke)에 넘길 RunnableConfig 조각을 만든다.

    관측 비활성/미설치면 빈 dict(`{}`)를 반환한다 — 호출부는 `config=cfg or None`로
    그대로 넘기면 비활성 시 기존 동작과 100% 동일하다.

    metadata의 `langfuse_*` 키는 v3 CallbackHandler가 trace 속성으로 인식한다:
    - `langfuse_user_id`  = tenant (멀티테넌트 비용·품질 분리)
    - `langfuse_session_id` = conversation 부재 시 request_id로 대체 (턴 단위 추적)
    - `langfuse_tags`     = [model, …] (필터·집계용)
    """
    handler = get_callback_handler()
    if handler is None:
        return {}

    metadata: dict[str, object] = {}
    if tenant:
        metadata["langfuse_user_id"] = tenant
    sid = session_id or request_id
    if sid:
        metadata["langfuse_session_id"] = sid
    if request_id:
        metadata["request_id"] = request_id
    tag_list = [t for t in [model, *(tags or [])] if t]
    if tag_list:
        metadata["langfuse_tags"] = tag_list

    return {"callbacks": [handler], "metadata": metadata}


@contextmanager
def _current_observation(*, name: str, as_type: str, input: Any = None, model: str | None = None) -> Iterator[Any]:
    """현재 OTEL 컨텍스트 아래에 observation(span/generation)을 열고 yield한다.

    비활성/미설치/생성 실패 시 None을 yield(no-op) — 관측이 응답 경로를 막지 않는다.
    `as_type="span"`으로 만든 루트 아래에 CallbackHandler 노드 스팬과 generation이 모두 중첩된다.
    """
    client = get_langfuse_client()
    if client is None:
        yield None
        return

    kwargs: dict[str, Any] = {"name": name, "as_type": as_type, "input": input}
    if model is not None:
        kwargs["model"] = model
    try:
        cm = client.start_as_current_observation(**kwargs)
        obs = cm.__enter__()
    except Exception as exc:  # 생성 실패 흡수
        logger.warning("langfuse_observation_start_failed", error=str(exc), as_type=as_type)
        yield None
        return

    # 본문 예외는 span에 error로 기록하고 그대로 전파(LLMError 등 유지).
    try:
        yield obs
    except BaseException as exc:
        cm.__exit__(type(exc), exc, exc.__traceback__)
        raise
    else:
        cm.__exit__(None, None, None)


def current_trace_id() -> str | None:
    """현재 활성 trace id (span 컨텍스트 안에서 호출). 비활성/없으면 None."""
    client = get_langfuse_client()
    if client is None:
        return None
    try:
        trace_id: str | None = client.get_current_trace_id()
        return trace_id
    except Exception as exc:
        logger.warning("langfuse_trace_id_failed", error=str(exc))
        return None


def score_current_trace(*, name: str, value: float, comment: str | None = None) -> None:
    """현재 활성 trace에 score를 부착 (span 컨텍스트 안에서). 비활성/실패 시 no-op."""
    client = get_langfuse_client()
    if client is None:
        return
    try:
        client.score_current_trace(name=name, value=value, comment=comment)
    except Exception as exc:  # 관측 실패가 응답 경로를 막지 않게
        logger.warning("langfuse_score_failed", error=str(exc), name=name)


def create_trace_score(
    *,
    trace_id: str,
    name: str,
    value: float,
    comment: str | None = None,
    data_type: Literal["NUMERIC", "BOOLEAN"] = "NUMERIC",
) -> bool:
    """trace_id로 score를 생성 (피드백 엔드포인트용). 비활성/실패 시 False."""
    client = get_langfuse_client()
    if client is None:
        return False
    try:
        client.create_score(trace_id=trace_id, name=name, value=value, comment=comment or None, data_type=data_type)
        return True
    except Exception as exc:
        logger.warning("langfuse_create_score_failed", error=str(exc), name=name)
        return False


@contextmanager
def langfuse_span(*, name: str, input: Any = None) -> Iterator[Any]:
    """한 작업 단위(예: 채팅 1턴)를 감싸는 루트 span. 비활성 시 no-op(None).

    이 안에서 LangGraph(CallbackHandler) invoke와 call_llm generation이 실행되면
    모두 이 span 아래로 중첩되어 **한 trace**가 된다.
    """
    with _current_observation(name=name, as_type="span", input=input) as obs:
        yield obs


@contextmanager
def langfuse_generation(*, name: str, model: str | None = None, input: Any = None) -> Iterator[Any]:
    """LLM 호출용 generation observation. 비활성 시 no-op(None).

    `gen.update(output=..., usage_details=...)`로 token을 채우면 Langfuse가 model 기준 cost를 계산한다.
    """
    with _current_observation(name=name, as_type="generation", input=input, model=model) as obs:
        yield obs
