"""Langfuse 클라이언트/콜백 팩토리 (LLMOps 관측 토대, #121).

설계 원칙:
- **kill-switch 우선**: `langfuse_enabled=false`(기본)면 모든 팩토리가 None을 반환한다.
  키가 없거나 SDK가 안 깔려도 앱은 정상 기동한다 (관측은 부가 기능, 절대 응답 경로를 막지 않는다).
- **lazy import**: `langfuse` 패키지를 모듈 최상단에서 import하지 않는다.
  enabled일 때만 import해, 미설치 환경(예: 일부 워커/테스트)에서도 안전하다.
- 실제 그래프 계측(CallbackHandler 주입)은 후속(E2, #120)에서 이 팩토리를 사용한다.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import structlog

from app.config import get_settings

if TYPE_CHECKING:
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
