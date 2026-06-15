"""LLMOps 관측 (Langfuse) — 클라이언트/콜백 팩토리.

`langfuse_enabled=false`(기본)면 전부 no-op(None 반환)이라 키 없이도 앱이 기동한다.
"""

from app.observability.langfuse import get_callback_handler, get_langfuse_client, is_enabled

__all__ = ["get_callback_handler", "get_langfuse_client", "is_enabled"]
