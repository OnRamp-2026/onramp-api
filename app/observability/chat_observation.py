from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass
class ChatObservationState:
    retry_count: int = 0


_chat_observation_var: ContextVar[ChatObservationState | None] = ContextVar(
    "chat_observation_state",
    default=None,
)


@contextmanager
def chat_observation_scope() -> Iterator[ChatObservationState]:
    state = ChatObservationState()
    token = _chat_observation_var.set(state)
    try:
        yield state
    finally:
        _chat_observation_var.reset(token)


def record_chat_retry_count(retry_count: int) -> None:
    state = _chat_observation_var.get()
    if state is not None:
        state.retry_count = max(0, int(retry_count))


def current_chat_observation() -> ChatObservationState | None:
    return _chat_observation_var.get()
