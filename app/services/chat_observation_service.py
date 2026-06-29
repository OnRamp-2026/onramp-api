from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import ChatObservation
from app.middleware.request_id import request_id_var
from app.models.request import ChatRequest
from app.models.response import ChatResponse

logger = logging.getLogger(__name__)

SUCCESS_STATUSES = {"answerable", "partially_answerable"}
FAILURE_STATUSES = {"not_enough_evidence", "conflicting_evidence", "outdated_evidence"}

# 1차 cost는 설명용 추정치다. 실제 청구·정산 기준은 provider billing과 별도로 맞춘다.
MODEL_COST_PER_1M_TOKENS: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (5.00, 15.00),
    "azure-gpt4": (5.00, 15.00),
    "azure-gpt-4o": (5.00, 15.00),
}


def classify_result_bucket(answerability_status: str, retry_count: int, *, failed: bool = False) -> str:
    normalized = (answerability_status or "").strip().lower()
    if failed or normalized in FAILURE_STATUSES:
        return "failure"
    if retry_count > 0:
        return "requery"
    if normalized in SUCCESS_STATUSES:
        return "success"
    return "failure"


def estimate_cost_usd(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    normalized = (model_name or "").strip().lower()
    rates = MODEL_COST_PER_1M_TOKENS.get(normalized)
    if rates is None:
        return 0.0
    input_rate, output_rate = rates
    return round((prompt_tokens / 1_000_000 * input_rate) + (completion_tokens / 1_000_000 * output_rate), 6)


async def persist_chat_observation(
    db: AsyncSession,
    *,
    tenant_id: str,
    request: ChatRequest,
    response: ChatResponse | None,
    retry_count: int,
    usage: dict[str, int],
    duration_ms: int,
    failed: bool,
) -> None:
    settings = get_settings()
    requested_model = request.model or settings.default_model or ""
    model_used = (response.model_used if response is not None else "") or requested_model
    prompt_tokens = max(0, int(usage.get("input", 0)))
    completion_tokens = max(0, int(usage.get("output", 0)))
    total_tokens = max(0, int(usage.get("total", prompt_tokens + completion_tokens)))
    answerability_status = response.answerability_status if response is not None else ""
    observation = ChatObservation(
        request_id=request_id_var.get() or "",
        tenant_id=tenant_id,
        requested_model=requested_model,
        model_used=model_used,
        domain=response.domain if response is not None else "",
        answerability_status=answerability_status,
        retry_count=max(0, int(retry_count)),
        duration_ms=max(0, int(duration_ms)),
        source_count=len(response.sources) if response is not None else 0,
        result_bucket=classify_result_bucket(answerability_status, retry_count, failed=failed),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=estimate_cost_usd(model_used, prompt_tokens, completion_tokens),
    )
    try:
        db.add(observation)
        await db.commit()
    except Exception:
        await db.rollback()
        logger.exception("chat observation 저장 실패")
