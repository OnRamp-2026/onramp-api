"""로컬 monitoring 대시보드 검증용 chat_observations 시드 스크립트.

LLM 호출 없이도 Postgres -> monitoring API -> monitoring UI 흐름을 확인할 수 있도록
chat_observations 샘플 row를 적재한다.

사용 예:
    python scripts/seed_monitoring_local.py
    python scripts/seed_monitoring_local.py --tenant-id onramp --clear
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from sqlalchemy import delete

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.db.models import ChatObservation  # noqa: E402
from app.db.postgres import get_session_factory  # noqa: E402


def build_rows(tenant_id: str) -> list[ChatObservation]:
    now = datetime.now(UTC)
    return [
        _row(
            tenant_id=tenant_id,
            result_bucket="success",
            created_at=now - timedelta(days=29),
            total_tokens=700,
            cost=0.000285,
            duration_ms=940,
            source_count=3,
        ),
        _row(
            tenant_id=tenant_id,
            result_bucket="requery",
            created_at=now - timedelta(days=23),
            total_tokens=860,
            cost=0.000354,
            duration_ms=1410,
            retry_count=1,
            source_count=2,
        ),
        _row(
            tenant_id=tenant_id,
            result_bucket="success",
            created_at=now - timedelta(days=13),
            total_tokens=920,
            cost=0.000378,
            duration_ms=1020,
            source_count=4,
        ),
        _row(
            tenant_id=tenant_id,
            result_bucket="failure",
            created_at=now - timedelta(days=7),
            total_tokens=880,
            cost=0.00036,
            duration_ms=1920,
            source_count=1,
            answerability_status="not_enough_evidence",
        ),
        _row(
            tenant_id=tenant_id,
            result_bucket="success",
            created_at=now - timedelta(days=1),
            total_tokens=1140,
            cost=0.000468,
            duration_ms=980,
            source_count=5,
        ),
    ]


def _row(
    *,
    tenant_id: str,
    result_bucket: str,
    created_at: datetime,
    total_tokens: int,
    cost: float,
    duration_ms: int,
    source_count: int,
    retry_count: int = 0,
    answerability_status: str = "answerable",
) -> ChatObservation:
    prompt_tokens = total_tokens // 2
    completion_tokens = total_tokens - prompt_tokens
    return ChatObservation(
        request_id=str(uuid4()),
        tenant_id=tenant_id,
        requested_model="gpt-4o-mini",
        model_used="gpt-4o-mini",
        domain="search",
        answerability_status=answerability_status,
        retry_count=retry_count,
        duration_ms=duration_ms,
        source_count=source_count,
        result_bucket=result_bucket,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=cost,
        created_at=created_at,
    )


async def run(*, tenant_id: str, clear: bool) -> None:
    session_factory = get_session_factory()
    async with session_factory() as session:
        if clear:
            await session.execute(delete(ChatObservation).where(ChatObservation.tenant_id == tenant_id))
        rows = build_rows(tenant_id)
        session.add_all(rows)
        await session.commit()
    print(f"seeded {len(rows)} chat_observations rows for tenant={tenant_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant-id", default="onramp", help="시드할 tenant_id (기본: onramp)")
    parser.add_argument("--clear", action="store_true", help="같은 tenant_id 기존 row를 지우고 다시 적재")
    args = parser.parse_args()
    asyncio.run(run(tenant_id=args.tenant_id, clear=args.clear))


if __name__ == "__main__":
    main()
