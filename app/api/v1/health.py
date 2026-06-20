from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.db.postgres import check_postgres
from app.db.qdrant import check_qdrant
from app.db.redis import check_redis

router = APIRouter()


@router.get("/health")
async def health_check():
    settings = get_settings()
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.app_version,
    }


@router.get("/health/ready")
async def readiness_check():
    """DB 연결 상태 확인 헬스체크."""
    qdrant_ok = await check_qdrant()
    postgres_ok = await check_postgres()
    redis_ok = await check_redis()

    checks = {
        "qdrant": "ok" if qdrant_ok else "error",
        "postgres": "ok" if postgres_ok else "error",
        "redis": "ok" if redis_ok else "error",
    }

    all_ok = all(v == "ok" for v in checks.values())
    payload = {
        "status": "ok" if all_ok else "degraded",
        "checks": checks,
    }
    if all_ok:
        return payload
    return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=payload)
