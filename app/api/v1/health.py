from fastapi import APIRouter

from app.config import get_settings

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
    """DB 연결 상태 확인 상세 헬스체크."""
    checks = {
        "qdrant": "not_connected",
        "postgres": "not_connected",
        "redis": "not_connected",
    }

    # TODO: 실제 연결 체크로 교체
    # try:
    #     await qdrant_client.health()
    #     checks["qdrant"] = "ok"
    # except Exception:
    #     checks["qdrant"] = "error"

    all_ok = all(v == "ok" for v in checks.values())
    return {
        "status": "ok" if all_ok else "degraded",
        "checks": checks,
    }