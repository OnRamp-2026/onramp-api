from fastapi import APIRouter

from app.api.v1.auth import router as auth_router
from app.api.v1.asset import router as asset_router
from app.api.v1.chat import router as chat_router
from app.api.v1.health import router as health_router
from app.api.v1.reports import router as reports_router
from app.api.v1.transcriptions import router as transcriptions_router


def build_v1_router(*, enable_slack_auth: bool) -> APIRouter:
    router = APIRouter()
    router.include_router(health_router, tags=["Health"])
    if enable_slack_auth:
        router.include_router(auth_router)
    router.include_router(chat_router, tags=["Chat"])
    router.include_router(asset_router, tags=["Asset"])
    router.include_router(transcriptions_router, tags=["Transcriptions"])
    router.include_router(reports_router, tags=["Reports"])
    return router
