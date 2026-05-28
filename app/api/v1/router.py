from fastapi import APIRouter

from app.api.v1.health import router as health_router

v1_router = APIRouter()

v1_router.include_router(health_router, tags=["Health"])

# TODO: 아래 라우터는 구현 후 등록
# from app.api.v1.chat import router as chat_router
# from app.api.v1.asset import router as asset_router
# v1_router.include_router(chat_router, tags=["Chat"])
# v1_router.include_router(asset_router, tags=["Asset"])