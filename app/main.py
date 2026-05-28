from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import v1_router
from app.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──
    settings = get_settings()
    print(f"{settings.app_name} {settings.app_version} starting...")
    # TODO: DB 커넥션 풀 초기화 (Qdrant, PostgreSQL, Redis)
    yield
    # ── Shutdown ──
    # TODO: DB 커넥션 풀 정리
    print("Shutting down...")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 운영 시 onramp-web 도메인으로 제한
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(v1_router, prefix="/v1")

    return app


app = create_app()