from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth import router as auth_router
from app.api.slack import router as slack_router
from app.api.v1.router import v1_router
from app.config import get_settings
from app.db.postgres import close_postgres, get_engine
from app.db.qdrant import close_qdrant, get_qdrant
from app.db.redis import close_redis, get_redis
from app.middleware.error_handler import register_error_handlers
from app.middleware.logging import LoggingMiddleware
from app.middleware.request_id import RequestIdMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──
    settings = get_settings()
    print(f"Starting {settings.app_name} {settings.app_version}...")

    get_qdrant()
    print("  Qdrant client ready")

    get_engine()
    print("  PostgreSQL engine ready")

    get_redis()
    print("  Redis client ready")

    yield

    # ── Shutdown ──
    close_qdrant()
    await close_postgres()
    await close_redis()
    print("Shutdown complete")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Middleware (등록 역순으로 실행 — RequestId가 가장 먼저)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Error handlers
    register_error_handlers(app)

    # Routers
    app.include_router(v1_router, prefix="/v1")
    app.include_router(auth_router, prefix="/auth", tags=["Auth"])
    app.include_router(slack_router, prefix="/slack", tags=["Slack"])

    return app


app = create_app()
