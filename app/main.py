from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.slack import router as slack_router
from app.api.v1.auth_browser import browser_router as browser_auth_router
from app.api.v1.router import build_v1_router
from app.config import get_settings
from app.db.opensearch import close_opensearch
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
    for close_fn in (close_opensearch, close_postgres, close_redis):
        try:
            await close_fn()
        except Exception:
            print(f"Shutdown warning: {close_fn.__name__} failed")
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
    app.include_router(build_v1_router(enable_slack_auth=settings.auth_enable_slack_login), prefix="/v1")
    if settings.auth_enable_slack_login:
        # 브라우저 SPA = 쿠키 세션 + redirect(/auth/login·callback·me·logout). API/토큰은 /v1/auth/slack/*.
        app.include_router(browser_auth_router)
    app.include_router(slack_router, prefix="/slack", tags=["Slack"])

    return app


app = create_app()
