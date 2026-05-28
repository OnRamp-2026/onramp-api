import time

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.middleware.request_id import request_id_var

logger = structlog.get_logger()


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    모든 요청/응답을 구조화 로그로 기록.
    - 메서드, 경로, 상태코드, 소요시간
    - Request ID 자동 포함
    - 헬스체크는 로그 생략 (노이즈 방지)
    """

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        if request.url.path.startswith("/v1/health"):
            response: Response = await call_next(request)
            return response

        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        logger.info(
            "request",
            request_id=request_id_var.get(),
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=duration_ms,
        )

        return response
