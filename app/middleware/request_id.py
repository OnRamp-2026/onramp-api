import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# 어디서든 현재 요청의 ID를 가져올 수 있는 컨텍스트 변수
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


class RequestIdMiddleware(BaseHTTPMiddleware):
    """
    모든 요청에 X-Request-ID를 부여.
    - 클라이언트가 보낸 게 있으면 그대로 사용
    - 없으면 새로 생성
    - 응답 헤더에도 포함해서 클라이언트가 추적 가능
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        rid = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request_id_var.set(rid)

        response = await call_next(request)
        response.headers["X-Request-ID"] = rid

        return response
