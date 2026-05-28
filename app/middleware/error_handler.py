import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.middleware.request_id import request_id_var

logger = structlog.get_logger()


class OnRampError(Exception):
    """비즈니스 로직 에러 기본 클래스."""

    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code


class LLMError(OnRampError):
    """LLM 호출 실패."""

    def __init__(self, message: str = "LLM 호출에 실패했습니다"):
        super().__init__(message, status_code=502)


class RetrieverError(OnRampError):
    """문서 검색 실패."""

    def __init__(self, message: str = "문서 검색에 실패했습니다"):
        super().__init__(message, status_code=500)


class NotAnswerableError(OnRampError):
    """답변 불가 판정."""

    def __init__(self, message: str = "해당 질문에 답변할 수 없습니다"):
        super().__init__(message, status_code=422)


def register_error_handlers(app: FastAPI) -> None:
    """FastAPI 앱에 전역 예외 핸들러를 등록합니다."""

    @app.exception_handler(OnRampError)
    async def onramp_error_handler(request: Request, exc: OnRampError):
        logger.warning(
            "business_error",
            request_id=request_id_var.get(),
            error=exc.message,
            status=exc.status_code,
            path=request.url.path,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.message,
                "request_id": request_id_var.get(),
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        logger.error(
            "unhandled_error",
            request_id=request_id_var.get(),
            error=str(exc),
            error_type=type(exc).__name__,
            path=request.url.path,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "서버 내부 오류가 발생했습니다",
                "request_id": request_id_var.get(),
            },
        )
