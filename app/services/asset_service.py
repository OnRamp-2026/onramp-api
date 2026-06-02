"""자산화 서비스 — 회의 녹취 → 5요소 보고서 → HITL → Confluence 등록.

Graph 미사용: asset_service가 llm_selector.call_llm을 직접 호출한다.
초안 저장소는 P0에서 메모리(dict). P1에서 Redis/PostgreSQL로 이전.
"""

from __future__ import annotations

import html as html_lib
import logging
from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, ValidationError, field_validator

from app.db.confluence import ConfluenceClient
from app.middleware.error_handler import OnRampError
from app.models.request import AssetRequest, AssetUpdateRequest
from app.models.response import AssetApproveResponse, AssetResponse, FiveElementsResponse
from app.services.llm_selector import call_llm

logger = logging.getLogger(__name__)

ASSET_SYSTEM_PROMPT = """너는 회의 녹취록에서 5요소 구조화 보고서를 작성하는 AI다.
녹취록은 구어체·반복·불완전한 문장이 섞여 있으니, 핵심만 정리해 JSON으로 반환한다.
지어내지 말고 녹취에 등장한 내용만 근거로 삼는다.

[5요소]
- situation: 현재 상황 / 배경
- cause: 원인
- evidence: 근거 (녹취에서 언급된 사실)
- solution: 해결 / 조치 (단계가 여럿이면 줄바꿈으로)
- infra_context: 인프라 환경·설정·의존성

[title]
- 보고서 제목을 한 줄로 생성한다 (요청에 title이 있으면 그대로 둬도 된다).

[출력 형식]
- 반드시 JSON만 반환. 키: title, situation, cause, evidence, solution, infra_context
- 각 5요소는 문자열 하나로. 단계가 여럿이면 배열이 아니라 줄바꿈으로 연결한다.
"""

# P0 초안 저장소 (메모리)
_draft_store: dict[str, AssetResponse] = {}


class _AssetDraft(BaseModel):
    """ASSET LLM 응답 파싱 스키마."""

    title: str = ""
    situation: str = ""
    cause: str = ""
    evidence: str = ""
    solution: str = ""
    infra_context: str = ""

    @field_validator("title", "situation", "cause", "evidence", "solution", "infra_context", mode="before")
    @classmethod
    def _coerce_str(cls, value: object) -> str:
        if isinstance(value, list):
            return "\n".join(str(item) for item in value)
        if value is None:
            return ""
        return str(value)


def _now() -> str:
    return datetime.now(UTC).isoformat()


async def create_report(request: AssetRequest) -> AssetResponse:
    """녹취 텍스트에서 5요소 보고서 초안을 생성한다 (status=draft)."""
    user_prompt = f"카테고리: {request.category}\n녹취록:\n{request.transcript}"
    try:
        raw = await call_llm(ASSET_SYSTEM_PROMPT, user_prompt, model=request.model, json_mode=True)
        draft = _AssetDraft.model_validate_json(raw)
    except ValidationError as exc:
        logger.warning("ASSET 응답 파싱 실패", exc_info=True)
        raise OnRampError("보고서 생성에 실패했습니다 (파싱 오류)", status_code=502) from exc
    except OnRampError:
        raise
    except Exception as exc:  # LLM 호출 실패
        logger.exception("ASSET LLM 호출 실패")
        raise OnRampError("보고서 생성 중 오류가 발생했습니다", status_code=502) from exc

    # 형식은 맞지만 내용이 빈 응답 방어 — 핵심 요소가 모두 비면 생성 실패로 본다
    if not (draft.situation.strip() or draft.evidence.strip() or draft.solution.strip()):
        logger.warning("ASSET 보고서 내용이 비어 있음")
        raise OnRampError("보고서 생성에 실패했습니다 (빈 응답)", status_code=502)

    now = _now()
    report = AssetResponse(
        report_id=str(uuid4()),
        title=request.title or draft.title or "제목 없는 보고서",
        report=FiveElementsResponse(
            situation=draft.situation,
            cause=draft.cause,
            evidence=draft.evidence,
            solution=draft.solution,
            infra_context=draft.infra_context,
        ),
        category=request.category,
        status="draft",
        confluence_url="",
        created_at=now,
        updated_at=now,
    )
    _draft_store[report.report_id] = report
    return report


def get_report(report_id: str) -> AssetResponse:
    """초안을 조회한다 (없으면 404)."""
    report = _draft_store.get(report_id)
    if report is None:
        raise OnRampError("보고서를 찾을 수 없습니다", status_code=404)
    return report


def update_report(report_id: str, update: AssetUpdateRequest) -> AssetResponse:
    """HITL 부분 수정 — None이 아닌 필드만 덮어쓴다 (없으면 404, published면 409)."""
    report = get_report(report_id)
    if report.status == "published":
        raise OnRampError("이미 등록된 보고서는 수정할 수 없습니다", status_code=409)
    data = report.model_dump()
    five = data["report"]

    if update.title is not None:
        data["title"] = update.title
    if update.category is not None:
        data["category"] = update.category
    for field in ("situation", "cause", "evidence", "solution", "infra_context"):
        value = getattr(update, field)
        if value is not None:
            five[field] = value

    data["updated_at"] = _now()
    updated = AssetResponse(**data)
    _draft_store[report_id] = updated
    return updated


async def approve_report(report_id: str) -> AssetApproveResponse:
    """현재 초안을 Confluence에 등록하고 published로 전환한다 (없으면 404, 이미 published면 409)."""
    report = get_report(report_id)
    if report.status == "published":
        raise OnRampError("이미 등록된 보고서입니다", status_code=409)
    html = _five_elements_to_wiki(report.report, report.category)
    page = await ConfluenceClient().create_page(title=report.title, html=html)

    data = report.model_dump()
    data["status"] = "published"
    data["confluence_url"] = page.url
    data["updated_at"] = _now()
    _draft_store[report_id] = AssetResponse(**data)
    return AssetApproveResponse(report_id=report_id, status="published", confluence_url=page.url)


def _five_elements_to_wiki(report: FiveElementsResponse, category: str) -> str:
    """5요소를 Confluence storage HTML(헤딩 구조)로 변환한다 — 인덱싱 청킹 친화적."""
    sections = [
        ("현재 상황", report.situation),
        ("원인", report.cause),
        ("근거", report.evidence),
        ("해결", report.solution),
        ("인프라 맥락", report.infra_context),
    ]
    parts = [f"<p><strong>분류</strong>: {_esc(category)}</p>"]
    for heading, content in sections:
        parts.append(f"<h2>{_esc(heading)}</h2><p>{_esc(content) or '-'}</p>")
    parts.append("<hr/><p><em>이 문서는 OnRamp에 의해 자동 생성되었습니다.</em></p>")
    return "".join(parts)


def _esc(text: str) -> str:
    return html_lib.escape(text or "").replace("\n", "<br/>")
