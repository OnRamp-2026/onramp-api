"""`/slack/events` — Slack 봇 (Events API). 멘션·DM → RAG 답변 회신.

로그인(OIDC, #98)과 **완전 별개**. 신규 입사자·웹앱 미접속자가 Slack에서 봇을 멘션하거나
DM으로 질문하면 `/v1/chat`(RAG)을 그대로 호출해 5요소 답변을 돌려준다. (#146)
채널 멘션(`app_mention`)은 원 메시지 스레드로, DM(`message`+`channel_type=im`)은 평문으로 회신.

흐름: Slack 이벤트 → 서명 검증 → **3초 내 200 ack** → 백그라운드에서 chat → `chat.postMessage`.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import time
from typing import Any

import httpx
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request, Response, status

from app.config import Settings, get_settings
from app.models.request import ChatRequest
from app.models.response import ChatResponse
from app.services.chat_service import chat as chat_service

logger = logging.getLogger(__name__)
router = APIRouter()

SLACK_POST_MESSAGE = "https://slack.com/api/chat.postMessage"
_MENTION_RE = re.compile(r"<@[A-Z0-9]+>")  # 봇 멘션 토큰 제거
_MAX_SKEW_SECONDS = 60 * 5  # 서명 timestamp 허용 오차(replay 방지)


def _require_bot(settings: Settings) -> None:
    if not settings.slack_bot_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found.")
    if not settings.slack_signing_secret.get_secret_value() or not settings.slack_bot_token.get_secret_value():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Slack 봇이 구성되지 않았습니다.")


def _verify_signature(body: bytes, timestamp: str, signature: str, signing_secret: str) -> bool:
    """Slack 서명 검증 (X-Slack-Signature = v0=HMAC-SHA256(secret, 'v0:ts:body'))."""
    try:
        if abs(time.time() - int(timestamp)) > _MAX_SKEW_SECONDS:
            return False  # replay
    except ValueError:
        return False
    base = b"v0:" + timestamp.encode() + b":" + body
    computed = "v0=" + hmac.new(signing_secret.encode(), base, hashlib.sha256).hexdigest()
    return hmac.compare_digest(computed, signature)


@router.post("/events")
async def events(
    request: Request,
    background: BackgroundTasks,
    x_slack_signature: str = Header(default=""),
    x_slack_request_timestamp: str = Header(default=""),
    x_slack_retry_num: str = Header(default=""),
) -> Response:
    settings = get_settings()
    _require_bot(settings)
    body = await request.body()

    if not _verify_signature(
        body, x_slack_request_timestamp, x_slack_signature, settings.slack_signing_secret.get_secret_value()
    ):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Slack 서명 검증 실패.")

    try:
        payload: dict[str, Any] = json.loads(body)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="잘못된 페이로드.") from exc

    # Events API Request URL 등록 시 챌린지
    if payload.get("type") == "url_verification":
        return Response(content=str(payload.get("challenge", "")), media_type="text/plain")

    # Slack 재시도(3초 ack 실패 시 재전송) → 중복 답변 방지: 즉시 ack만
    if x_slack_retry_num:
        return Response(status_code=status.HTTP_200_OK)

    event = payload.get("event", {})
    # 봇 자신/다른 봇 메시지·서브타입(편집·입장·봇메시지 등)은 무시 — DM 루프 방지의 핵심
    if not event.get("bot_id") and event.get("subtype") is None:
        event_type = event.get("type")
        # 채널 멘션 또는 봇과의 1:1 DM만 처리(채널 일반 메시지는 소음이라 제외)
        if event_type == "app_mention" or (event_type == "message" and event.get("channel_type") == "im"):
            background.add_task(_handle_message, event, settings)

    return Response(status_code=status.HTTP_200_OK)  # 3초 내 ack (실처리는 백그라운드)


async def _handle_message(event: dict[str, Any], settings: Settings) -> None:
    question = _MENTION_RE.sub("", str(event.get("text", ""))).strip()
    channel = str(event.get("channel", ""))
    # DM은 스레드 없이 평문 회신, 채널 멘션은 원 메시지 스레드로
    thread_ts = "" if event.get("channel_type") == "im" else str(event.get("thread_ts") or event.get("ts") or "")
    if not question or not channel:
        return
    try:
        result = await chat_service(ChatRequest(query=question))
        text = _format_answer(result)
    except Exception:
        logger.exception("Slack 멘션 처리 실패")
        text = "죄송합니다, 답변 생성 중 오류가 발생했습니다."
    await _post_message(channel, thread_ts, text, settings)


def _format_answer(r: ChatResponse) -> str:
    """답변 + 출처를 Slack mrkdwn으로. 라우터 포맷(#191)에 따라 분기 — freeform 산문 / 5요소 구조화."""
    if r.answer_format == "freeform":
        body = r.answer_text.strip() or "관련 근거를 충분히 찾지 못했습니다."
    else:
        a = r.answer
        sections = [
            ("현재 상황", a.situation),
            ("원인", a.cause),
            ("근거", a.evidence),
            ("해결", a.solution),
            ("인프라 맥락", a.infra_context),
        ]
        parts = [f"*{label}*\n{value}" for label, value in sections if value]
        body = "\n\n".join(parts) or "관련 근거를 충분히 찾지 못했습니다."
    links = [f"• <{s.url}|{s.title or s.url}>" for s in r.sources[:3] if s.url]
    if links:
        body += "\n\n*출처*\n" + "\n".join(links)
    return body


async def _post_message(channel: str, thread_ts: str, text: str, settings: Settings) -> None:
    message: dict[str, Any] = {"channel": channel, "text": text}
    if thread_ts:  # DM은 thread_ts 없음 → 평문 회신(빈 값 전송 시 Slack이 거부)
        message["thread_ts"] = thread_ts
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            res = await client.post(
                SLACK_POST_MESSAGE,
                headers={"Authorization": f"Bearer {settings.slack_bot_token.get_secret_value()}"},
                json=message,
            )
    except httpx.HTTPError:
        logger.exception("Slack chat.postMessage 통신 실패")
        return
    try:
        payload = res.json()
    except ValueError:
        logger.error("chat.postMessage 비정상 응답(non-JSON): %s", res.text[:200])
        return
    if not payload.get("ok", False):
        logger.error("chat.postMessage 실패: %s", payload.get("error"))
