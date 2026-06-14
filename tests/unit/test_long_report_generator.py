from __future__ import annotations

import pytest

from app.models.response import FiveElementsResponse
from app.services.asset_service import GeneratedReport
from app.services.long_report_generator import WindowedReportGenerator, split_transcript


def test_split_transcript_uses_overlap_without_empty_windows() -> None:
    transcript = "\n".join(f"문장 {index}: " + ("가" * 20) for index in range(10))

    windows = split_transcript(transcript, max_chars=90, overlap_chars=20)

    assert len(windows) > 1
    assert all(window.strip() for window in windows)
    assert all(len(window) <= 110 for window in windows)
    assert any(windows[index][-20:] in windows[index + 1] for index in range(len(windows) - 1))


@pytest.mark.asyncio
async def test_short_transcript_uses_single_generation_call() -> None:
    calls: list[str] = []

    async def generate(transcript: str, category: str, title: str = "", model: str = "") -> GeneratedReport:
        calls.append(transcript)
        return GeneratedReport(title=title or "제목", report=FiveElementsResponse(situation="상황"))

    generator = WindowedReportGenerator(max_chars=100, overlap_chars=10, generate_content=generate)

    result = await generator("짧은 녹취", "회의록", "회의")

    assert result.title == "회의"
    assert calls == ["짧은 녹취"]


@pytest.mark.asyncio
async def test_long_transcript_generates_windows_then_final_merge() -> None:
    calls: list[str] = []

    async def generate(transcript: str, category: str, title: str = "", model: str = "") -> GeneratedReport:
        calls.append(transcript)
        index = len(calls)
        return GeneratedReport(
            title=title or f"부분 {index}",
            report=FiveElementsResponse(
                situation=f"상황 {index}",
                cause=f"원인 {index}",
                evidence=f"근거 {index}",
                solution=f"해결 {index}",
                infra_context=f"환경 {index}",
            ),
        )

    generator = WindowedReportGenerator(max_chars=50, overlap_chars=10, generate_content=generate)

    result = await generator("첫 문단입니다.\n" * 20, "장애대응", "최종 제목")

    assert len(calls) > 2
    assert "[부분 보고서 1]" in calls[-1]
    assert result.title == "최종 제목"
