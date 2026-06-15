from __future__ import annotations

import signal

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


def test_split_transcript_always_advances_with_large_overlap() -> None:
    transcript = ("가" * 50 + "\n") * 10

    def fail_on_timeout(signum: int, frame: object) -> None:
        raise TimeoutError("split_transcript did not advance")

    previous_handler = signal.signal(signal.SIGALRM, fail_on_timeout)
    signal.alarm(1)
    try:
        windows = split_transcript(transcript, max_chars=100, overlap_chars=99)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)

    assert windows
    assert "".join(window if index == 0 else window[99:] for index, window in enumerate(windows))


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


@pytest.mark.asyncio
async def test_long_transcript_merges_partials_in_bounded_batches() -> None:
    calls: list[str] = []

    async def generate(transcript: str, category: str, title: str = "", model: str = "") -> GeneratedReport:
        calls.append(transcript)
        return GeneratedReport(
            title=title or "부분",
            report=FiveElementsResponse(
                situation="상황",
                cause="원인",
                evidence="근거",
                solution="해결",
                infra_context="환경",
            ),
        )

    generator = WindowedReportGenerator(
        max_chars=30,
        overlap_chars=5,
        merge_batch_size=2,
        generate_content=generate,
    )

    result = await generator("문단입니다.\n" * 40, "장애대응", "최종 제목")

    merge_calls = [call for call in calls if "[부분 보고서" in call]
    assert len(merge_calls) > 1
    assert all(call.count("[부분 보고서") <= 2 for call in merge_calls)
    assert result.title == "최종 제목"
