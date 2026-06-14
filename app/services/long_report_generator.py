from __future__ import annotations

from collections.abc import Awaitable, Callable

from app.services.asset_service import GeneratedReport, generate_report_content

GenerateContent = Callable[[str, str, str, str], Awaitable[GeneratedReport]]


def split_transcript(transcript: str, *, max_chars: int, overlap_chars: int) -> list[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be between 0 and max_chars")
    if len(transcript) <= max_chars:
        return [transcript]

    windows: list[str] = []
    start = 0
    while start < len(transcript):
        end = min(start + max_chars, len(transcript))
        if end < len(transcript):
            boundary = transcript.rfind("\n", start + max_chars // 2, end)
            if boundary > start:
                end = boundary + 1
        windows.append(transcript[start:end])
        if end == len(transcript):
            break
        start = max(end - overlap_chars, start + 1)
    return windows


class WindowedReportGenerator:
    def __init__(
        self,
        *,
        max_chars: int,
        overlap_chars: int,
        merge_batch_size: int = 4,
        generate_content: GenerateContent = generate_report_content,
    ) -> None:
        if merge_batch_size < 2:
            raise ValueError("merge_batch_size must be at least 2")
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.merge_batch_size = merge_batch_size
        self.generate_content = generate_content

    async def __call__(self, transcript: str, category: str, title: str) -> GeneratedReport:
        windows = split_transcript(
            transcript,
            max_chars=self.max_chars,
            overlap_chars=self.overlap_chars,
        )
        if len(windows) == 1:
            return await self.generate_content(transcript, category, title, "")

        partials = [await self.generate_content(window, category, "", "") for window in windows]
        while len(partials) > 1:
            next_level: list[GeneratedReport] = []
            for offset in range(0, len(partials), self.merge_batch_size):
                batch = partials[offset : offset + self.merge_batch_size]
                if len(batch) == 1:
                    next_level.append(batch[0])
                    continue
                merged_input = "\n\n".join(
                    _render_partial(index, partial) for index, partial in enumerate(batch, start=1)
                )
                final_merge = len(partials) <= self.merge_batch_size
                next_level.append(
                    await self.generate_content(
                        "다음은 긴 녹취를 구간별로 구조화한 부분 보고서다. "
                        "중복을 제거하고 하나의 보고서로 병합하라.\n\n" + merged_input,
                        category,
                        title if final_merge else "",
                        "",
                    )
                )
            partials = next_level
        return partials[0]


def _render_partial(index: int, partial: GeneratedReport) -> str:
    report = partial.report
    return "\n".join(
        [
            f"[부분 보고서 {index}]",
            f"title: {partial.title}",
            f"situation: {report.situation}",
            f"cause: {report.cause}",
            f"evidence: {report.evidence}",
            f"solution: {report.solution}",
            f"infra_context: {report.infra_context}",
        ]
    )
