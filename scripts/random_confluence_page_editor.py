"""Randomly update Confluence pages with an OnRamp cleaner test section."""

from __future__ import annotations

import argparse
import asyncio
import html
import logging
import random
import re
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from app.config import get_settings
from app.db.confluence import ConfluenceClient

logger = logging.getLogger(__name__)

SECTION_START = "<!-- ONRAMP_TEST_SECTION_START -->"
SECTION_END = "<!-- ONRAMP_TEST_SECTION_END -->"
SECTION_PATTERN = re.compile(rf"{re.escape(SECTION_START)}.*?{re.escape(SECTION_END)}", flags=re.DOTALL)


@dataclass(frozen=True)
class PageUpdatePreview:
    page_id: str
    title: str
    version: int
    next_version: int
    url: str


def _upsert_test_section(existing_html: str, page_title: str, timestamp: str) -> str:
    section = _build_test_section(page_title, timestamp)
    if SECTION_START in existing_html and SECTION_END in existing_html:
        return SECTION_PATTERN.sub(section, existing_html)
    return f"{existing_html}{section}"


def _build_test_section(page_title: str, timestamp: str) -> str:
    escaped_title = html.escape(page_title)
    escaped_timestamp = html.escape(timestamp)
    return f"""
{SECTION_START}
<hr />
<h2>OnRamp 정제 테스트</h2>
<p>테스트 수정 시각: {escaped_timestamp}</p>
<p>이 섹션은 OnRamp recent fetcher와 TextCleaner 검증을 위해 자동으로 갱신된다.</p>
<h3>점검 항목</h3>
<ul>
  <li><p>코드블록 보존</p></li>
  <li><p>표 구조 보존</p></li>
  <li><p>내부/외부 링크 변환 확인</p></li>
</ul>
<ol>
  <li><p>최근 수정 페이지 조회</p></li>
  <li><p>Storage HTML 저장</p></li>
  <li><p>Markdown 정제 결과 비교</p></li>
</ol>
<ac:structured-macro ac:name="code">
  <ac:parameter ac:name="breakoutMode">wide</ac:parameter>
  <ac:parameter ac:name="breakoutWidth">760</ac:parameter>
  <ac:plain-text-body><![CDATA[kubectl get pod <pod-name>
]]></ac:plain-text-body>
</ac:structured-macro>
<ac:structured-macro ac:name="code">
  <ac:plain-text-body><![CDATA[curl -X GET https://example.internal/health
]]></ac:plain-text-body>
</ac:structured-macro>
<table>
  <tbody>
    <tr><th><p>항목</p></th><th><p>기대 결과</p></th></tr>
    <tr><td><p>코드블록</p></td><td><p>wide760 같은 레이아웃 값이 제거된다.</p></td></tr>
    <tr><td><p>링크</p></td><td><p>외부 URL은 본문에 남는다.</p></td></tr>
  </tbody>
</table>
<p><a href="#OnRamp-정제-테스트">내부 링크 예시: {escaped_title}</a></p>
<p><a href="https://example.com/onramp-cleaner-test">외부 링크 예시</a></p>
{SECTION_END}
"""


async def update_random_pages(count: int, candidate_limit: int, seed: int | None, apply: bool) -> list[PageUpdatePreview]:
    settings = get_settings()
    confluence = ConfluenceClient(settings=settings)
    candidates = await confluence.fetch_candidate_pages(limit=candidate_limit)
    if not candidates:
        logger.info("No candidate pages found in space %s", settings.confluence_space_key)
        return []

    rng = random.Random(seed)
    selected = rng.sample(candidates, k=min(count, len(candidates)))
    timestamp = datetime.now(ZoneInfo(settings.confluence_timezone)).strftime("%Y-%m-%d %H:%M %Z")
    previews: list[PageUpdatePreview] = []

    for page in selected:
        if page.version is None:
            logger.warning("Skipping page %s because version is missing", page.page_id)
            continue

        next_version = page.version + 1
        previews.append(
            PageUpdatePreview(
                page_id=page.page_id,
                title=page.title,
                version=page.version,
                next_version=next_version,
                url=page.url,
            )
        )

        if apply:
            await confluence.update_page(page, _upsert_test_section(page.html, page.title, timestamp), next_version)
            logger.info("Updated %s v%s -> v%s %s", page.page_id, page.version, next_version, page.title)
        else:
            logger.info("Dry run: would update %s v%s -> v%s %s", page.page_id, page.version, next_version, page.title)

    return previews


def main() -> None:
    parser = argparse.ArgumentParser(description="Randomly update Confluence pages with an OnRamp cleaner test section.")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--candidate-limit", type=int, default=100)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    previews = asyncio.run(
        update_random_pages(
            count=args.count,
            candidate_limit=args.candidate_limit,
            seed=args.seed,
            apply=args.apply,
        )
    )
    for preview in previews:
        mode = "updated" if args.apply else "would update"
        logger.info("%s: %s v%s -> v%s %s", mode, preview.page_id, preview.version, preview.next_version, preview.title)
        if preview.url:
            logger.info("url: %s", preview.url)


if __name__ == "__main__":
    main()
