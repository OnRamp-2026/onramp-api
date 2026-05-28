"""Clean Confluence storage HTML into RAG-friendly Markdown."""

from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Comment
from bs4.element import CData, NavigableString, Tag
from markdownify import markdownify as md

logger = logging.getLogger(__name__)

COMMON_NOISE_SELECTORS = (
    "nav",
    "header",
    "footer",
    "aside",
    "script",
    "style",
    ".cookie-banner",
)

KNOWN_DOC_NOISE_SELECTORS = (
    ".td-sidebar",
    ".td-toc",
    ".feedback-widget",
    "#pre-footer",
    ".navbar",
    ".footer",
    ".edit-page",
    "#left-column",
    "#footer",
    "[data-nav]",
    ".sidebar",
    ".feedback",
)


class TextCleaner:
    """Convert noisy HTML into Markdown suitable for chunking and retrieval."""

    def clean(self, html: str) -> str:
        """Clean HTML and return normalized Markdown."""

        if not html.strip():
            logger.warning("Received empty HTML")
            return ""

        soup = BeautifulSoup(html, "html.parser")
        soup = self._remove_noise(soup)
        soup = self._preserve_confluence_code_macros(soup)
        soup = self._remove_confluence_noise_macros(soup)
        soup = self._preserve_code_blocks(soup)
        self._convert_images(soup)
        self._convert_links(soup)
        markdown = self._to_markdown(soup)
        return self._postprocess(markdown)

    def _remove_noise(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Remove known non-content elements from documentation pages."""

        for selector in (*COMMON_NOISE_SELECTORS, *KNOWN_DOC_NOISE_SELECTORS):
            for tag in soup.select(selector):
                tag.decompose()

        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        return soup

    def _preserve_confluence_code_macros(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Convert Confluence code macros to normal pre/code blocks."""

        for macro in soup.find_all("ac:structured-macro"):
            if not isinstance(macro, Tag) or macro.get("ac:name") != "code":
                continue

            body = macro.find("ac:plain-text-body")
            code_text = body.get_text() if body else ""
            pre_tag = soup.new_tag("pre")
            code_tag = soup.new_tag("code")
            code_tag.string = CData(code_text.rstrip("\n"))
            pre_tag.append(code_tag)
            macro.replace_with(pre_tag)

        return soup

    def _remove_confluence_noise_macros(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Remove Confluence macros that usually render navigation or layout noise."""

        for macro in soup.find_all("ac:structured-macro"):
            if isinstance(macro, Tag):
                macro.decompose()

        for tag in soup.find_all("ac:layout"):
            tag.unwrap()

        for tag in soup.find_all(("ac:layout-section", "ac:layout-cell")):
            tag.unwrap()

        return soup

    def _preserve_code_blocks(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Normalize pre/code blocks so markdownify keeps them as fenced code."""

        for pre_tag in soup.find_all("pre"):
            if not isinstance(pre_tag, Tag):
                continue
            code_text = pre_tag.get_text()
            pre_tag.clear()
            code_tag = soup.new_tag("code")
            code_tag.string = code_text.rstrip("\n")
            pre_tag.append(code_tag)

        return soup

    def _to_markdown(self, soup: BeautifulSoup) -> str:
        """Convert cleaned HTML into Markdown."""

        return md(
            str(soup),
            heading_style="ATX",
            bullets="-",
            strip=("script", "style"),
        )

    def _postprocess(self, markdown: str) -> str:
        """Normalize spacing while preserving Markdown block structure."""

        markdown = markdown.replace("\xa0", " ")
        markdown = re.sub(r"(?m)^#[0-9A-Fa-f]{6}\s*$\n?", "", markdown)
        markdown = re.sub(r"(?m)^page,whiteboard,database,blog\d+concisetrue\s*$\n?", "", markdown)
        markdown = re.sub(r"[ \t]+\n", "\n", markdown)
        markdown = re.sub(r"\n{3,}", "\n\n", markdown)
        lines = [line.rstrip() for line in markdown.splitlines()]
        return "\n".join(lines).strip() + "\n" if any(line.strip() for line in lines) else ""

    def _convert_images(self, soup: BeautifulSoup) -> None:
        """Replace images with Korean alt text markers for retrieval context."""

        for image in soup.find_all("img"):
            alt_text = image.get("alt", "").strip()
            replacement = f"[이미지: {alt_text}]" if alt_text else ""
            image.replace_with(NavigableString(replacement))

    def _convert_links(self, soup: BeautifulSoup) -> None:
        """Keep external link URLs and collapse internal links to visible text."""

        for anchor in soup.find_all("a"):
            href = anchor.get("href", "").strip()
            text = anchor.get_text(" ", strip=True)
            if not text:
                anchor.decompose()
                continue

            replacement = f"{text} ({href})" if self._is_external_link(href) else text
            anchor.replace_with(NavigableString(replacement))

    def _is_external_link(self, href: str) -> bool:
        """Return True for absolute HTTP(S) links."""

        parsed = urlparse(href)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
