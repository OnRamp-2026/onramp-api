"""Mask sensitive Markdown before chunking and embedding."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


class MarkdownMasker:
    """Remove or normalize sensitive values from Markdown text."""

    _EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    _PRIVATE_IP_PATTERN = re.compile(
        r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
        r"172\.(?:1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3}|"
        r"192\.168\.\d{1,3}\.\d{1,3})\b"
    )
    _JWT_PATTERN = re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")
    _LONG_SECRET_PATTERN = re.compile(r"\b[A-Za-z0-9_./+=-]{32,}\b")
    _AUTH_HEADER_PATTERN = re.compile(
        r"(?im)^(\s*authorization\s*:\s*(?:bearer|basic)\s+)([^\s`]+)",
    )
    _SECRET_ASSIGNMENT_PATTERN = re.compile(
        r"(?im)^(\s*(?:api[_-]?key|token|secret|password|passwd|access[_-]?key|client[_-]?secret)\s*[:=]\s*)(.+)$"
    )

    def mask(self, markdown: str) -> str:
        """Return Markdown with sensitive values replaced by stable placeholders."""

        if not markdown.strip():
            logger.warning("Received empty Markdown for masking")
            return ""

        masked = markdown
        masked = self._AUTH_HEADER_PATTERN.sub(r"\1[MASKED_TOKEN]", masked)
        masked = self._SECRET_ASSIGNMENT_PATTERN.sub(r"\1[MASKED_SECRET]", masked)
        masked = self._JWT_PATTERN.sub("[MASKED_TOKEN]", masked)
        masked = self._EMAIL_PATTERN.sub("[MASKED_EMAIL]", masked)
        masked = self._PRIVATE_IP_PATTERN.sub("[MASKED_PRIVATE_IP]", masked)
        masked = self._mask_long_secret_like_values(masked)
        return self._normalize(masked)

    def _mask_long_secret_like_values(self, text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            value = match.group(0)
            if re.fullmatch(r"\d+", value):
                return value
            return "[MASKED_SECRET]"

        return self._LONG_SECRET_PATTERN.sub(replace, text)

    def _normalize(self, markdown: str) -> str:
        markdown = markdown.replace("\xa0", " ")
        markdown = re.sub(r"[ \t]+\n", "\n", markdown)
        markdown = re.sub(r"\n{3,}", "\n\n", markdown)
        markdown = self._balance_code_fences(markdown)
        lines = [line.rstrip() for line in markdown.splitlines()]
        return "\n".join(lines).strip() + "\n" if any(line.strip() for line in lines) else ""

    def _balance_code_fences(self, markdown: str) -> str:
        fence_count = sum(1 for line in markdown.splitlines() if re.match(r"^[ \t]{0,3}```", line))
        if fence_count % 2 == 0:
            return markdown
        return markdown.rstrip() + "\n```\n"
