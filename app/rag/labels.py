"""Confluence 라벨 → 버전 메타 파싱 (Trust 재설계 #94 선행).

크롤러 업로더가 페이지에 남긴 라벨(site-*, version-*)을 색인 페이로드 필드로 변환한다.
순수 함수만 둔다 — 외부 의존 없음(re·logging만).

규칙:
    site            "site-apache" → "apache" (첫 매치)
    product_version "version-" prefix 라벨만 인정 — category-*/source-* 등 다른 라벨을
                    버전으로 오해석하지 않는다 (k8s의 category-v1-25 변칙 방어).
                    "version-2-4" → "2.4" / "version-v1-25" → "v1.25" / "version-latest" → "latest"
    doc_key         site + 업로드 suffix 제거한 제목 슬러그. 버전 형제(같은 문서의 버전별
                    페이지)를 묶는 키 — 제목은 같고 suffix만 다르다.
                    예: "Content Negotiation [a78792-639072]"(2.4) ↔ "[a78792-4dddd3]"(2.2)
                    → 둘 다 "apache:content-negotiation"
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence

logger = logging.getLogger(__name__)

# 업로더가 title 유니크 보장을 위해 붙이는 suffix: "[<run-id 6hex>-<path-hash 6hex>]"
UPLOAD_SUFFIX_RE = re.compile(r"\s*\[[0-9a-f]{6}-[0-9a-f]{6}\]\s*$")

_SITE_PREFIX = "site-"
_VERSION_PREFIX = "version-"
# prefix 제거 후 허용 토큰: "latest" 또는 "v1-25"/"2-4" 류 숫자 시퀀스
_VERSION_TOKEN_RE = re.compile(r"^v?\d+(-\d+)*$")
# 버전 문자열("v1.25"/"2.4")의 숫자 추출용
_VERSION_NUMBERS_RE = re.compile(r"\d+")


def strip_upload_suffix(title: str) -> str:
    """업로드 suffix를 제거한 원 제목."""
    return UPLOAD_SUFFIX_RE.sub("", title).strip()


def slugify(text: str) -> str:
    """공백 연속 → '-' + 소문자화. 그 외 문자는 보존(과정규화로 인한 오결합 방지)."""
    return re.sub(r"\s+", "-", text.strip().lower())


def make_doc_key(site: str, title: str) -> str:
    """버전 형제를 묶는 키. site 없으면 빈 문자열 — Trust가 page 단위로 폴백 처리한다."""
    if not site:
        return ""
    return f"{site}:{slugify(strip_upload_suffix(title))}"


def parse_site(labels: Sequence[str]) -> str:
    """ "site-apache" → "apache". 첫 매치, 없으면 ""."""
    for label in labels:
        if label.startswith(_SITE_PREFIX) and len(label) > len(_SITE_PREFIX):
            return label[len(_SITE_PREFIX) :]
    return ""


def parse_product_version(labels: Sequence[str]) -> str:
    """ "version-2-4" → "2.4", "version-v1-25" → "v1.25", "version-latest" → "latest".

    version- prefix가 없는 라벨은 절대 버전으로 해석하지 않는다.
    prefix는 있는데 토큰이 규칙 밖이면 "" + 경고 로그 (조용한 오염 방지).
    """
    for label in labels:
        if not label.startswith(_VERSION_PREFIX):
            continue
        token = label[len(_VERSION_PREFIX) :]
        if token == "latest":
            return "latest"
        if _VERSION_TOKEN_RE.match(token):
            return token.replace("-", ".")
        logger.warning("해석 불가 version 라벨 무시: %r", label)
    return ""


def is_eol(site: str, product_version: str, eol_map: dict[str, list[str]]) -> bool:
    """EOL 맵(config.eol_versions) 기반 판정. site/버전 불명이면 False(보수적이지 않은 방향이지만
    버전 축이 중립 처리되므로 안전 — 설계 문서 v1.5 4.1)."""
    if not site or not product_version:
        return False
    return product_version in eol_map.get(site, [])


def version_sort_key(version: str) -> tuple:
    """계보 내 버전 정렬 키. "latest"는 항상 최상위, 그 외는 숫자 튜플.

    site 무관 일반 규칙으로 충분 — 현 코퍼스(apache 2.2/2.4, k8s v1.18~v1.33) 검증.
    숫자가 없으면 최하위(미상 버전이 latest로 오인되지 않게).
    """
    if version == "latest":
        return (float("inf"),)
    numbers = [int(n) for n in _VERSION_NUMBERS_RE.findall(version)]
    return tuple(numbers) if numbers else (-1,)


def versions_equal(a: str, b: str) -> bool:
    """정규화 동치: "1.25" ≡ "v1.25" (Router 추출값 ↔ payload 값 매칭용)."""
    if not a or not b:
        return False
    if a == b:
        return True
    ka, kb = version_sort_key(a), version_sort_key(b)
    return ka == kb and ka != (-1,)


def latest_version(versions: Iterable[str]) -> str:
    """계보에서 최신 버전. 빈 입력이면 ""."""
    ordered = sorted(versions, key=version_sort_key)
    return ordered[-1] if ordered else ""
