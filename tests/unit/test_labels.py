"""app/rag/labels.py — Confluence 라벨 → 버전 메타 파싱 (#94)."""

import pytest

from app.rag.labels import (
    is_eol,
    latest_version,
    make_doc_key,
    parse_product_version,
    parse_site,
    slugify,
    strip_upload_suffix,
    version_sort_key,
    versions_equal,
)

# ── strip_upload_suffix ──────────────────────────────────────────────


def test_strip_upload_suffix_removes_run_id_hash() -> None:
    assert strip_upload_suffix("Content Negotiation [a78792-639072]") == "Content Negotiation"


def test_strip_upload_suffix_keeps_title_without_suffix() -> None:
    assert strip_upload_suffix("EKS Pod CrashLoopBackOff 대응 런북") == "EKS Pod CrashLoopBackOff 대응 런북"


def test_strip_upload_suffix_keeps_non_hex_brackets() -> None:
    # OpenMetrics 2.0 [EXPERIMENTAL] 같은 본문 대괄호는 suffix가 아니다
    assert strip_upload_suffix("OpenMetrics 2.0 [EXPERIMENTAL]") == "OpenMetrics 2.0 [EXPERIMENTAL]"


# ── doc_key ──────────────────────────────────────────────────────────


def test_make_doc_key_joins_version_siblings() -> None:
    # 검증된 실데이터: 2.4와 2.2 형제가 같은 키로 묶여야 collapse가 동작한다
    newer = make_doc_key("apache", "Content Negotiation [a78792-639072]")
    older = make_doc_key("apache", "Content Negotiation [a78792-4dddd3]")
    assert newer == older == "apache:content-negotiation"


def test_make_doc_key_without_site_is_empty() -> None:
    assert make_doc_key("", "어떤 제목") == ""


def test_slugify_collapses_whitespace_and_lowercases() -> None:
    assert slugify("  Apache   MPM Worker ") == "apache-mpm-worker"


# ── 라벨 파싱 ────────────────────────────────────────────────────────


def test_parse_site_first_match() -> None:
    assert parse_site(["auto-imported", "site-prometheus", "version-latest"]) == "prometheus"


def test_parse_site_missing_returns_empty() -> None:
    assert parse_site(["auto-imported", "trustrag"]) == ""


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (["version-2-4"], "2.4"),
        (["version-v1-25"], "v1.25"),
        (["version-latest"], "latest"),
        (["category-mod", "source-git"], ""),  # version- prefix 없는 라벨은 버전 아님
        (["category-v1-25"], ""),  # k8s 변칙: category 라벨을 버전으로 오해석 금지
        (["version-???"], ""),  # 규칙 밖 토큰은 무시(경고 로그)
    ],
)
def test_parse_product_version(labels: list[str], expected: str) -> None:
    assert parse_product_version(labels) == expected


# ── EOL / 정렬 / 동치 ────────────────────────────────────────────────


def test_is_eol_uses_site_map() -> None:
    eol_map = {"apache": ["2.2"], "kubernetes": ["v1.18"]}
    assert is_eol("apache", "2.2", eol_map) is True
    assert is_eol("apache", "2.4", eol_map) is False
    assert is_eol("", "2.2", eol_map) is False  # site 불명 → 판정 불가


def test_version_sort_key_orders_versions() -> None:
    assert sorted(["v1.33", "v1.18", "v1.25"], key=version_sort_key) == ["v1.18", "v1.25", "v1.33"]
    assert sorted(["2.4", "2.2"], key=version_sort_key) == ["2.2", "2.4"]


def test_version_sort_key_latest_is_highest() -> None:
    assert sorted(["latest", "2.4"], key=version_sort_key)[-1] == "latest"


def test_versions_equal_normalizes_leading_v() -> None:
    assert versions_equal("1.25", "v1.25") is True
    assert versions_equal("1.25", "1.33") is False
    assert versions_equal("", "1.25") is False
    assert versions_equal("alpha", "beta") is False  # 숫자 없는 토큰끼리 오결합 금지


def test_latest_version() -> None:
    assert latest_version({"2.2", "2.4"}) == "2.4"
    assert latest_version({"latest", "2.4"}) == "latest"
    assert latest_version([]) == ""
