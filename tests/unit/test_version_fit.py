"""app/rag/version_fit.py — currency 모드 채점 (#103, 설계 4.1)."""

from app.config import Settings
from app.rag.version_fit import CURRENCY_MODE, compute_version_fit, currency_fit, version_fit_from_payload

S = Settings()  # 기본값: eol {apache:[2.2], k8s:[v1.18,v1.25,v1.29]}, 보수 캡 0.7, EOL 캡 0.3

K8S = frozenset({"v1.18", "v1.25", "v1.29", "v1.33"})


def test_currency_linear_rank_in_lineage() -> None:
    assert currency_fit("v1.18", "kubernetes", True, K8S, S) <= S.trust_eol_cap  # 0.25 → EOL 캡과 무관하게 낮음
    assert currency_fit("v1.33", "kubernetes", False, K8S, S) == 1.0
    assert currency_fit("v1.29", "kubernetes", False, K8S, S) == 0.75


def test_currency_single_lineage_single_version_site_is_full() -> None:
    # Datadog/Prometheus 'latest' — 형제 없음 = 그 문서가 곧 현행 → 1.0 (중립 0.5면 부당 감점)
    assert currency_fit("latest", "datadog", False, frozenset({"latest"}), S) == 1.0


def test_currency_single_lineage_multi_version_site_capped() -> None:
    # 다버전 site(apache)에서 단일 계보 = doc_key 정규화 실패 고아 의심 → 보수 캡
    assert currency_fit("2.4", "apache", False, frozenset({"2.4"}), S) == S.trust_single_lineage_cap


def test_eol_cap_applies_to_all_branches() -> None:
    """EOL 캡은 모든 분기의 최종 출력에 min — 워크스루 D (설계 v1.4 순서 규칙).

    고아가 된 Apache 2.2: 단일 계보 분기(0.7)를 타더라도 EOL이면 0.3으로 캡.
    순차 적용으로 구현하면 0.7을 받는 역설이 생긴다 — 이 테스트가 그 회귀를 막는다.
    """
    orphan = currency_fit("2.2", "apache", True, frozenset({"2.2"}), S)
    assert orphan == S.trust_eol_cap  # 0.7 → min(·, 0.3) = 0.3
    # 정상 계보의 EOL도 동일 캡
    assert currency_fit("2.2", "apache", True, frozenset({"2.2", "2.4"}), S) == S.trust_eol_cap


def test_currency_neutral_when_unknown() -> None:
    assert currency_fit("", "", False, frozenset(), S) == 0.5  # 라벨 없는 문서 → 중립
    assert currency_fit("2.4", "apache", False, frozenset(), S) == 0.5  # 계보 조회 실패 → 중립
    # 자기 버전이 계보에 없는 드문 불일치(캐시 직후 재색인) → 중립
    assert currency_fit("9.9", "apache", False, frozenset({"2.2", "2.4"}), S) == 0.5


def test_compute_version_fit_currency_mode() -> None:
    fit = compute_version_fit(
        product_version="v1.33", site="kubernetes", eol=False, lineage=K8S, target_versions=[], settings=S
    )
    assert fit.mode == CURRENCY_MODE
    assert fit.fit == fit.raw_currency == 1.0


def test_version_fit_from_payload_defaults() -> None:
    # payload에 메타가 없으면(재색인 전 stale 포인트 등) 중립 0.5 — 안전 동작
    fit = version_fit_from_payload({}, {}, [], S)
    assert fit.fit == 0.5


# ── match 모드 (#108) ────────────────────────────────────────────────


def test_match_exact_version_full_score() -> None:
    from app.rag.version_fit import MATCH_MODE

    fit = compute_version_fit(
        product_version="v1.25", site="kubernetes", eol=False, lineage=K8S, target_versions=["1.25"], settings=S
    )
    assert fit.mode == MATCH_MODE
    assert fit.fit == 1.0  # "1.25" ≡ "v1.25" 동치 매칭
    assert fit.raw_currency == 0.5  # raw currency는 모드와 무관하게 보존 (타이브레이커용)


def test_match_adjacent_partial_score() -> None:
    fit = compute_version_fit(
        product_version="v1.29", site="kubernetes", eol=False, lineage=K8S, target_versions=["1.25"], settings=S
    )
    assert fit.fit == S.trust_adjacent_version_fit  # 계보 정렬상 v1.25의 이웃


def test_match_distant_zero() -> None:
    fit = compute_version_fit(
        product_version="v1.18", site="kubernetes", eol=False, lineage=K8S, target_versions=["1.33"], settings=S
    )
    assert fit.fit == 0.0


def test_match_version_agnostic_doc_neutral() -> None:
    """버전 무관 문서(라벨 없음)는 match 모드에서 처벌하지 않는다 — 중립 0.5."""
    fit = compute_version_fit(
        product_version="", site="", eol=False, lineage=frozenset(), target_versions=["1.25"], settings=S
    )
    assert fit.fit == 0.5


def test_match_multi_target_comparison() -> None:
    """비교 질의: 요청 집합에 든 버전은 모두 1.0."""
    for v in ("v1.25", "v1.33"):
        fit = compute_version_fit(
            product_version=v,
            site="kubernetes",
            eol=False,
            lineage=K8S,
            target_versions=["1.25", "1.33"],
            settings=S,
        )
        assert fit.fit == 1.0
