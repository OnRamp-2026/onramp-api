"""version_fit — 버전 적합성 단일 축 (#103, 설계 문서 4.1).

질의에 버전 명시가 있으면 match 모드, 없으면 currency 모드(절대적 최신성)로 채점한다.
랭킹 부스트(retriever)와 collapse 승자 선정(trust)이 **같은 함수**를 사용해
두 단계의 판단이 모순되지 않게 한다(설계 7.3).

#103 범위는 currency 모드만 — match 모드(target_versions)는 Trust 재설계 본체에서 추가한다.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

from app.config import Settings
from app.rag.labels import version_sort_key

CURRENCY_MODE = "currency"
MATCH_MODE = "match"  # Trust 재설계 본체에서 사용


@dataclass(frozen=True)
class VersionFit:
    """버전 적합성 채점 결과 (전부 [0,1])."""

    fit: float  # overall·랭킹 부스트·collapse가 쓰는 유일한 값
    mode: str  # "currency" | "match" — 관측·디버깅용
    raw_currency: float  # 디버그 + collapse 타이브레이커 (match 모드 동률용)


def currency_fit(
    product_version: str,
    site: str,
    eol: bool,
    lineage: Collection[str],
    settings: Settings,
) -> float:
    """currency 모드 — 자기 계보(doc_key) 안에서 얼마나 최신 버전인가.

    분기:
        계보/버전 불명           → 0.5 중립 (라벨 없는 문서가 부당 감점되지 않게)
        단일 계보 + 단일버전 site → 1.0 (형제가 없다 = 그 문서가 곧 현행)
        단일 계보 + 다버전 site   → 보수 캡 (doc_key 정규화 실패 고아가 만점 받는 경로 차단)
        다버전 계보              → 선형 순위 (v1.18→0.25, v1.33→1.0)

    EOL 캡은 **모든 분기의 최종 출력에** min으로 적용한다 — 순차 적용으로 구현하면
    고아가 된 EOL 문서(보수 캡 분기)에서 EOL 캡이 무력화되는 역설이 생긴다(설계 v1.4).
    """
    if not product_version or not lineage:
        base = 0.5
    elif len(lineage) == 1:
        base = settings.trust_single_lineage_cap if site in settings.multi_version_sites else 1.0
    else:
        ordered = sorted(lineage, key=version_sort_key)
        try:
            base = (ordered.index(product_version) + 1) / len(ordered)
        except ValueError:  # 자기 버전이 계보에 없음(캐시 직후 재색인 등 드문 불일치) → 중립
            base = 0.5
    return min(base, settings.trust_eol_cap) if eol else base


def compute_version_fit(
    *,
    product_version: str,
    site: str,
    eol: bool,
    lineage: Collection[str],
    target_versions: Sequence[str],
    settings: Settings,
) -> VersionFit:
    """문서 한 건의 버전 적합성. target_versions가 있으면 match 모드(후속), 없으면 currency.

    #103에서는 target_versions가 항상 비어 있어 currency 모드만 동작한다 —
    match 모드 분기는 Trust 재설계 본체(#다음 이슈)에서 추가된다.
    """
    raw_currency = currency_fit(product_version, site, eol, lineage, settings)
    return VersionFit(fit=raw_currency, mode=CURRENCY_MODE, raw_currency=raw_currency)


def version_fit_from_payload(
    payload: dict,
    lineages: dict[str, frozenset[str]],
    target_versions: Sequence[str],
    settings: Settings,
) -> VersionFit:
    """Qdrant payload에서 버전 메타를 꺼내 채점한다 (랭킹 부스트 경로의 진입점)."""
    doc_key = payload.get("doc_key", "") or ""
    return compute_version_fit(
        product_version=payload.get("product_version", "") or "",
        site=payload.get("site", "") or "",
        eol=bool(payload.get("is_eol", False)),
        lineage=lineages.get(doc_key, frozenset()),
        target_versions=target_versions,
        settings=settings,
    )
