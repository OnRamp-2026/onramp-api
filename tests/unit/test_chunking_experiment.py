"""청킹 A/B 실험 헬퍼 단위 테스트 (#212) — 순수 함수, 인프라 불필요."""

import pytest

from app.eval.chunking_experiment import ChunkingConfig, chunk_page, page_from_row

_MD = """# 가이드

EKS 클러스터 설치 절차를 설명합니다. kubectl과 helm이 필요합니다.

## 배포

helm install로 배포하고 helm rollback으로 되돌립니다.
"""


def test_config_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="알 수 없는 전략"):
        ChunkingConfig(strategy="nonsense")


def test_config_hash_is_deterministic_and_param_sensitive() -> None:
    a = ChunkingConfig("token", chunk_tokens=400, chunk_overlap=50)
    b = ChunkingConfig("token", chunk_tokens=400, chunk_overlap=50)
    c = ChunkingConfig("token", chunk_tokens=200, chunk_overlap=50)
    assert a.hash == b.hash  # 같은 파라미터 → 같은 해시
    assert a.hash != c.hash  # 크기 다르면 해시 다름
    assert len(a.hash) == 8


def test_collection_name_is_temp_and_never_production() -> None:
    name = ChunkingConfig("recursive").collection_name()
    assert name.startswith("onramp-eval-recursive-")
    assert name != "onramp"  # production 컬렉션과 절대 겹치지 않음


def test_page_from_row_maps_fields_and_handles_none() -> None:
    page = page_from_row(page_id=107194, title=None, markdown=None, source_url=None)
    assert page.page_id == "107194"  # str 변환
    assert page.page_title == ""  # None → 빈 문자열
    assert page.markdown == ""
    assert page.space_key == "OnRamp"  # 기본값


@pytest.mark.parametrize("strategy", ["onramp", "token", "markdown", "recursive"])
def test_chunk_page_dispatches_all_strategies(strategy) -> None:
    page = page_from_row(page_id="107194", title="가이드", markdown=_MD)
    chunks = chunk_page(ChunkingConfig(strategy, chunk_tokens=64, chunk_overlap=8), page)
    assert chunks, "전략별로 최소 1개 청크"
    assert all(c.chunk_id.startswith("107194_") for c in chunks)  # page-level 호환 포맷
    # onramp는 parent 구조 보유, baseline은 flat(parent_id 빈값)
    if strategy == "onramp":
        assert all(c.parent_id for c in chunks)
    else:
        assert all(c.parent_id == "" for c in chunks)
