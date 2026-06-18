"""청킹 A/B 실험 헬퍼 단위 테스트 (#212) — 순수 함수, 인프라 불필요."""

import pytest

from app.eval.chunking_experiment import (
    DEFAULT_CHILD_TARGET,
    DEFAULT_PARENT_TARGET,
    ChunkingConfig,
    chunk_page,
    onramp_chunker,
    page_from_row,
)

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


def test_onramp_default_chunker_unchanged() -> None:
    """size 미지정 onramp는 기존 동작 그대로 — 팀원 splitter 비교 경로 보존."""
    c = onramp_chunker(ChunkingConfig("onramp"))
    assert c.child_target_tokens == DEFAULT_CHILD_TARGET
    assert c.parent_target_tokens == DEFAULT_PARENT_TARGET
    assert c.overlap_tokens == 120  # SemanticChunker 기본 (파생 안 함)


def test_onramp_size_params_derive_chunker() -> None:
    """child/parent target 지정 시 target 반영 + overlap 15% 파생(토큰 수만 sweep)."""
    c = onramp_chunker(ChunkingConfig("onramp", child_target=256, parent_target=2048))
    assert c.child_target_tokens == 256
    assert c.parent_target_tokens == 2048
    assert c.overlap_tokens == round(256 * 0.15)  # 38 — 10~20% 고정
    assert c.child_max_tokens > c.child_target_tokens  # overlap 헤드룸 확보


def test_size_params_change_hash_and_collection() -> None:
    """사이즈 변형은 distinct 컬렉션명 — 같은 onramp라도 충돌 안 함(#212)."""
    base = ChunkingConfig("onramp")
    c256 = ChunkingConfig("onramp", child_target=256, parent_target=1200)
    c512 = ChunkingConfig("onramp", child_target=512, parent_target=1200)
    assert len({base.hash, c256.hash, c512.hash}) == 3
    assert c256.collection_name() != c512.collection_name()


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
