"""onramp 사이즈 sweep 헬퍼 단위 테스트 (#212) — 순수 함수, 인프라 불필요."""

from app.eval.size_sweep import (
    DEFAULT_CHILD_TARGET,
    DEFAULT_PARENT_TARGET,
    SizeConfig,
    build_onramp_chunker,
    chunk_page_sized,
    page_from_row,
)

_MD = """# 가이드

EKS 클러스터 설치 절차를 설명합니다. kubectl과 helm이 필요합니다.

## 배포

helm install로 배포하고 helm rollback으로 되돌립니다.
"""


def test_build_chunker_derives_params_with_fixed_overlap() -> None:
    c = build_onramp_chunker(child_target=256, parent_target=2048)
    assert c.child_target_tokens == 256
    assert c.parent_target_tokens == 2048
    assert c.overlap_tokens == round(256 * 0.15)  # 10~20% 고정(=15%)
    assert c.child_max_tokens > c.child_target_tokens  # overlap 헤드룸 확보
    assert c.child_min_tokens >= 1


def test_default_targets() -> None:
    c = build_onramp_chunker(DEFAULT_CHILD_TARGET, DEFAULT_PARENT_TARGET)
    assert (c.child_target_tokens, c.parent_target_tokens) == (400, 1200)


def test_chunk_page_sized_produces_page_compatible_children() -> None:
    page = page_from_row(page_id="107194", title="가이드", markdown=_MD)
    chunks = chunk_page_sized(page, child_target=64, parent_target=256)
    assert chunks, "최소 1개 청크"
    assert all(c.chunk_id.startswith("107194_") for c in chunks)  # page-level 지표 호환 포맷
    assert all(c.parent_id for c in chunks)  # onramp는 parent 구조 보유


def test_invalid_targets_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="1 이상"):
        build_onramp_chunker(0, 1200)


def test_sizeconfig_hash_and_collection_distinct() -> None:
    a = SizeConfig(256, 1200)
    b = SizeConfig(256, 1200)
    c = SizeConfig(512, 1200)
    assert a.hash == b.hash  # 결정성
    assert a.hash != c.hash  # 사이즈 다르면 다름
    assert a.collection_name() != c.collection_name()
    assert a.collection_name().startswith("onramp-eval-size-c256-p1200-")
    assert a.collection_name() != "onramp"  # production 미충돌


def test_page_from_row_handles_none() -> None:
    page = page_from_row(page_id=107194, title=None, markdown=None, source_url=None)
    assert page.page_id == "107194"
    assert page.page_title == ""
    assert page.space_key == "OnRamp"
