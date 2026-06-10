"""골든셋 부트스트랩 순수 헬퍼 단위 테스트 (Qdrant/LLM 불필요)."""

import importlib.util
from pathlib import Path

# scripts/는 패키지가 아니므로 파일 경로로 로드
_SPEC = importlib.util.spec_from_file_location(
    "bootstrap_golden", Path(__file__).resolve().parents[2] / "scripts" / "bootstrap_golden.py"
)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
group_adjacent_chunks = _MOD.group_adjacent_chunks
sample_per_domain = _MOD.sample_per_domain


def _chunk(page_id: str, index: int, domain: str = "manual") -> dict:
    return {
        "page_id": page_id,
        "chunk_id": f"{page_id}_{index:03d}",
        "chunk_index": index,
        "content": f"내용 {page_id}/{index}",
        "domain": domain,
    }


def test_group_adjacent_chunks_pairs_consecutive_indexes():
    payloads = [_chunk("p1", 0), _chunk("p1", 1), _chunk("p1", 2), _chunk("p2", 0)]
    groups = group_adjacent_chunks(payloads, span=2, max_groups_per_page=2)
    # p1: (0,1) 묶음 1개 — (2)는 span 미달, p2: 청크 1개라 그룹 없음
    assert [[c["chunk_id"] for c in g] for g in groups] == [["p1_000", "p1_001"]]


def test_group_adjacent_chunks_skips_non_consecutive():
    payloads = [_chunk("p1", 0), _chunk("p1", 2)]  # 1 누락 → 비연속
    assert group_adjacent_chunks(payloads, span=2) == []


def test_group_adjacent_chunks_respects_max_groups_per_page():
    payloads = [_chunk("p1", i) for i in range(6)]
    groups = group_adjacent_chunks(payloads, span=2, max_groups_per_page=2)
    assert len(groups) == 2


def test_group_adjacent_chunks_span_three():
    payloads = [_chunk("p1", i) for i in range(3)]
    groups = group_adjacent_chunks(payloads, span=3)
    assert [[c["chunk_id"] for c in g] for g in groups] == [["p1_000", "p1_001", "p1_002"]]


def test_sample_per_domain_caps_per_domain():
    payloads = [_chunk("p1", i, domain="manual") for i in range(5)] + [
        _chunk("p2", i, domain="incident") for i in range(2)
    ]
    sampled = sample_per_domain(payloads, per_domain=3)
    domains = sorted(c["domain"] for c in sampled)
    assert domains == ["incident", "incident", "manual", "manual", "manual"]
