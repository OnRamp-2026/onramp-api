"""골든셋 로더 단위 테스트 (네트워크/LLM 불필요)."""

import json
from pathlib import Path

import pytest

from app.eval.dataset import load_golden_set


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def _paths(tmp_path: Path, queries: list[dict], qrels: list[dict]) -> tuple[Path, Path]:
    q = tmp_path / "queries.jsonl"
    r = tmp_path / "qrels.jsonl"
    _write(q, queries)
    _write(r, qrels)
    return q, r


def test_load_join_ok(tmp_path: Path) -> None:
    q, r = _paths(
        tmp_path,
        [
            {"qid": "q1", "query": "질문1", "domain": "manual", "is_answerable": True},
            {"qid": "q2", "query": "질문2", "domain": None, "is_answerable": False},
        ],
        [
            {"qid": "q1", "relevant_chunk_ids": ["p1_000", "p1_001"]},
            {"qid": "q2", "relevant_chunk_ids": []},
        ],
    )
    golden = load_golden_set(q, r)

    assert [g.qid for g in golden] == ["q1", "q2"]
    assert golden[0].relevant_chunk_ids == ("p1_000", "p1_001")
    assert golden[0].domain == "manual"
    assert golden[1].domain is None
    assert golden[1].is_answerable is False
    assert golden[1].relevant_chunk_ids == ()


def test_duplicate_qid_raises(tmp_path: Path) -> None:
    q, r = _paths(
        tmp_path,
        [{"qid": "q1", "query": "a"}, {"qid": "q1", "query": "b"}],
        [{"qid": "q1", "relevant_chunk_ids": []}],
    )
    with pytest.raises(ValueError, match="중복 qid"):
        load_golden_set(q, r)


def test_missing_qrels_for_query_raises(tmp_path: Path) -> None:
    q, r = _paths(
        tmp_path,
        [{"qid": "q1", "query": "a"}, {"qid": "q2", "query": "b"}],
        [{"qid": "q1", "relevant_chunk_ids": []}],
    )
    with pytest.raises(ValueError, match="qrels 누락"):
        load_golden_set(q, r)


def test_dangling_qrels_raises(tmp_path: Path) -> None:
    q, r = _paths(
        tmp_path,
        [{"qid": "q1", "query": "a"}],
        [{"qid": "q1", "relevant_chunk_ids": []}, {"qid": "ghost", "relevant_chunk_ids": ["x_000"]}],
    )
    with pytest.raises(ValueError, match="queries 에 없는 qid"):
        load_golden_set(q, r)


def test_empty_query_raises(tmp_path: Path) -> None:
    q, r = _paths(
        tmp_path,
        [{"qid": "q1", "query": "  "}],
        [{"qid": "q1", "relevant_chunk_ids": []}],
    )
    with pytest.raises(ValueError, match="query 누락"):
        load_golden_set(q, r)


def test_draft_flag_loaded(tmp_path: Path) -> None:
    q, r = _paths(
        tmp_path,
        [{"qid": "q1", "query": "a", "_draft": True}],
        [{"qid": "q1", "relevant_chunk_ids": ["p_000"]}],
    )
    golden = load_golden_set(q, r)
    assert golden[0].is_draft is True


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_golden_set(tmp_path / "nope.jsonl", tmp_path / "nope2.jsonl")


def test_non_dict_line_raises(tmp_path: Path) -> None:
    q = tmp_path / "queries.jsonl"
    r = tmp_path / "qrels.jsonl"
    q.write_text("[1, 2, 3]\n", encoding="utf-8")  # 배열 = dict 아님
    r.write_text('{"qid":"q1","relevant_chunk_ids":[]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="딕셔너리"):
        load_golden_set(q, r)


def test_non_bool_is_answerable_raises(tmp_path: Path) -> None:
    q, r = _paths(
        tmp_path,
        [{"qid": "q1", "query": "a", "is_answerable": "false"}],
        [{"qid": "q1", "relevant_chunk_ids": []}],
    )
    with pytest.raises(ValueError, match="bool"):
        load_golden_set(q, r)


# ── gold_domains (멀티 도메인) ────────────────────────────────────────────


def test_gold_domains_explicit_multi(tmp_path: Path) -> None:
    q, r = _paths(
        tmp_path,
        [
            {
                "qid": "m1",
                "query": "장애 대응",
                "domain": "incident",
                "gold_domains": ["incident", "api_reference"],
                "is_answerable": True,
            }
        ],
        [{"qid": "m1", "relevant_chunk_ids": ["p1_000", "p2_003"]}],
    )
    g = load_golden_set(q, r)[0]
    assert g.gold_domains == ("incident", "api_reference")
    assert g.is_multi_domain is True


def test_gold_domains_default_single(tmp_path: Path) -> None:
    """gold_domains 생략 + answerable → (domain,) 기본."""
    q, r = _paths(
        tmp_path,
        [{"qid": "q1", "query": "a", "domain": "manual", "is_answerable": True}],
        [{"qid": "q1", "relevant_chunk_ids": ["p_000"]}],
    )
    g = load_golden_set(q, r)[0]
    assert g.gold_domains == ("manual",)
    assert g.is_multi_domain is False


def test_gold_domains_default_unanswerable_empty(tmp_path: Path) -> None:
    q, r = _paths(
        tmp_path,
        [{"qid": "q1", "query": "a", "domain": None, "is_answerable": False}],
        [{"qid": "q1", "relevant_chunk_ids": []}],
    )
    g = load_golden_set(q, r)[0]
    assert g.gold_domains == ()
    assert g.is_multi_domain is False


def test_gold_domains_unknown_value_raises(tmp_path: Path) -> None:
    q, r = _paths(
        tmp_path,
        [
            {
                "qid": "q1",
                "query": "a",
                "domain": "incident",
                "gold_domains": ["incident", "nonsense"],
                "is_answerable": True,
            }
        ],
        [{"qid": "q1", "relevant_chunk_ids": ["p_000"]}],
    )
    with pytest.raises(ValueError, match="알 수 없는 도메인"):
        load_golden_set(q, r)


def test_gold_domains_must_contain_domain(tmp_path: Path) -> None:
    """라우터 단일 픽(domain)이 gold_domains에 없으면 라벨 불일치 에러."""
    q, r = _paths(
        tmp_path,
        [
            {
                "qid": "q1",
                "query": "a",
                "domain": "manual",
                "gold_domains": ["incident", "api_reference"],
                "is_answerable": True,
            }
        ],
        [{"qid": "q1", "relevant_chunk_ids": ["p_000"]}],
    )
    with pytest.raises(ValueError, match="라벨 불일치"):
        load_golden_set(q, r)


def test_domain_unknown_value_raises(tmp_path: Path) -> None:
    """gold_domains 생략 시에도 domain(라우터 단일 픽) 오타를 잡아야 한다."""
    q, r = _paths(
        tmp_path,
        [{"qid": "q1", "query": "a", "domain": "nonsense", "is_answerable": True}],
        [{"qid": "q1", "relevant_chunk_ids": ["p_000"]}],
    )
    with pytest.raises(ValueError, match="알 수 없는 도메인"):
        load_golden_set(q, r)


def test_domain_non_string_raises(tmp_path: Path) -> None:
    q, r = _paths(
        tmp_path,
        [{"qid": "q1", "query": "a", "domain": 123, "is_answerable": True}],
        [{"qid": "q1", "relevant_chunk_ids": ["p_000"]}],
    )
    with pytest.raises(ValueError, match="알 수 없는 도메인"):
        load_golden_set(q, r)


def test_gold_domains_not_list_raises(tmp_path: Path) -> None:
    q, r = _paths(
        tmp_path,
        [{"qid": "q1", "query": "a", "domain": "manual", "gold_domains": "manual", "is_answerable": True}],
        [{"qid": "q1", "relevant_chunk_ids": ["p_000"]}],
    )
    with pytest.raises(ValueError, match="리스트"):
        load_golden_set(q, r)
