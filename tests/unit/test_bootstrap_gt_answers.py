"""GT 답변 부트스트랩 순수 헬퍼 단위 테스트 (Qdrant/LLM 불필요)."""

import importlib.util
from pathlib import Path

# scripts/는 패키지가 아니므로 파일 경로로 로드
_SPEC = importlib.util.spec_from_file_location(
    "bootstrap_gt_answers", Path(__file__).resolve().parents[2] / "scripts" / "bootstrap_gt_answers.py"
)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
build_content_map = _MOD.build_content_map
collect_contexts = _MOD.collect_contexts


class _Point:
    def __init__(self, payload):
        self.payload = payload


def test_build_content_map_filters_incomplete() -> None:
    points = [
        _Point({"chunk_id": "a_001", "content": "내용A"}),
        _Point({"chunk_id": "b_001", "content": ""}),  # content 없음 → 제외
        _Point({"chunk_id": "", "content": "내용C"}),  # chunk_id 없음 → 제외
        _Point(None),  # payload 없음 → 제외
    ]
    m = build_content_map(points)
    assert m == {"a_001": "내용A"}


def test_collect_contexts_skips_missing() -> None:
    content_map = {"a_001": "A", "b_002": "B"}
    assert collect_contexts(["a_001", "b_002"], content_map) == ["A", "B"]
    assert collect_contexts(["a_001", "zzz"], content_map) == ["A"]  # 없는 id 건너뜀
    assert collect_contexts([], content_map) == []
