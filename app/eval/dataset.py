"""검색 평가 골든셋 로더.

`queries.jsonl`(질문)과 `qrels.jsonl`(정답 chunk_id 라벨)을 `qid`로 조인한다.
네트워크/LLM 의존이 없어 단위 테스트로 검증 가능하다.

포맷:
    queries.jsonl  1줄=1질문:
        {"qid":"q001","query":"...","domain":"incident","is_answerable":true,
         "ground_truth_answer":"...(선택, #C 전용)","_draft":false}
    qrels.jsonl    1줄=1질문 라벨(chunk_id 단위):
        {"qid":"q001","relevant_chunk_ids":["<page_id>_003", ...]}   # unanswerable이면 []
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_QUERIES_PATH = Path("data/eval/queries.jsonl")
DEFAULT_QRELS_PATH = Path("data/eval/qrels.jsonl")


@dataclass(frozen=True)
class GoldenQuery:
    """평가용 골든 질문 한 건 (queries + qrels 조인 결과)."""

    qid: str
    query: str
    domain: str | None
    is_answerable: bool
    relevant_chunk_ids: tuple[str, ...]
    ground_truth_answer: str | None = None
    is_draft: bool = False  # 부트스트랩 초안(_draft) — 팀 검수 전


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"골든셋 파일이 없습니다: {path}")
    rows: list[dict] = []
    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{i} JSON 파싱 실패: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"{path}:{i} JSON 객체(딕셔너리)여야 합니다")
        rows.append(obj)
    return rows


def _load_qrels(qrels_path: Path) -> dict[str, tuple[str, ...]]:
    qrels: dict[str, tuple[str, ...]] = {}
    for row in _read_jsonl(qrels_path):
        qid = row.get("qid")
        if not qid:
            raise ValueError(f"{qrels_path}: qid 누락된 행 {row}")
        if qid in qrels:
            raise ValueError(f"{qrels_path}: 중복 qid '{qid}'")
        ids = row.get("relevant_chunk_ids", [])
        if not isinstance(ids, list):
            raise ValueError(f"{qrels_path}: '{qid}' relevant_chunk_ids 는 리스트여야 합니다")
        qrels[qid] = tuple(str(c) for c in ids)
    return qrels


def load_golden_set(
    queries_path: Path | str = DEFAULT_QUERIES_PATH,
    qrels_path: Path | str = DEFAULT_QRELS_PATH,
) -> list[GoldenQuery]:
    """골든셋을 로드해 `GoldenQuery` 리스트로 반환한다.

    중복 qid / qid 누락 / queries↔qrels 불일치(dangling) 시 ValueError.
    `_draft` 행이 섞여 있으면 경고만 하고 그대로 로드한다(팀 검수 신호).
    """
    queries_path = Path(queries_path)
    qrels_path = Path(qrels_path)

    qrels = _load_qrels(qrels_path)

    seen: set[str] = set()
    golden: list[GoldenQuery] = []
    draft_n = 0
    for row in _read_jsonl(queries_path):
        qid = row.get("qid")
        if not qid:
            raise ValueError(f"{queries_path}: qid 누락된 행 {row}")
        if qid in seen:
            raise ValueError(f"{queries_path}: 중복 qid '{qid}'")
        seen.add(qid)
        if not str(row.get("query", "")).strip():
            raise ValueError(f"{queries_path}: '{qid}' query 누락")
        if qid not in qrels:
            raise ValueError(f"qrels 누락: '{qid}' (queries 에 있으나 qrels 없음)")
        is_answerable = row.get("is_answerable", True)
        if not isinstance(is_answerable, bool):
            raise ValueError(f"{queries_path}: '{qid}' is_answerable 는 bool 이어야 합니다")
        is_draft = bool(row.get("_draft", False))
        draft_n += int(is_draft)
        golden.append(
            GoldenQuery(
                qid=qid,
                query=str(row["query"]),
                domain=row.get("domain"),
                is_answerable=is_answerable,
                relevant_chunk_ids=qrels[qid],
                ground_truth_answer=row.get("ground_truth_answer"),
                is_draft=is_draft,
            )
        )

    dangling = set(qrels) - seen
    if dangling:
        raise ValueError(f"{qrels_path}: queries 에 없는 qid {sorted(dangling)}")
    if draft_n:
        logger.warning("골든셋에 _draft 행 %d개 — 팀 검수 후 _draft 제거 필요", draft_n)
    return golden
