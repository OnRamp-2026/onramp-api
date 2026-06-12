"""qrels 마이그레이션 검증 — 새 코퍼스에서 골든셋 정답 청크의 생존 여부를 분류한다.

코퍼스를 재적재하면 chunk_id(`{page_id}_{idx}`)가 바뀌어 옛 qrels가 깨질 수 있다. 이 도구는
**재검수 범위를 좁히는 1차 게이트**다 — 각 질문의 정답 청크가 현재 컬렉션에 존재하는지 분류하고,
누락분에는 같은 page_id의 현재 청크를 후보로 제시한다.

⚠️ 한계: 존재(intact)해도 페이지 내용이 바뀌었으면 정답이 아닐 수 있다 — **intact도 사람이 내용 검수**해야 한다.
이 도구는 qrels.jsonl을 **자동 수정하지 않는다**. 결과는 검수용 draft(gitignore, 원문 preview 포함)로만 저장한다.

전제: Qdrant 기동 + 색인 완료. 사용:
    python scripts/validate_qrels.py                 # 콘솔 요약 + draft 저장
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from app.config import get_settings
from app.eval.dataset import load_golden_set

_ROOT = Path(__file__).resolve().parents[1]
_QUERIES = _ROOT / "data/eval/queries.jsonl"
_QRELS = _ROOT / "data/eval/qrels.jsonl"
_DEFAULT_OUT = str(_ROOT / "data/eval/reviews/qrels_migration.draft.json")  # gitignore (preview 포함)
_SCROLL_BATCH = 256
_PREVIEW_CHARS = 160

_STATUS = ("intact", "partial", "missing", "empty")


def _page_id(chunk_id: str) -> str:
    """chunk_id `{page_id}_{idx}` → page_id (마지막 _ 앞)."""
    return chunk_id.rsplit("_", 1)[0] if "_" in chunk_id else chunk_id


def _classify(relevant: tuple[str, ...], existing: set[str]) -> str:
    """qrels 상태: empty(unanswerable) / intact(전부 존재) / missing(전부 없음) / partial(일부)."""
    if not relevant:
        return "empty"
    present = sum(1 for c in relevant if c in existing)
    if present == len(relevant):
        return "intact"
    if present == 0:
        return "missing"
    return "partial"


def _candidates(missing: list[str], page_to_chunks: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """누락 chunk_id마다 **같은 page_id**의 현재 청크 후보(chunk_id+preview)를 제공."""
    return {cid: page_to_chunks.get(_page_id(cid), []) for cid in missing}


def _scroll_corpus(client, collection: str, *, preview_chars: int = _PREVIEW_CHARS) -> tuple[set[str], dict]:
    """Qdrant 전체 payload를 **pagination(scroll)** 으로 수집 → (chunk_id 집합, page_id→[{chunk_id,preview}])."""
    existing: set[str] = set()
    page_to_chunks: dict[str, list[dict]] = defaultdict(list)
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=_SCROLL_BATCH,
            offset=offset,
            with_payload=["chunk_id", "content", "page_id"],
            with_vectors=False,
        )
        for p in points:
            payload = p.payload or {}
            cid = payload.get("chunk_id")
            if not cid:
                continue
            existing.add(cid)
            pid = payload.get("page_id") or _page_id(cid)
            page_to_chunks[pid].append({"chunk_id": cid, "preview": (payload.get("content") or "")[:preview_chars]})
        if offset is None:
            break
    return existing, dict(page_to_chunks)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return ""


def _build_report(golden, existing: set[str], page_to_chunks: dict, *, collection: str, settings) -> dict:
    rows: list[dict] = []
    for g in golden:
        relevant = tuple(g.relevant_chunk_ids)
        status = _classify(relevant, existing)
        present = [c for c in relevant if c in existing]
        missing = [c for c in relevant if c not in existing]
        rows.append(
            {
                "qid": g.qid,
                "is_answerable": g.is_answerable,
                "status": status,
                "relevant_total": len(relevant),
                "relevant_existing": present,
                "relevant_missing": missing,
                "candidates_same_page": _candidates(missing, page_to_chunks),
            }
        )

    counts = {s: sum(1 for r in rows if r["status"] == s) for s in _STATUS}
    answerable_rows = [r for r in rows if r["is_answerable"] and r["relevant_total"]]
    rel_total = sum(r["relevant_total"] for r in answerable_rows)
    rel_existing = sum(len(r["relevant_existing"]) for r in answerable_rows)
    needs_review = sorted(r["qid"] for r in rows if r["status"] in ("partial", "missing"))
    with open(_QUERIES, "rb") as f:
        import hashlib

        golden_sha = hashlib.sha256(f.read()).hexdigest()[:12]
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "corpus": {"collection": collection, "points_count": len(existing)},
        "reproduction": {
            "golden_sha": golden_sha,
            "code_commit_sha": _git_sha(),
            "embedding_model": settings.embedding_model,
        },
        "status_counts": counts,
        "chunk_survival_rate": round(rel_existing / rel_total, 4) if rel_total else None,
        "chunk_survival": f"{rel_existing}/{rel_total}",
        "needs_review_qids": needs_review,
        "note": "intact도 내용 검수 필요(존재≠정답 유효). qrels.jsonl 자동 수정 안 함 — 사람이 draft를 보고 확정.",
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="qrels 마이그레이션 검증 (새 코퍼스 생존율)")
    parser.add_argument("--out", default=_DEFAULT_OUT, help="검수용 draft 출력(gitignore, preview 포함)")
    args = parser.parse_args()

    from app.db.qdrant import get_qdrant

    settings = get_settings()
    golden = load_golden_set(_QUERIES, _QRELS)
    existing, page_to_chunks = _scroll_corpus(get_qdrant(), settings.qdrant_collection)
    report = _build_report(golden, existing, page_to_chunks, collection=settings.qdrant_collection, settings=settings)

    # 콘솔 요약(preview 없음)
    print(f"\n=== qrels 마이그레이션 검증 (collection={report['corpus']['collection']}, chunks={len(existing)}) ===")
    print(f"상태: {report['status_counts']}  (전체 {sum(report['status_counts'].values())}문항)")
    print(f"정답 청크 생존율: {report['chunk_survival']} = {report['chunk_survival_rate']}")
    print(f"재검수 대상(partial/missing) {len(report['needs_review_qids'])}건: {report['needs_review_qids'][:20]}")
    print("→ intact도 내용 검수 필요(존재≠정답). qrels.jsonl은 자동 수정 안 함.")

    # draft 저장(원문 preview 포함 → gitignore)
    d = os.path.dirname(args.out)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    os.replace(tmp, args.out)
    print(f"\n✅ 검수 draft 저장(gitignore): {args.out}")


if __name__ == "__main__":
    main()
