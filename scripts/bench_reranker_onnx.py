"""#60 — 리랭커 백엔드 벤치: torch vs ONNX(int8). 쿼리당 속도 + 골든셋 품질(hit@5/recall@5/mrr@10).

동일 dense top-k 후보 풀에 두 백엔드를 적용해 공정 비교한다. 절대 latency는 머신마다 다르므로
"속도 배수"와 "품질 동등 여부"로 판단할 것(운영 파드에서 재측정 권장).

전제: 로컬 Qdrant onramp 색인 + data/eval/{queries,qrels}.jsonl + OpenAI 키.
의존성: uv pip install -e ".[rerank,onnx]"
실행:   PYTHONPATH=. python scripts/bench_reranker_onnx.py --onnx-dir models/bge-reranker-onnx-int8 [--n 20]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import urllib.request

from app.config import get_settings
from app.eval.metrics import hit_rate_at_k, recall_at_k, reciprocal_rank
from app.rag.embedder import get_embedder

QDRANT = "http://localhost:6333/collections/onramp"


def _post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        QDRANT + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    return json.load(urllib.request.urlopen(req, timeout=15))["result"]


def _dense(vec: list[float], k: int) -> list[tuple[str, str]]:
    pts = _post("/points/query", {"query": vec, "limit": k, "with_payload": ["chunk_id", "content"]})["points"]
    return [(p["payload"]["chunk_id"], p["payload"].get("content", "")) for p in pts]


def _present() -> set[str]:
    ids: set[str] = set()
    offset = None
    while True:
        body = {"limit": 256, "with_payload": ["chunk_id"]}
        if offset:
            body["offset"] = offset
        r = _post("/points/scroll", body)
        ids.update(p["payload"]["chunk_id"] for p in r["points"] if p["payload"].get("chunk_id"))
        offset = r.get("next_page_offset")
        if not offset:
            return ids


async def _evalset(n: int, top_k: int) -> list[dict]:
    with open("data/eval/queries.jsonl") as f:
        queries = {json.loads(line)["qid"]: json.loads(line) for line in f}
    with open("data/eval/qrels.jsonl") as f:
        qrels = {json.loads(line)["qid"]: json.loads(line)["relevant_chunk_ids"] for line in f}
    present = _present()
    items = [
        (q["query"], qrels[qid])
        for qid, q in queries.items()
        if q.get("is_answerable") and qrels.get(qid) and set(qrels[qid]) <= present
    ][:n]
    vecs = await get_embedder().embed_documents([q for q, _ in items])
    return [{"query": q, "relevant": rids, "cands": _dense(v, top_k)} for (q, rids), v in zip(items, vecs, strict=True)]


def _bench(name, predict, evalset):
    predict(evalset[0]["query"], [c for _, c in evalset[0]["cands"]])  # warmup (모델 로드 제외)
    lat, mts = [], []
    for e in evalset:
        ids = [c for c, _ in e["cands"]]
        t = time.perf_counter()
        scores = predict(e["query"], [c for _, c in e["cands"]])
        lat.append(time.perf_counter() - t)
        ranked = [c for c, _ in sorted(zip(ids, scores, strict=True), key=lambda x: -x[1])]
        rel = set(e["relevant"])
        mts.append((hit_rate_at_k(ranked, rel, 5), recall_at_k(ranked, rel, 5), reciprocal_rank(ranked, rel, 10)))

    def avg(i: int) -> float:
        return round(statistics.mean(m[i] for m in mts), 3)

    res = {
        "lat_med_s": round(statistics.median(lat), 2),
        "lat_mean_s": round(statistics.mean(lat), 2),
        "hit@5": avg(0),
        "recall@5": avg(1),
        "mrr@10": avg(2),
    }
    print(
        f"[{name:<10}] lat med {res['lat_med_s']}s mean {res['lat_mean_s']}s | "
        f"hit@5 {res['hit@5']} recall@5 {res['recall@5']} mrr@10 {res['mrr@10']}",
        flush=True,
    )
    return res


def main() -> None:
    p = argparse.ArgumentParser(description="리랭커 torch vs ONNX(int8) 벤치 (#60)")
    p.add_argument("--onnx-dir", required=True, help="build_reranker_onnx.py 산출 디렉토리")
    p.add_argument("--onnx-file", default="model_quantized.onnx")
    p.add_argument("--n", type=int, default=20, help="평가 질문 수")
    args = p.parse_args()

    settings = get_settings()
    evalset = asyncio.run(_evalset(args.n, settings.retriever_top_k))
    print(f"평가셋 {len(evalset)}문항 / 후보 {settings.retriever_top_k}", flush=True)

    import numpy as np
    import onnxruntime as ort
    from sentence_transformers import CrossEncoder
    from transformers import AutoTokenizer

    out = {}
    ce = CrossEncoder(settings.reranker_model, device="cpu")
    out["torch"] = _bench("torch", lambda q, ps: ce.predict([(q, p) for p in ps]), evalset)

    # 실제 OnnxCrossEncoderReranker와 동일 경로(순수 onnxruntime + numpy, torch 미사용)로 벤치
    tok = AutoTokenizer.from_pretrained(settings.reranker_model)
    sess = ort.InferenceSession(f"{args.onnx_dir}/{args.onnx_file}", providers=["CPUExecutionProvider"])
    input_names = {i.name for i in sess.get_inputs()}

    def pred8(query: str, passages: list[str]):
        feats = tok(
            [query] * len(passages), passages, padding=True, truncation=True, max_length=512, return_tensors="np"
        )
        logits = sess.run(None, {k: v for k, v in feats.items() if k in input_names})[0]
        return (1.0 / (1.0 + np.exp(-logits))).reshape(-1)

    out["onnx_int8"] = _bench("onnx_int8", pred8, evalset)

    t, q = out["torch"], out["onnx_int8"]
    print(
        f"\n속도: {t['lat_mean_s']}s → {q['lat_mean_s']}s = {round(t['lat_mean_s'] / q['lat_mean_s'], 1)}x", flush=True
    )
    print(
        f"품질: hit@5 {t['hit@5']}→{q['hit@5']} / recall@5 {t['recall@5']}→{q['recall@5']} / mrr@10 {t['mrr@10']}→{q['mrr@10']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
