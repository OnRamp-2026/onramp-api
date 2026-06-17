"""청크 도메인 분류 품질을 LLM으로 평가.

룰 기반(app/rag/classifier.py)이 부여한 domain이 타당한지, 같은 분류 체계를 준 LLM이
독립적으로 재분류한 결과와 비교한다. 마스킹된 청크만 보내므로 임베딩 파이프라인과 동일한
데이터 노출 수준이다.

예) python scripts/eval_domain_classification.py --per-domain 30
    python scripts/eval_domain_classification.py --per-domain 20 --space TrustRAG --model gpt-4o
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import httpx

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings  # noqa: E402
from app.services.llm_selector import call_llm  # noqa: E402

logger = logging.getLogger(__name__)

# 룰과 동일한 5분류 체계 — LLM에 정의를 그대로 준다 (app/rag/classifier.py:DOMAIN_RULES).
DOMAIN_DEFS = {
    "incident": "장애/인시던트 대응. 장애 타임라인·영향·원인(root cause)·복구·재발방지(postmortem).",
    "api_reference": "API 명세/레퍼런스. 엔드포인트·요청/응답·HTTP 상태코드·에러코드.",
    "meeting_note": "회의록. 참석자·논의·결정사항·액션아이템·스프린트 미팅 기록.",
    "planning": "기획/설계. 요구사항·PRD/RFC·정책·목표·범위·아키텍처 설계.",
    "manual": "운영 매뉴얼/런북. 설치·절차·검증·롤백·트러블슈팅·운영 명령(kubectl/helm).",
}
DOMAINS = list(DOMAIN_DEFS)

SYSTEM_PROMPT = (
    "너는 사내 문서 청크를 아래 5개 도메인 중 하나로 분류하는 분류기다. "
    "청크의 제목·섹션 경로·본문을 보고 가장 적합한 도메인 하나를 고른다.\n\n"
    + "\n".join(f"- {k}: {v}" for k, v in DOMAIN_DEFS.items())
    + '\n\n반드시 JSON만 출력: {"domain": "<위 5개 중 하나>", "confidence": <0~1 float>, "reason": "<한 문장>"}'
)


async def _sample(client: httpx.AsyncClient, base: str, domain: str, per_domain: int, space: str | None) -> list[dict]:
    must: list[dict] = [{"term": {"domain": domain}}]
    if space:
        must.append({"term": {"space_key": space}})
    body = {
        "size": per_domain,
        "_source": ["chunk_id", "domain", "page_title", "heading_path", "content"],
        "query": {
            "function_score": {"query": {"bool": {"must": must}}, "random_score": {"seed": 42, "field": "_seq_no"}}
        },
    }
    r = await client.post(f"{base}/onramp-chunks/_search", json=body)
    r.raise_for_status()
    return [h["_source"] for h in r.json()["hits"]["hits"]]


async def _classify(chunk: dict) -> str | None:
    heading = " > ".join(chunk.get("heading_path") or [])
    user = f"제목: {chunk.get('page_title', '')}\n섹션 경로: {heading}\n본문:\n{(chunk.get('content') or '')[:1500]}"
    try:
        raw = await call_llm(SYSTEM_PROMPT, user, temperature=0.0, max_tokens=200, json_mode=True)
        label = json.loads(raw).get("domain", "").strip()
        return label if label in DOMAINS else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("분류 실패 (%s): %s", chunk.get("chunk_id"), exc)
        return None


async def run(per_domain: int, space: str | None, concurrency: int) -> None:
    s = get_settings()
    base = f"{s.opensearch_scheme}://{s.opensearch_host}:{s.opensearch_port}"

    async with httpx.AsyncClient(timeout=15) as client:
        samples: list[dict] = []
        for d in DOMAINS:
            rows = await _sample(client, base, d, per_domain, space)
            samples.extend(rows)
    random.shuffle(samples)
    logger.info("샘플 %d개 (도메인별 최대 %d, space=%s) — LLM 분류 시작", len(samples), per_domain, space or "전체")

    sem = asyncio.Semaphore(concurrency)

    async def _one(chunk: dict) -> tuple[str, str | None]:
        async with sem:
            return chunk["domain"], await _classify(chunk)

    results = await asyncio.gather(*(_one(c) for c in samples))
    pairs = [(rule, llm, c) for (rule, llm), c in zip(results, samples, strict=True) if llm]

    # 집계
    n = len(pairs)
    if not n:
        logger.warning("유효 분류 결과 없음 — 종료")
        return
    agree = sum(1 for rule, llm, _ in pairs if rule == llm)
    per_domain_stat: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # rule domain → [agree, total]
    confusion: Counter = Counter()
    for rule, llm, _ in pairs:
        per_domain_stat[rule][1] += 1
        if rule == llm:
            per_domain_stat[rule][0] += 1
        else:
            confusion[(rule, llm)] += 1

    print("\n" + "=" * 64)
    print(f"LLM 도메인 분류 평가 — 유효 {n}건 (분류 실패 {len(samples) - n} 제외)")
    print("=" * 64)
    print(f"\n전체 일치율(rule==LLM): {agree}/{n} = {agree / n * 100:.1f}%\n")

    print("[룰 도메인별 일치율]")
    for d in DOMAINS:
        a, t = per_domain_stat[d]
        if t:
            print(f"    {d:14s} {a:3d}/{t:3d} = {a / t * 100:5.1f}%")

    print("\n[주요 불일치 (rule → LLM): 횟수]")
    for (rule, llm), cnt in confusion.most_common(12):
        print(f"    {rule:14s} → {llm:14s} : {cnt}")

    print("\n[불일치 청크 샘플 8건]")
    shown = 0
    for rule, llm, c in pairs:
        if rule != llm and shown < 8:
            title = (c.get("page_title") or "")[:50]
            print(f"    [{rule} → {llm}] {c.get('chunk_id')}  {title!r}")
            shown += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="청크 도메인 분류를 LLM으로 평가(룰 vs LLM 일치율).")
    parser.add_argument("--per-domain", type=int, default=25, help="도메인별 샘플 수 (총 5×N)")
    parser.add_argument("--space", default=None, help="space_key 필터 (예: TrustRAG=Confluence)")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    asyncio.run(run(args.per_domain, args.space, args.concurrency))


if __name__ == "__main__":
    main()
