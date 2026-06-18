"""지식맵 그래프 검증(읽기전용) — Qdrant 실데이터로 build_graph 통계만 출력.

목적: 하이브리드 지식맵의 "개념(concept) 그래프가 시각적으로 쓸 만한가"를 정량 판단.
  - cross-domain 비율 = 차별점("개념이 도메인을 가로질러 문서를 잇는다")의 실측치
  - hairball 지표 = 시각적 난잡 위험 (특정 개념이 너무 많은 문서에 붙음)
  - 고립 문서 비율 = 밋밋함 위험 (개념 연결이 없는 문서)

크리덴셜은 출력하지 않는다. RAG 파이프라인 무접촉(Qdrant 읽기전용 scroll).
파드 내 실행:  python scripts/validate_knowledge_graph.py
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import replace

from app.config import get_settings
from app.db.qdrant import get_qdrant
from app.rag.labels import strip_upload_suffix
from app.services.knowledge_graph import GraphDocument, build_graph
from app.services.llm_selector import call_llm

# ── concept(keyword) 정제 ─────────────────────────────────────────────────
# keyword 출처(chunker._extract_keywords): 백틱 코드 / 명령어 / *Error류 / heading 마지막 2개.
# → 마스킹 placeholder·CodeRabbit 봇·PR 템플릿 헤딩·불용어가 노이즈로 유입. 아래는 데이터 검증서 드러난 패턴.
_NOISE_SUBSTR = ("masked_", "coderabbit")  # 마스킹토큰([MASKED_*]) · CodeRabbit 봇
_NOISE_KW_SUBSTR = ("checklist", "failed checks", "passed checks", "warning)", "bugfix")  # CI/PR 템플릿
_NOISE_EXACT = {
    # 한글 불용어성 헤딩
    "영향 범위", "확인 방법", "개요", "요약", "목차", "비고", "기타", "참고", "주의", "배경", "목적", "결론", "제목",
    "변경 사항", "변경사항", "온램프 정제 테스트", "onramp 정제 테스트",
    # 영어 불용어
    "on", "off", "summary", "please note", "note", "see", "see also", "example", "overview",
    "yes", "no", "true", "false", "todo", "done", "wip", "n/a", "tbd", "changes", "none",
}


def is_noise_keyword(kw: str) -> bool:
    """노이즈 keyword면 True. 진짜 개념(mod_rewrite·GaugeHistogram 등)은 통과."""
    k = re.sub(r"\s+", " ", kw).strip(" .,;:()[]")
    if not k or len(k) <= 1 or len(k) > 40:  # 빈/단글자/문장형
        return True
    if not k[0].isalnum():  # 이모지·특수문자(@,❌,▶…) 시작
        return True
    kl = k.lower()
    if kl in _NOISE_EXACT:
        return True
    if any(s in kl for s in _NOISE_SUBSTR) or any(s in kl for s in _NOISE_KW_SUBSTR):
        return True
    return False


def clean_docs(docs: list[GraphDocument]) -> list[GraphDocument]:
    return [replace(d, keywords=tuple(k for k in d.keywords if not is_noise_keyword(k))) for d in docs]


def diagnose_keywords(docs: list[GraphDocument]) -> None:
    raw: Counter[str] = Counter()
    for d in docs:
        for k in d.keywords:
            raw[k] += 1
    removed = sorted(((c, k) for k, c in raw.items() if is_noise_keyword(k)), reverse=True)
    kept = sorted(((c, k) for k, c in raw.items() if not is_noise_keyword(k)), reverse=True)
    print(f"\n[정제] keyword 종류 {len(raw)} → 제거 {len(removed)} / 유지 {len(kept)}")
    print("  ❌제거 top25: " + ", ".join(f"{k}({c})" for c, k in removed[:25]))
    print("  ✅유지 top25: " + ", ".join(f"{k}({c})" for c, k in kept[:25]))


def collect_documents() -> list[GraphDocument]:
    """Qdrant 전체 scroll → page_id 단위로 묶어 GraphDocument 구성.

    keyword는 한 문서(page_id)에 속한 모든 청크 keyword의 합집합.
    source는 payload.source(멀티소스) → payload.site(confluence 사이트) 순 폴백.
    """
    client = get_qdrant()
    settings = get_settings()
    collection = settings.qdrant_collection

    titles: dict[str, str] = {}
    domains: dict[str, str] = {}
    sources: dict[str, str] = {}
    versions: dict[str, str] = {}
    last_mods: dict[str, str] = {}
    keywords: dict[str, set[str]] = defaultdict(set)
    chunk_total = 0

    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=512,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            pl = p.payload or {}
            pid = str(pl.get("page_id") or "").strip()
            if not pid:
                continue
            chunk_total += 1
            titles.setdefault(pid, str(pl.get("page_title") or pid))
            domains.setdefault(pid, str(pl.get("domain") or ""))
            # page_id가 gh:로 시작하면 GitHub (payload에 source 구분 필드 없음 → prefix로 판별).
            if pid.startswith("gh:"):
                sources[pid] = "github"
            else:
                sources.setdefault(pid, str(pl.get("source") or pl.get("site") or "confluence"))
            versions.setdefault(pid, str(pl.get("product_version") or pl.get("version") or ""))
            last_mods.setdefault(pid, str(pl.get("last_modified") or ""))
            for kw in pl.get("keywords") or []:
                kw = str(kw).strip()
                if kw:
                    keywords[pid].add(kw)
        if offset is None:
            break

    docs = [
        GraphDocument(
            page_id=pid,
            title=titles[pid],
            source=sources[pid],
            domain=domains[pid],
            version=versions[pid],
            last_modified=last_mods[pid],
            keywords=tuple(sorted(keywords.get(pid, ()))),
        )
        for pid in titles
    ]
    print(f"[scroll] 청크 {chunk_total} → 문서(page_id) {len(docs)}")
    return docs


def report(docs: list[GraphDocument], threshold: int) -> None:
    g = build_graph(docs, min_concept_docs=threshold)

    by_type = Counter(n.type for n in g.nodes)
    by_rel = Counter(e.rel for e in g.edges)

    # concept별 연결 문서 / 그 문서들의 domain 집합
    concept_docs: dict[str, set[str]] = defaultdict(set)
    concept_domains: dict[str, set[str]] = defaultdict(set)
    doc_domain = {f"doc::{d.page_id}": (d.domain or "기타") for d in docs}
    docs_with_concept: set[str] = set()
    for e in g.edges:
        if e.rel == "MENTIONS":
            concept_docs[e.target].add(e.source)
            concept_domains[e.target].add(doc_domain.get(e.source, "기타"))
            docs_with_concept.add(e.source)

    n_concepts = len(concept_docs)
    cross = [c for c, doms in concept_domains.items() if len(doms) >= 2]
    total_docs = len(docs)
    isolated = total_docs - len(docs_with_concept)

    degrees = sorted(((len(d), c) for c, d in concept_docs.items()), reverse=True)
    top_degree = degrees[0][0] if degrees else 0
    hairball_pct = (top_degree / total_docs * 100) if total_docs else 0
    avg_degree = (sum(d for d, _ in degrees) / n_concepts) if n_concepts else 0

    print(f"\n===== min_concept_docs = {threshold} =====")
    print(f"노드 {len(g.nodes)}  | " + "  ".join(f"{t}:{c}" for t, c in by_type.items()))
    print(f"엣지 {len(g.edges)}  | " + "  ".join(f"{r}:{c}" for r, c in by_rel.items()))
    print(f"concept 노드            : {n_concepts}")
    print(f"⭐ cross-domain concept : {len(cross)} ({(len(cross)/n_concepts*100 if n_concepts else 0):.0f}%)  <- 차별점 지표")
    print(f"   고립 문서(개념 0)     : {isolated}/{total_docs} ({(isolated/total_docs*100 if total_docs else 0):.0f}%)  <- 밋밋함 위험")
    print(f"   concept 평균 degree   : {avg_degree:.1f}  / 최대 degree {top_degree} ({hairball_pct:.0f}% 문서)  <- hairball 위험")
    print(f"   top concept(문서수)   : " + ", ".join(f"{c.replace('concept::','')}({d})" for d, c in degrees[:8]))
    print(f"   cross-domain 예시     : " + ", ".join(
        f"{c.replace('concept::','')}[{'/'.join(sorted(concept_domains[c]))}]" for c in cross[:6]
    ))


# ── graphReal.ts 스냅샷 생성 (--emit) ──────────────────────────────────────
# 프론트(onramp-web/src/api/graphReal.ts)가 소비하는 SphNode 트리 + SphEdge(의미엣지 type="s").
# 계층: ROOT → SOURCE → FOLDER(source×domain) → PAGE. 의미엣지=cross-domain concept이 잇는 도메인 대표 page 쌍.
_SRC_DISP = {"apache": "Apache", "datadog": "Datadog", "prometheus": "Prometheus", "kubernetes": "Kubernetes", "confluence": "Confluence", "github": "GitHub"}
_SRC_ORDER = ["Apache", "Datadog", "Prometheus", "Kubernetes", "Confluence", "GitHub"]
_SRC_COLOR = {"Kubernetes": "#4aa3ff", "Datadog": "#b48bff", "Apache": "#2bd4bd", "Prometheus": "#ff9a4d", "Confluence": "#fb7185", "GitHub": "#fbbf24", "기타": "#9fb6da"}
_DOMAIN_KO = {"manual": "운영매뉴얼", "api_reference": "API명세", "incident": "장애대응", "meeting_note": "회의록", "planning": "기획서"}


def _disp(src: str) -> str:
    s = (src or "").lower()
    return _SRC_DISP.get(s, (src or "기타").capitalize())


def build_snapshot(docs: list[GraphDocument], *, min_concept_docs: int = 3, max_tags: int = 8, max_sem: int = 600):
    cleaned = clean_docs(docs)
    doc_by_id = {d.page_id: d for d in cleaned}

    concept_pages: dict[str, set[str]] = defaultdict(set)
    for d in cleaned:
        for k in set(d.keywords):
            concept_pages[k].add(d.page_id)
    concepts = {k for k, p in concept_pages.items() if len(p) >= min_concept_docs}

    src_pages: dict[str, list[GraphDocument]] = defaultdict(list)
    fol_pages: dict[tuple[str, str], list[GraphDocument]] = defaultdict(list)
    for d in cleaned:
        src_pages[_disp(d.source)].append(d)
        fol_pages[(_disp(d.source), d.domain or "기타")].append(d)

    nodes: list[dict] = [
        {"id": "ROOT", "t": "OnRamp", "s": "기타", "k": "root", "p": None,
         "d": len(cleaned), "tg": [], "ver": 0, "mod": "",
         "sum": f"OnRamp 지식베이스 — {len(src_pages)}개 소스 · {len(cleaned)}개 문서 (멀티소스 적재)."}
    ]
    for disp in [s for s in _SRC_ORDER if s in src_pages]:
        nodes.append({"id": f"SRC::{disp}", "t": disp, "s": disp, "k": "source", "p": "ROOT",
                      "d": len(src_pages[disp]), "tg": [], "ver": 0, "mod": "", "sum": f"{disp} 문서 {len(src_pages[disp])}건."})
    for (disp, dom), ds in fol_pages.items():
        dom_ko = _DOMAIN_KO.get(dom, dom)
        nodes.append({"id": f"FOL::{disp}::{dom}", "t": dom_ko, "s": disp, "k": "folder", "p": f"SRC::{disp}",
                      "d": len(ds), "tg": [], "ver": 0, "mod": "", "sum": f"{disp} · {dom_ko} {len(ds)}건."})
    for d in cleaned:
        tags = [k for k in d.keywords if k in concepts][:max_tags] or list(d.keywords)[:max_tags]
        nodes.append({"id": d.page_id, "t": strip_upload_suffix(d.title) or d.page_id, "s": _disp(d.source), "k": "page",
                      "p": f"FOL::{_disp(d.source)}::{d.domain or '기타'}", "d": 1, "tg": list(tags),
                      "ver": 1 if d.version else 0, "mod": (d.last_modified or "")[:10], "sum": ""})

    edges: list[dict] = []
    for disp in src_pages:
        edges.append({"a": "ROOT", "b": f"SRC::{disp}"})
    for (disp, dom) in fol_pages:
        edges.append({"a": f"SRC::{disp}", "b": f"FOL::{disp}::{dom}"})
    for d in cleaned:
        edges.append({"a": f"FOL::{_disp(d.source)}::{d.domain or '기타'}", "b": d.page_id})

    # 의미 엣지: cross-domain concept이 잇는 도메인 대표 page 체인 (degree 높은 concept 우선)
    sem: list[dict] = []
    for c in sorted(concepts, key=lambda k: -len(concept_pages[k])):
        by_dom: dict[str, list[str]] = defaultdict(list)
        for pid in concept_pages[c]:
            dd = doc_by_id.get(pid)
            if dd:
                by_dom[dd.domain or "기타"].append(pid)
        if len(by_dom) < 2:
            continue
        reps = [pids[0] for pids in by_dom.values()]
        for i in range(len(reps) - 1):
            sem.append({"a": reps[i], "b": reps[i + 1], "type": "s", "c": c})
    sem = sem[:max_sem]
    edges += sem
    return nodes, edges, len(sem)


def emit_ts(nodes: list[dict], edges: list[dict]) -> str:
    src_order = [s for s in _SRC_ORDER if any(n["s"] == s for n in nodes)]
    meta = {s: {"color": _SRC_COLOR.get(s, "#9fb6da")} for s in src_order}
    meta["기타"] = {"color": _SRC_COLOR["기타"]}
    j = lambda o: json.dumps(o, ensure_ascii=False)  # noqa: E731
    return "\n".join([
        "/* eslint-disable */",
        "// @ts-nocheck — 자동생성 데이터 파일. 거대 리터럴 TS2590 회피 + eslint 제외 (타입은 SphNode/SphEdge 인터페이스 참조).",
        "// AUTO-GENERATED — onramp-api/scripts/validate_knowledge_graph.py --emit (실 Qdrant 적재 기준).",
        "// 뉴런구체 + 온톨로지(의미엣지) 지식맵. 계층 ROOT→SOURCE→FOLDER(domain)→PAGE + cross-domain 의미엣지.",
        "export interface SphNode { id:string; t:string; s:string; k:'root'|'source'|'folder'|'page'; p:string|null; d:number; tg:string[]; ver:number; mod:string; sum:string }",
        "export interface SphEdge { a:string; b:string; type?:'s'; c?:string }",
        f"export const SOURCE_META: Record<string,{{color:string}}> = {j(meta)}",
        f"export const SOURCE_ORDER = {j(src_order)} as const",
        f"export const SPH_NODES: SphNode[] = {j(nodes)}",
        f"export const SPH_EDGES: SphEdge[] = {j(edges)}",
        "",
    ])


# ── AI 자동 요약 (source/folder 노드만, 실 LLM 호출) ──────────────────────
# page(1266)는 과다하므로 제외. source(6)+folder(19)=25개만 → 비용·시간 최소.
async def _summarize_node(label: str, kind_ko: str, concepts: list[str], titles: list[str], settings) -> str:
    sys_prompt = "기술 지식베이스의 한 영역이 어떤 지식을 담는지 한 문장으로 소개하는 큐레이터다."
    user_prompt = (
        f"영역: {label} ({kind_ko})\n"
        f"핵심 개념: {', '.join(concepts[:12]) or '(없음)'}\n"
        f"대표 문서: {', '.join(titles[:6]) or '(없음)'}\n"
        "이 영역이 담는 지식을 한국어 한 문장(35자 내외)으로 요약하라. 따옴표·머리말 없이 문장만 출력."
    )
    try:
        out = await call_llm(sys_prompt, user_prompt, max_tokens=80, temperature=0.3, settings=settings)
        return out.strip().strip('"').strip()
    except Exception:
        return ""  # 실패 시 빈 문자열 → 호출측이 카운트 sum 유지


async def enrich_summaries(nodes: list[dict], docs: list[GraphDocument], *, min_concept_docs: int = 3) -> int:
    settings = get_settings()
    cleaned = clean_docs(docs)
    concept_pages: dict[str, set[str]] = defaultdict(set)
    for d in cleaned:
        for k in set(d.keywords):
            concept_pages[k].add(d.page_id)
    concepts_set = {k for k, p in concept_pages.items() if len(p) >= min_concept_docs}

    src_c: dict[str, Counter] = defaultdict(Counter)
    fol_c: dict[tuple, Counter] = defaultdict(Counter)
    src_t: dict[str, list] = defaultdict(list)
    fol_t: dict[tuple, list] = defaultdict(list)
    for d in cleaned:
        disp, dom = _disp(d.source), d.domain or "기타"
        title = strip_upload_suffix(d.title)
        src_t[disp].append(title)
        fol_t[(disp, dom)].append(title)
        for k in set(d.keywords):
            if k in concepts_set:
                src_c[disp][k] += 1
                fol_c[(disp, dom)][k] += 1

    targets: list[dict] = []
    tasks = []
    for n in nodes:
        if n["k"] == "source":
            disp = n["t"]
            tasks.append(_summarize_node(disp, "소스", [c for c, _ in src_c[disp].most_common(12)], src_t[disp], settings))
            targets.append(n)
        elif n["k"] == "folder":
            _, disp, dom = n["id"].split("::")  # FOL::disp::dom
            tasks.append(_summarize_node(f"{disp} · {n['t']}", "도메인", [c for c, _ in fol_c[(disp, dom)].most_common(12)], fol_t[(disp, dom)], settings))
            targets.append(n)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    n_ok = 0
    for node, r in zip(targets, results):
        if isinstance(r, str) and r:
            node["sum"] = r
            n_ok += 1
    return n_ok


def run_emit() -> None:
    docs = collect_documents()
    if not docs:
        print("⚠️ 문서 0건 — 적재 상태 확인 필요.")
        return
    nodes, edges, n_sem = build_snapshot(docs, min_concept_docs=3)
    n_sum = asyncio.run(enrich_summaries(nodes, docs))
    print(f"[summary] source/folder AI 요약 생성: {n_sum}/25")
    out = "/tmp/graphReal.ts"
    with open(out, "w", encoding="utf-8") as f:
        f.write(emit_ts(nodes, edges))
    by_k = Counter(n["k"] for n in nodes)
    print(f"[emit] {out} 생성")
    print(f"  노드 {len(nodes)} (" + " ".join(f"{k}:{c}" for k, c in by_k.items()) + ")")
    print(f"  엣지 {len(edges)} (계층 {len(edges) - n_sem} + 의미 {n_sem})")


def main() -> None:
    if "--emit" in sys.argv:
        run_emit()
        return
    docs = collect_documents()
    if not docs:
        print("⚠️ 문서 0건 — Qdrant 컬렉션이 비었거나 page_id payload 없음. 적재 상태 확인 필요.")
        return
    print(f"source 분포: {dict(Counter(d.source for d in docs))}")
    print(f"domain 분포: {dict(Counter(d.domain or '기타' for d in docs))}")

    diagnose_keywords(docs)

    print("\n##################### RAW (정제 전) #####################")
    report(docs, 3)

    cleaned = clean_docs(docs)
    print("\n##################### CLEANED (정제 후) #####################")
    for th in (2, 3, 4):
        report(cleaned, th)


if __name__ == "__main__":
    main()
