"""POST /v1/chat 통합 테스트.

LLM·retriever 경계를 stub해 네트워크 없이 그래프 전체(Router→Retriever→Answer)를
실제로 통과시키고 ChatResponse 매핑까지 검증한다.
"""

import json

import pytest


def _router_resp(use_case: str = "검색", domain: str = "incident") -> str:
    """Router LLM 응답(JSON 문자열)을 만든다."""
    domains = [domain] if domain else []
    return json.dumps({"use_case": use_case, "domains": domains, "refined_query": "정제된 질문", "confidence": 0.9})


def _answer_resp() -> str:
    """Answer LLM 응답(answerable 5요소 JSON)을 만든다."""
    return json.dumps(
        {
            "situation": "상황",
            "cause": "원인",
            "evidence": "근거",
            "solution": "해결",
            "infra_context": "맥락",
            "answerability_status": "answerable",
            "source_indices": [0],
        }
    )


def _mk(resp: str):
    """고정 문자열을 반환하는 call_llm 대체(async) 함수를 만든다."""

    async def _call(*args, **kwargs):
        return resp

    return _call


class _Embedder:
    """retriever용 임베더 stub."""

    async def embed_query(self, text: str) -> list[float]:
        """쿼리 임베딩 대신 고정 벡터를 반환한다."""
        return [0.0, 0.0, 0.0]


def _hit():
    """Qdrant 검색 결과 한 건(payload + score)을 흉내 낸다."""
    payload = {
        "chunk_id": "c1",
        "content": "EKS Pod CrashLoopBackOff 대응 절차",
        "page_title": "장애 대응 가이드",
        "source_url": "http://x",
        "space_key": "OnRamp",
        "domain": "incident",
        "last_modified": "",
        "site": "kubernetes",
        "product_version": "v1.33",
        "doc_key": "kubernetes:장애-대응-가이드",
    }
    # id 포함 — hybrid 경로의 _merge_hits()가 point.id로 dedupe한다 (fixture 공유 대비)
    return type("SP", (), {"id": "c1", "payload": payload, "score": 0.9})()


class _Reranker:
    """리랭커 stub."""

    def rerank(self, query, candidates):
        """후보를 τ(trust_rerank_floor) 위 점수로 통과시킨다 — 보정으로 τ가 바뀌어도 의도 유지."""
        from app.config import get_settings

        passing = get_settings().trust_rerank_floor + 0.1
        return [(passing, payload) for _, payload in candidates]


@pytest.fixture
def stub_pipeline(monkeypatch):
    """router/answer LLM + retriever 임베더/검색/리랭커를 stub (네트워크 0)."""

    async def _search(qvec, top_k, *, domain=None, **kwargs):
        return [_hit()]

    def _lineages(keys, **kwargs):
        # 계보 facet은 라이브 Qdrant 의존 — CI(서비스 없음)에서도 돌도록 스텁 (#108)
        return {k: frozenset({"v1.33"}) if k else frozenset() for k in keys}

    monkeypatch.setattr("app.agents.router.node.call_llm", _mk(_router_resp()))
    monkeypatch.setattr("app.agents.answer.node.call_llm", _mk(_answer_resp()))
    monkeypatch.setattr("app.agents.retriever.node.get_embedder", lambda *a, **k: _Embedder())
    monkeypatch.setattr("app.agents.retriever.search.dense_search", _search)
    monkeypatch.setattr("app.agents.retriever.node.get_reranker", lambda *a, **k: _Reranker())
    monkeypatch.setattr("app.agents.retriever.node.get_lineages", _lineages)
    monkeypatch.setattr("app.agents.trust.node.get_lineages", _lineages)
    return monkeypatch


@pytest.mark.asyncio
async def test_chat_success(client, stub_pipeline):
    """검색 질문 → 200 + answerable 5요소 + 출처 매핑."""
    resp = await client.post("/v1/chat", json={"query": "EKS Pod 장애 해결법"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answerability_status"] == "answerable"
    assert data["answer"]["situation"] != ""
    assert data["domain"] != ""
    assert len(data["sources"]) > 0
    # 버전 계보 메타 노출 (#108) — 비교 질의에서 출처 버전 구분용
    assert data["sources"][0]["site"] == "kubernetes"
    assert data["sources"][0]["product_version"] == "v1.33"


@pytest.mark.asyncio
async def test_chat_unanswerable(client, stub_pipeline):
    """router가 UNANSWERABLE → 즉시 종료, status 미설정 → 기본 NOT_ENOUGH."""
    stub_pipeline.setattr("app.agents.router.node.call_llm", _mk(_router_resp(use_case="답변불가")))
    resp = await client.post("/v1/chat", json={"query": "오늘 점심 뭐 먹지"})
    assert resp.status_code == 200
    assert resp.json()["answerability_status"] != "answerable"


@pytest.mark.asyncio
async def test_chat_no_model_passes_empty_to_router(client, stub_pipeline):
    """model 미지정 시 routing model은 빈 문자열 — default_model이 provider 선택으로 새지 않음."""
    captured: dict = {}

    async def _router(system, user, **kwargs):
        captured["model"] = kwargs.get("model", "MISSING")
        return _router_resp()

    stub_pipeline.setattr("app.agents.router.node.call_llm", _router)
    await client.post("/v1/chat", json={"query": "테스트"})
    assert captured["model"] == ""


@pytest.mark.asyncio
async def test_chat_empty_query(client):
    """빈 query는 Pydantic 검증 실패로 422."""
    resp = await client.post("/v1/chat", json={"query": ""})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_chat_with_model(client, stub_pipeline):
    """요청 model이 응답 model_used로 전달된다."""
    resp = await client.post("/v1/chat", json={"query": "테스트 질문", "model": "gpt-4o"})
    assert resp.status_code == 200
    assert resp.json()["model_used"] == "gpt-4o"


@pytest.mark.asyncio
async def test_chat_swagger(client):
    """Swagger 문서(/docs) 접근 가능."""
    resp = await client.get("/docs")
    assert resp.status_code == 200
