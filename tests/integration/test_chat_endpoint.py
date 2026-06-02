"""POST /v1/chat 통합 테스트.

LLM·retriever 경계를 stub해 네트워크 없이 그래프 전체(Router→Retriever→Answer)를
실제로 통과시키고 ChatResponse 매핑까지 검증한다.
"""

import json

import pytest


def _router_resp(use_case: str = "검색", domain: str = "incident") -> str:
    return json.dumps({"use_case": use_case, "domain": domain, "refined_query": "정제된 질문", "confidence": 0.9})


def _answer_resp() -> str:
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
    async def _call(*args, **kwargs):
        return resp

    return _call


class _Embedder:
    async def embed_query(self, text: str) -> list[float]:
        return [0.0, 0.0, 0.0]


def _hit():
    payload = {
        "chunk_id": "c1",
        "content": "EKS Pod CrashLoopBackOff 대응 절차",
        "page_title": "장애 대응 가이드",
        "source_url": "http://x",
        "space_key": "OnRamp",
        "domain": "incident",
        "last_modified": "",
    }
    return type("SP", (), {"payload": payload, "score": 0.9})()


class _Reranker:
    def rerank(self, query, candidates):
        return [(0.5, payload) for _, payload in candidates]


@pytest.fixture
def stub_pipeline(monkeypatch):
    """router/answer LLM + retriever 임베더/검색/리랭커를 stub (네트워크 0)."""

    async def _search(qvec, top_k, *, domain=None, **kwargs):
        return [_hit()]

    monkeypatch.setattr("app.agents.router.node.call_llm", _mk(_router_resp()))
    monkeypatch.setattr("app.agents.answer.node.call_llm", _mk(_answer_resp()))
    monkeypatch.setattr("app.agents.retriever.node.get_embedder", lambda *a, **k: _Embedder())
    monkeypatch.setattr("app.agents.retriever.node.dense_search", _search)
    monkeypatch.setattr("app.agents.retriever.node.get_reranker", lambda *a, **k: _Reranker())
    return monkeypatch


@pytest.mark.asyncio
async def test_chat_success(client, stub_pipeline):
    resp = await client.post("/v1/chat", json={"query": "EKS Pod 장애 해결법"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answerability_status"] == "answerable"
    assert data["answer"]["situation"] != ""
    assert data["domain"] != ""
    assert len(data["sources"]) > 0


@pytest.mark.asyncio
async def test_chat_unanswerable(client, stub_pipeline):
    # router가 UNANSWERABLE → 그래프 즉시 종료, answer 미실행 → status 미설정 → 기본 NOT_ENOUGH
    stub_pipeline.setattr("app.agents.router.node.call_llm", _mk(_router_resp(use_case="답변불가")))
    resp = await client.post("/v1/chat", json={"query": "오늘 점심 뭐 먹지"})
    assert resp.status_code == 200
    assert resp.json()["answerability_status"] != "answerable"


@pytest.mark.asyncio
async def test_chat_empty_query(client):
    resp = await client.post("/v1/chat", json={"query": ""})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_chat_with_model(client, stub_pipeline):
    resp = await client.post("/v1/chat", json={"query": "테스트 질문", "model": "gpt-4o"})
    assert resp.status_code == 200
    assert resp.json()["model_used"] == "gpt-4o"


@pytest.mark.asyncio
async def test_chat_swagger(client):
    resp = await client.get("/docs")
    assert resp.status_code == 200
