"""RemoteReranker (#72) — 서비스 없이 fake client로 매핑·정렬·폴백·백엔드 분기 검증."""

import pytest
from pydantic import ValidationError

from app.agents.retriever.rerank import RemoteReranker, RerankerUnavailableError, get_reranker, reset_reranker
from app.config import Settings


class _FakeResp:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.calls: list[tuple[str, dict]] = []

    def post(self, url: str, json: dict):  # noqa: A002 - httpx 시그니처 모사
        self.calls.append((url, json))
        return _FakeResp({"scores": self._scores})


def _remote(scores: list[float]) -> RemoteReranker:
    r = RemoteReranker(Settings(reranker_backend="remote", reranker_service_url="http://x:8080"))
    r._client = _FakeClient(scores)  # type: ignore[assignment]
    return r


def test_rerank_maps_scores_to_payloads_and_sorts_desc():
    r = _remote([0.2, 0.9, 0.5])
    cands = [("a", {"chunk_id": "a"}), ("b", {"chunk_id": "b"}), ("c", {"chunk_id": "c"})]
    out = r.rerank("질의", cands)
    assert [p["chunk_id"] for _, p in out] == ["b", "c", "a"]  # 점수 내림차순
    assert [round(s, 2) for s, _ in out] == [0.9, 0.5, 0.2]
    url, body = r._client.calls[0]  # type: ignore[attr-defined]
    assert url == "/rerank" and body == {"query": "질의", "passages": ["a", "b", "c"]}  # passages만 전송


def test_rerank_empty_candidates_does_not_call_service():
    r = _remote([])
    assert r.rerank("q", []) == []
    assert r._client.calls == []  # type: ignore[attr-defined]


def test_rerank_length_mismatch_raises():
    # 후보 2개에 점수 1개 → strict zip ValueError → retriever_node가 잡아 vector 폴백
    r = _remote([0.1])
    with pytest.raises(ValueError):
        r.rerank("q", [("a", {}), ("b", {})])


def test_get_reranker_returns_remote_for_remote_backend():
    reset_reranker()
    s = Settings(reranker_backend="remote", reranker_service_url="http://onramp-reranker:8080")
    assert isinstance(get_reranker(s), RemoteReranker)
    reset_reranker()


def test_settings_remote_requires_service_url():
    with pytest.raises(ValidationError):
        Settings(reranker_backend="remote", reranker_service_url="")


@pytest.mark.parametrize("bad_url", ["onramp-reranker:8080", "ftp://host:8080", "not a url"])
def test_settings_remote_rejects_malformed_url(bad_url):
    # fail-fast: 스킴/형식이 잘못된 URL은 기동 시 거부 (첫 요청까지 미루지 않는다)
    with pytest.raises(ValidationError):
        Settings(reranker_backend="remote", reranker_service_url=bad_url)


def test_reset_reranker_closes_remote_client():
    # 교체/리셋 시 httpx 연결을 닫는다 (커넥션 누수 방지)
    reset_reranker()
    s = Settings(reranker_backend="remote", reranker_service_url="http://onramp-reranker:8080")
    r = get_reranker(s)
    assert isinstance(r, RemoteReranker)
    _ = r.client  # lazy 클라이언트 생성
    assert r._client is not None
    reset_reranker()
    assert r._client is None  # close()로 정리됨


class _FailingClient:
    def __init__(self) -> None:
        self.calls = 0

    def post(self, url: str, json: dict):  # noqa: A002
        self.calls += 1
        raise RuntimeError("boom")  # 연결/응답 실패 모사


class _FlakyClient:
    """처음 fail_first 번은 실패, 그 뒤 성공 — 서킷 복구(half-open→close) 검증용."""

    def __init__(self, scores: list[float], fail_first: int) -> None:
        self._scores = scores
        self._fail_remaining = fail_first
        self.calls = 0

    def post(self, url: str, json: dict):  # noqa: A002
        self.calls += 1
        if self._fail_remaining > 0:
            self._fail_remaining -= 1
            raise RuntimeError("boom")
        return _FakeResp({"scores": self._scores})


def _breaker_remote(client) -> RemoteReranker:
    r = RemoteReranker(
        Settings(
            reranker_backend="remote",
            reranker_service_url="http://x:8080",
            reranker_breaker_fail_threshold=3,
            reranker_breaker_cooldown_s=30,
        )
    )
    r._client = client  # type: ignore[assignment]
    return r


def test_circuit_breaker_opens_after_threshold_and_skips_calls():
    fc = _FailingClient()
    r = _breaker_remote(fc)
    cands = [("a", {})]
    for _ in range(3):  # 임계치(3)까지 실제 호출되어 실패
        with pytest.raises(RuntimeError):
            r.rerank("q", cands)
    assert fc.calls == 3
    # 회로 open → 다음 호출은 스킵하고 RerankerUnavailableError (vector 폴백 유도)
    with pytest.raises(RerankerUnavailableError):
        r.rerank("q", cands)
    assert fc.calls == 3  # 호출 안 늘어남 — 즉시 폴백


def test_circuit_breaker_resets_on_success():
    fc = _FlakyClient([0.5], fail_first=2)  # 2회 실패(임계치 3 미만) 후 성공
    r = _breaker_remote(fc)
    cands = [("a", {"chunk_id": "a"})]
    for _ in range(2):
        with pytest.raises(RuntimeError):
            r.rerank("q", cands)
    out = r.rerank("q", cands)  # 3번째는 성공 → 카운터 리셋
    assert out == [(0.5, {"chunk_id": "a"})]
    assert r._cb_failures == 0  # 성공으로 초기화 (회로 안 열림)
