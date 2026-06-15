"""RemoteReranker (#72/#73) — 서비스 없이 fake client로 매핑·정렬·폴백·백엔드 분기·동적 URL 검증."""

import time

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


def _prime(r: RemoteReranker, client, url: str) -> None:
    # #73: Redis 조회 없이 고정 URL+클라 주입 — _resolve_url 캐시를 채우고 _client_url을 맞춰
    # _get_client가 주입한 fake client를 그대로 반환하게 한다(단위 테스트 격리).
    r._client = client  # type: ignore[assignment]
    r._client_url = url
    r._resolved_url = url
    r._url_resolved = True  # 캐시 적중 → rerank가 실제 Redis를 조회하지 않게
    r._url_checked_at = time.monotonic()


def _remote(scores: list[float]) -> RemoteReranker:
    r = RemoteReranker(Settings(reranker_backend="remote", reranker_service_url="http://x:8080"))
    _prime(r, _FakeClient(scores), "http://x:8080")
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


def test_settings_remote_allows_empty_url_redis_supplies():
    # #73: remote여도 env URL이 비어 있을 수 있다(런타임 Redis가 공급) → 기동은 통과.
    s = Settings(reranker_backend="remote", reranker_service_url="")
    assert s.reranker_service_url == ""


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
    _ = r._get_client("http://onramp-reranker:8080")  # 클라이언트 생성
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
    _prime(r, client, "http://x:8080")
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


# ── #73 동적 URL (Redis 우선 → env 폴백) ───────────────────────────────


class _FakeRedis:
    """get(key) → 지정값/예외. Redis 없이 _resolve_url 분기 검증."""

    def __init__(self, value=None, raises: bool = False) -> None:
        self._value = value
        self._raises = raises

    def get(self, key: str):
        if self._raises:
            raise RuntimeError("redis down")
        return self._value


def _resolver(value=None, raises: bool = False, env_url: str = "http://env:8080") -> RemoteReranker:
    r = RemoteReranker(Settings(reranker_backend="remote", reranker_service_url=env_url))
    r._redis = _FakeRedis(value, raises)  # 지연 redis 클라 대체
    return r


def test_resolve_url_prefers_redis():
    r = _resolver(value=b"http://vessl-new:8080")
    assert r._resolve_url() == "http://vessl-new:8080"  # Redis 값 우선


def test_resolve_url_falls_back_to_env_when_redis_empty():
    r = _resolver(value=None)  # 키 없음(GPU OFF/미설정)
    assert r._resolve_url() == "http://env:8080"  # settings 폴백


def test_resolve_url_falls_back_on_redis_error():
    r = _resolver(raises=True)  # Redis 장애
    assert r._resolve_url() == "http://env:8080"  # 예외 삼키고 settings 폴백


def test_resolve_url_empty_when_no_source():
    r = _resolver(value=None, env_url="")  # Redis도 env도 없음
    assert r._resolve_url() == ""  # rerank가 RerankerUnavailableError로 폴백 유도


def test_url_change_recreates_client():
    r = _resolver(value=b"http://url-a:8080")
    c1 = r._get_client(r._resolve_url())
    assert r._client_url == "http://url-a:8080"
    # 스핀업으로 URL 변경 → 캐시 만료 모사 후 새 URL → 새 클라이언트
    r._redis = _FakeRedis(b"http://url-b:8080")
    r._url_checked_at = 0.0  # 캐시 무효화
    c2 = r._get_client(r._resolve_url())
    assert r._client_url == "http://url-b:8080"
    assert c2 is not c1  # 기존 클라 닫고 재생성


def test_rerank_raises_unavailable_when_no_url():
    # env·Redis 둘 다 비면 호출 불가 → 회로/폴백 유도 (조용히 빈 결과 X)
    r = _resolver(value=None, env_url="")
    with pytest.raises(RerankerUnavailableError):
        r.rerank("q", [("a", {})])


def test_no_url_records_failure_and_caches_empty():
    # 리뷰: 빈 URL도 실패로 집계(회로 open 유도) + 결과 캐시(매 요청 Redis 재조회 방지)
    r = _resolver(value=None, env_url="")
    with pytest.raises(RerankerUnavailableError):
        r.rerank("q", [("a", {})])
    assert r._cb_failures == 1  # _record_failure로 집계됨
    assert r._url_resolved is True  # 빈 결과도 캐시 → 다음 호출은 Redis 재조회 안 함


def test_no_url_opens_breaker_after_threshold():
    r = _resolver(value=None, env_url="")
    for _ in range(3):  # 임계치(3)까지 실패 집계
        with pytest.raises(RerankerUnavailableError):
            r.rerank("q", [("a", {})])
    assert r._cb_open_until > time.monotonic()  # 회로 open → 이후 _resolve_url 자체 스킵


def test_resolve_url_rejects_malformed_redis_url():
    # 리뷰: Redis 값 검증 — 형식 불량이면 base_url로 쓰지 않고 env 폴백(SSRF 방지)
    r = _resolver(value=b"not-a-url", env_url="http://env:8080")
    assert r._resolve_url() == "http://env:8080"


def test_resolve_url_rejects_non_http_redis_url():
    r = _resolver(value=b"ftp://evil:21", env_url="http://env:8080")
    assert r._resolve_url() == "http://env:8080"  # http/https만 허용


# ── #123 리뷰: half-open 단일 probe 게이트 ───────────────────────────────


def _half_open(client, failures: int = 3) -> RemoteReranker:
    r = _breaker_remote(client)
    r._cb_failures = failures
    r._cb_open_until = time.monotonic() - 1  # 쿨다운 만료(half-open), != 0
    return r


def test_half_open_rejects_when_probe_inflight():
    fc = _FailingClient()
    r = _half_open(fc)
    r._cb_probe_inflight = True  # 이미 다른 요청이 probe 중
    with pytest.raises(RerankerUnavailableError):
        r.rerank("q", [("a", {})])
    assert fc.calls == 0  # probe 점유 중 → 원격 호출 스킵(즉시 폴백)


def test_half_open_allows_single_probe_and_closes_on_success():
    fc = _FakeClient([0.5])
    r = _half_open(fc)
    out = r.rerank("q", [("a", {"chunk_id": "a"})])  # probe 1개 통과
    assert out == [(0.5, {"chunk_id": "a"})]
    assert r._cb_probe_inflight is False  # finally에서 점유 해제
    assert r._cb_open_until == 0.0 and r._cb_failures == 0  # 성공 → close


def test_half_open_probe_failure_reopens():
    fc = _FailingClient()
    r = _half_open(fc)
    with pytest.raises(RuntimeError):
        r.rerank("q", [("a", {})])  # probe 실패
    assert fc.calls == 1  # probe는 실제 1회 호출
    assert r._cb_probe_inflight is False  # 점유 해제
    assert r._cb_open_until > time.monotonic()  # 재오픈(쿨다운 갱신)
