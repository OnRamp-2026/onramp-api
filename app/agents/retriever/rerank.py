"""bge-reranker Cross-Encoder 리랭킹 + 메타 가중 (검색측)."""

from __future__ import annotations

import logging
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from app.config import Settings, get_settings
from app.rag.version_fit import version_fit_from_payload

logger = logging.getLogger(__name__)


class RerankerUnavailableError(RuntimeError):
    """서킷브레이커 open 등으로 remote 리랭커 호출을 건너뛸 때 — retriever_node가 잡아 vector 폴백."""


class CrossEncoderReranker:
    """bge-reranker-v2-m3 Cross-Encoder. 모델은 첫 rerank 시 lazy-load."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model = None
        self._lock = threading.Lock()

    @property
    def model(self):
        # double-checked locking — anyio 스레드에서 동시 cold-start 시 모델 중복 로드 방지
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder  # 무거운 로드 → 지연

                    self._model = CrossEncoder(self.settings.reranker_model, device=self.settings.reranker_device)
        return self._model

    def rerank(self, query: str, candidates: list[tuple[str, dict]]) -> list[tuple[float, dict]]:
        if not candidates:
            return []
        pairs = [(query, text) for text, _ in candidates]
        scores = self.model.predict(pairs)
        ranked = [(float(score), payload) for score, (_, payload) in zip(scores, candidates, strict=True)]
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked


class OnnxCrossEncoderReranker:
    """#60: ONNX(int8) bge-reranker. CrossEncoderReranker와 동일 인터페이스, CPU 파드 경량화용.

    검증 범위(standalone): 변환(fp32→int8) + CPU 속도 + 골든셋 품질만 확인.
    동일 모델 그대로 양자화하므로 다국어 보존. in-app 경로는 운영 파드에서 재검증 필요(그래서 기본 backend는 torch).
    모델 디렉토리는 scripts/build_reranker_onnx.py로 사전 생성한다(산출물 model_quantized.onnx).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._session: Any = None  # onnxruntime.InferenceSession (지연 로드)
        self._tokenizer: Any = None  # transformers tokenizer (지연 로드)
        self._input_names: set[str] = set()
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        # 순수 onnxruntime + numpy로 추론 — torch 의존 없음([onnx] extra만으로 동작).
        if self._session is None:
            with self._lock:  # double-checked locking — 동시 cold-start 시 중복 로드 방지
                if self._session is None:
                    model_path = Path(self.settings.reranker_onnx_dir, self.settings.reranker_onnx_file)
                    if not model_path.is_file():
                        # config 검증을 우회한 경우의 2차 방어선(테스트/직접 생성). 정상 기동은 config가 먼저 막는다.
                        raise RuntimeError(
                            f"reranker_backend='onnx' 모델 파일 없음: {model_path} "
                            "(scripts/build_reranker_onnx.py 먼저 실행)"
                        )
                    import onnxruntime as ort  # 무거운 로드 → 지연
                    from transformers import AutoTokenizer

                    self._tokenizer = AutoTokenizer.from_pretrained(self.settings.reranker_model)
                    self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
                    self._input_names = {i.name for i in self._session.get_inputs()}

    def rerank(self, query: str, candidates: list[tuple[str, dict]]) -> list[tuple[float, dict]]:
        if not candidates:
            return []
        self._ensure_loaded()
        import numpy as np  # 지연

        passages = [text for text, _ in candidates]
        features = self._tokenizer(
            [query] * len(passages),
            passages,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )
        # 양자화 그래프가 받는 입력만 전달 (모델별 token_type_ids 유무 차이 흡수)
        inputs = {k: v for k, v in features.items() if k in self._input_names}
        logits = self._session.run(None, inputs)[0]
        scores = (1.0 / (1.0 + np.exp(-logits))).reshape(-1).tolist()  # sigmoid: torch 백엔드와 동일 점수 계약
        ranked = [(float(score), payload) for score, (_, payload) in zip(scores, candidates, strict=True)]
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked


class RemoteReranker:
    """#72: 리랭킹을 별도 서비스(onramp-reranker)에 위임. /rerank HTTP 호출 — 메모리를 API 파드 밖으로 분리.

    CrossEncoder/Onnx 리랭커와 **동일 계약**(query, candidates → [(score[0,1], payload)] desc).
    호출/응답 실패는 raise → retriever_node가 잡아 vector 폴백(API는 안 죽는다).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client: httpx.Client | None = None
        self._lock = threading.Lock()
        # 클라이언트 사이드 서킷브레이커 상태(파드별 인메모리). 별도 서비스/게이트웨이 불필요.
        self._cb_lock = threading.Lock()
        self._cb_failures = 0
        self._cb_open_until = 0.0  # monotonic 시각. time.monotonic() < 이 값이면 회로 open.

    @property
    def client(self) -> httpx.Client:
        # 동기 httpx 클라이언트(연결 재사용). rerank는 anyio.to_thread.run_sync로 스레드에서 호출됨.
        # connect 타임아웃을 짧게 분리 — GPU(VESSL) OFF 시 매 요청이 read 타임아웃까지 매달리지 않게.
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = httpx.Client(
                        base_url=self.settings.reranker_service_url,
                        timeout=httpx.Timeout(
                            self.settings.reranker_timeout_s,
                            connect=self.settings.reranker_connect_timeout_s,
                        ),
                    )
        return self._client

    def rerank(self, query: str, candidates: list[tuple[str, dict]]) -> list[tuple[float, dict]]:
        if not candidates:
            return []
        # 회로 open이면 호출 스킵 → 즉시 예외 → vector 폴백(GPU OFF 동안 매 요청 connect 지연 방지).
        with self._cb_lock:
            if time.monotonic() < self._cb_open_until:
                raise RerankerUnavailableError("reranker circuit open")
        passages = [text for text, _ in candidates]
        try:
            resp = self.client.post("/rerank", json={"query": query, "passages": passages})
            resp.raise_for_status()
            scores = resp.json()["scores"]
        except Exception:
            self._record_failure()
            raise
        self._record_success()
        # strict=True: 서비스가 길이 불일치 점수를 주면 ValueError → 폴백(방어)
        ranked = [(float(score), payload) for score, (_, payload) in zip(scores, candidates, strict=True)]
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked

    def _record_failure(self) -> None:
        # 임계치 도달 + 현재 open 상태가 아니면 회로 open(쿨다운). half-open 프로브 실패 시 재오픈도 처리.
        with self._cb_lock:
            self._cb_failures += 1
            now = time.monotonic()
            if self._cb_failures >= self.settings.reranker_breaker_fail_threshold and now >= self._cb_open_until:
                self._cb_open_until = now + self.settings.reranker_breaker_cooldown_s
                logger.warning(
                    "리랭커 서킷브레이커 open — %d회 연속 실패, %.0fs 동안 vector 폴백",
                    self._cb_failures,
                    self.settings.reranker_breaker_cooldown_s,
                )

    def _record_success(self) -> None:
        # 성공 시 회로 닫고 카운터 초기화(half-open 프로브 성공 = 정상 복귀).
        with self._cb_lock:
            if self._cb_open_until != 0.0:
                logger.info("리랭커 서킷브레이커 close — 정상 복귀")
            self._cb_failures = 0
            self._cb_open_until = 0.0

    def close(self) -> None:
        # 인스턴스 교체/리셋 시 httpx 연결 풀 정리 (커넥션 누수 방지).
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None


def apply_metadata_weight(rerank_score: float, payload: dict, settings: Settings) -> float:
    """최신성 가산 — 최근 문서일수록 점수를 더한다. 가산식이라 음수 점수에서도 단조 증가한다."""
    factor = _recency_factor(payload.get("last_modified", ""), settings.rerank_recency_half_life_days)
    return rerank_score + settings.rerank_recency_weight * factor


def apply_domain_weight(
    rerank_score: float, payload: dict, query_domains: list[str] | None, settings: Settings
) -> float:
    """문서 단일 도메인이 질의 도메인 집합에 들면 점수를 더한다(Soft 가산, 순서 우선, #61).

    가산식이라 음수 점수(Cross-Encoder logit)에서도 단조 증가.
    문서 domain == query.domains[0] → primary 가중 / domains[1:] → secondary 가중 / 아니면 원점수.
    query_domains가 비면 가산 없음. (문서는 단일 도메인 — payload["domain"]만 본다.)
    """
    if not query_domains:
        return rerank_score
    doc = payload.get("domain")
    if not doc:
        return rerank_score
    if doc == query_domains[0]:
        return rerank_score + settings.domain_primary_weight
    if doc in query_domains[1:]:
        return rerank_score + settings.domain_secondary_weight
    return rerank_score


def apply_version_weight(
    rerank_score: float,
    payload: dict,
    lineages: dict[str, frozenset[str]],
    target_versions: list[str],
    settings: Settings,
) -> float:
    """버전 적합성(version_fit) 가산 (#103, 설계 7장).

    최신 버전(또는 질의 target 버전) 문서가 형제들보다 위로 오게 한다 — 버전 형제가
    top-k 슬롯을 낭비하는 것을 랭킹 단계에서 줄인다. 가산식·단조 증가(기존 부스트 계약).
    """
    fit = version_fit_from_payload(payload, lineages, target_versions, settings).fit
    return rerank_score + settings.rank_version_weight * fit


def apply_authority_weight(rerank_score: float, payload: dict, settings: Settings) -> float:
    """site 권위 등급 가산 (#103, 설계 4.3 — 어댑터 자리).

    현 코퍼스는 전부 공식 문서라 변별력이 없지만, 사내 Confluence 전환 시
    space 등급·verified 라벨이 이 자리를 채운다. 미등록 site는 중립 기본값.
    """
    tier = settings.site_tier.get(payload.get("site", "") or "", settings.site_tier_default)
    return rerank_score + settings.rank_authority_weight * tier


def apply_ranking_boosts(
    raw_score: float,
    payload: dict,
    query_domains: list[str] | None,
    lineages: dict[str, frozenset[str]],
    target_versions: list[str],
    settings: Settings,
) -> float:
    """부스트 체인 단일 진입점 — 운영(retrieve_node)과 평가(retrieval_adapter)가 같은 함수를 쓴다.

    반환값은 **정렬 전용 ranking 점수**다. raw 점수(τ 진단·SourceDocument.raw_rerank_score)는
    호출측이 별도로 보존해야 한다 (#103 점수 분리 — 설계 7.3 "정렬은 블렌드, 진단은 원점수").
    """
    score = apply_metadata_weight(raw_score, payload, settings)
    score = apply_domain_weight(score, payload, query_domains, settings)
    score = apply_version_weight(score, payload, lineages, target_versions, settings)
    return apply_authority_weight(score, payload, settings)


def _recency_factor(last_modified: str, half_life_days: int) -> float:
    try:
        dt = datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return 0.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    age_days = max((datetime.now(UTC) - dt).days, 0)
    return float(0.5 ** (age_days / half_life_days))


# Trust Agent(Evidence Confidence)도 동일한 최신성 계수를 쓴다 — 공개 별칭.
recency_factor = _recency_factor


Reranker = CrossEncoderReranker | OnnxCrossEncoderReranker | RemoteReranker
_reranker: Reranker | None = None
_reranker_key: tuple[object, ...] | None = None
_reranker_lock = threading.Lock()  # _reranker/_reranker_key 동시 갱신 보호 (key↔instance 불일치 방지)


def get_reranker(settings: Settings | None = None) -> Reranker:
    # backend/model/device/artifact/url 조합이 바뀌면 재생성 (torch↔onnx↔remote 전환·테스트 격리 보장)
    global _reranker, _reranker_key
    cfg = settings or get_settings()
    key = (
        cfg.reranker_backend,
        cfg.reranker_model,
        cfg.reranker_device,
        cfg.reranker_onnx_dir,
        cfg.reranker_onnx_file,
        cfg.reranker_service_url,
        cfg.reranker_timeout_s,  # 변경 시 httpx 타임아웃이 반영되도록 키에 포함
    )
    with _reranker_lock:  # 동시 초기화·설정 전환에서 key와 instance가 어긋난 채 반환되는 것을 막는다
        if _reranker is None or _reranker_key != key:
            if isinstance(_reranker, RemoteReranker):  # 교체 전 이전 remote 연결 정리
                _reranker.close()
            _reranker_key = key
            if cfg.reranker_backend == "onnx":  # #60: int8 경량화 백엔드(opt-in)
                _reranker = OnnxCrossEncoderReranker(cfg)
            elif cfg.reranker_backend == "remote":  # #72: 별도 리랭커 서비스
                _reranker = RemoteReranker(cfg)
            else:
                _reranker = CrossEncoderReranker(cfg)
        return _reranker


def reset_reranker() -> None:
    global _reranker, _reranker_key
    with _reranker_lock:
        if isinstance(_reranker, RemoteReranker):  # remote 연결 정리 후 리셋
            _reranker.close()
        _reranker = None
        _reranker_key = None
