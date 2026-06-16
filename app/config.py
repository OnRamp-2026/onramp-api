import json
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    app_name: str = "OnRamp API"
    app_version: str = "0.1.0"
    debug: bool = False

    # Authentication
    auth_jwt_secret: SecretStr = SecretStr("")
    auth_jwt_issuer: str = ""
    auth_jwt_audience: str = "onramp-api"
    auth_session_ttl_seconds: int = Field(default=28800, ge=60)  # 세션 JWT 만료 (8h, 최소 60s)
    auth_dev_token_enabled: bool = False  # /auth/dev-token 게이트 (운영 false, dev/STT 테스트용)
    auth_default_tenant: str = "onramp"  # dev 토큰·미인증 fallback 테넌트
    auth_cookie_name: str = "onramp_session"  # 세션 쿠키 이름 (프론트는 credentials:include로 사용)
    auth_cookie_secure: bool = True  # HTTPS 전용 쿠키 (로컬 http 테스트 시 false)
    auth_cookie_samesite: Literal["lax", "strict", "none"] = "lax"  # OAuth top-level 복귀에 lax

    # OIDC RP — Slack "Sign in with Slack" (인증 서버 안 만듦, 클라이언트만)
    slack_client_id: str = ""
    slack_client_secret: SecretStr = SecretStr("")
    auth_base_url: str = ""  # 공개 base URL (redirect_uri 생성), 예: https://skala-cloud-team3-dev.skala-ai.com
    frontend_post_login_redirect: str = "/"  # 로그인 성공 후 프론트 복귀 경로

    # Slack 봇 (#146) — Events API + chat.postMessage (로그인 OIDC와 별개)
    slack_bot_enabled: bool = False  # kill-switch (false/미구성이면 /slack/events 404)
    slack_signing_secret: SecretStr = SecretStr("")  # 이벤트 요청 서명 검증
    slack_bot_token: SecretStr = SecretStr("")  # xoxb- (chat.postMessage)

    @model_validator(mode="after")
    def _check_auth_cookie(self) -> "Settings":
        # SameSite=None은 Secure=True가 아니면 브라우저가 세션 쿠키를 폐기한다.
        if self.auth_cookie_samesite == "none" and not self.auth_cookie_secure:
            raise ValueError("auth_cookie_samesite='none'이면 auth_cookie_secure=true가 필요합니다.")
        return self

    # LLM Provider: "openai" | "azure" | "self_hosted"
    llm_provider: str = ""

    # OpenAI
    openai_api_key: str = ""

    # Azure OpenAI
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""

    # Self-Hosted LLM (GPU 서버)
    self_hosted_llm_url: str = ""
    self_hosted_model_name: str = ""

    # 기본 모델
    default_model: str = ""

    # ── Observability (Langfuse, LLMOps) ──
    # kill-switch: false(기본)면 관측 전부 no-op — 키 없이도 앱이 기동한다.
    langfuse_enabled: bool = False
    langfuse_public_key: str = ""  # pk-lf-… (비밀 아님)
    langfuse_secret_key: SecretStr = SecretStr("")  # sk-lf-…
    langfuse_host: str = ""  # self-host URL 또는 https://cloud.langfuse.com

    @model_validator(mode="after")
    def _check_langfuse(self) -> "Settings":
        # fail-fast: 켜놓고 키/host가 비면 첫 요청에서 조용히 죽으므로 기동 단계에서 막는다.
        if self.langfuse_enabled:
            missing = [
                name
                for name, val in (
                    ("langfuse_public_key", self.langfuse_public_key),
                    ("langfuse_secret_key", self.langfuse_secret_key.get_secret_value()),
                    ("langfuse_host", self.langfuse_host),
                )
                if not val.strip()
            ]
            if missing:
                raise ValueError(f"langfuse_enabled=true면 다음 설정이 필요합니다: {', '.join(missing)}")
        return self

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "onramp"

    # OpenSearch / Hybrid Retrieval
    bm25_search_enabled: bool = False
    hybrid_search_enabled: bool = False
    opensearch_scheme: Literal["http", "https"] = "http"
    opensearch_host: str = "localhost"
    opensearch_port: int = 9200
    opensearch_username: str = ""
    opensearch_password: SecretStr = SecretStr("")
    opensearch_index: str = "onramp-chunks"
    opensearch_index_v1: str = "onramp-chunks-v1"
    opensearch_timeout_seconds: float = Field(default=10.0, gt=0)
    hybrid_rrf_k: int = Field(default=60, ge=1)
    hybrid_dense_top_k: int = Field(default=50, ge=1)
    hybrid_bm25_top_k: int = Field(default=50, ge=1)
    hybrid_final_top_k: int = Field(default=20, ge=1)

    # PostgreSQL
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/onramp"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_outbox_batch_size: int = Field(default=100, ge=1)
    redis_outbox_poll_interval_ms: int = Field(default=500, ge=1)
    redis_stream_block_ms: int = Field(default=5000, ge=1)
    redis_stream_read_count: int = Field(default=10, ge=1)
    redis_stream_reclaim_idle_ms: int = Field(default=300000, ge=1000)

    # STT event bus Redis (shared across tenant API instances).
    # Empty value falls back to redis_url for single-Redis/local environments.
    stt_redis_url: str = ""
    stt_consumer_group_suffix: str = ""

    @model_validator(mode="after")
    def _resolve_stt_redis(self) -> "Settings":
        normalized = self.stt_redis_url.strip()
        self.stt_redis_url = normalized or self.redis_url
        return self

    # STT internal API / report worker
    stt_service_base_url: str = "http://onramp-stt-api:8000"
    stt_service_token: SecretStr = SecretStr("")
    stt_result_timeout_seconds: float = Field(default=30.0, gt=0)
    report_worker_poll_interval_ms: int = Field(default=1000, ge=100)
    report_worker_processing_timeout_seconds: int = Field(default=300, ge=30)
    report_worker_max_retries: int = Field(default=3, ge=0)
    report_window_max_chars: int = Field(default=12000, ge=1000)
    report_window_overlap_chars: int = Field(default=500, ge=0)
    report_merge_batch_size: int = Field(default=4, ge=2)

    @model_validator(mode="after")
    def _check_report_window(self) -> "Settings":
        if self.report_window_overlap_chars >= self.report_window_max_chars:
            raise ValueError("report_window_overlap_chars must be smaller than report_window_max_chars")
        return self

    # Object Storage
    storage_bucket: str = "onramp-stt"
    storage_endpoint_url: str = ""
    storage_public_endpoint_url: str = ""
    storage_region: str = "ap-northeast-2"
    storage_access_key: SecretStr = SecretStr("")
    storage_secret_key: SecretStr = SecretStr("")
    storage_upload_expires_seconds: int = Field(default=900, ge=60, le=900)

    # Confluence
    confluence_base_url: str = ""
    confluence_api_token: str = ""
    confluence_user_email: str = ""
    confluence_space_key: str = "OnRamp"
    confluence_timezone: str = "Asia/Seoul"

    # RAG / Retrieval
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    self_hosted_embedding_url: str = ""  # P1: BGE-M3 (VesslAI GPU)
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cpu"  # P1: "cuda"
    # 리랭커 백엔드. "torch"(기본·현행) | "onnx"(#60 int8 경량) | "remote"(#72 별도 서비스 HTTP).
    # onnx 사용 시 scripts/build_reranker_onnx.py 산출물 디렉토리를 reranker_onnx_dir로 지정.
    reranker_backend: Literal["torch", "onnx", "remote"] = "torch"
    reranker_onnx_dir: str = ""
    reranker_onnx_file: str = "model_quantized.onnx"
    # #72 remote: 별도 리랭커 서비스(onramp-reranker) — 메모리 분리. 예: http://onramp-reranker:8080
    reranker_service_url: str = ""
    reranker_timeout_s: float = Field(default=10.0, gt=0)  # read 타임아웃(연결 후 응답 대기)
    # on-demand GPU(VESSL)가 꺼져 있을 때 매 요청을 오래 지연시키지 않도록 connect 타임아웃을 짧게 분리.
    reranker_connect_timeout_s: float = Field(default=2.0, gt=0)
    # 클라이언트 사이드 서킷브레이커(별도 서비스 X): 연속 실패 N회 → 쿨다운 동안 호출 스킵하고 즉시 vector 폴백.
    reranker_breaker_fail_threshold: int = Field(default=3, ge=1)
    reranker_breaker_cooldown_s: float = Field(default=30.0, gt=0)
    # #73 on-demand GPU(VESSL)는 스핀업마다 URL이 바뀐다. URL을 Redis 키에서 런타임 조회(rollout 없이 갱신).
    # 키가 없으면 reranker_service_url로 폴백. 둘 다 비면 remote는 매 요청 실패 → 서킷브레이커 → vector 폴백.
    reranker_url_redis_key: str = "reranker:service_url"
    reranker_url_cache_ttl_s: float = Field(default=30.0, ge=0)  # URL 재조회 주기(Redis 부하 완화)

    @model_validator(mode="after")
    def _check_reranker_remote(self) -> "Settings":
        # remote 백엔드: URL은 env(reranker_service_url) 또는 런타임 Redis(reranker_url_redis_key)에서 온다.
        # env가 비어 있어도 Redis가 공급할 수 있으므로 fail-fast하지 않는다(둘 다 비면 폴백으로 흡수).
        # 단, env URL이 지정됐다면 형식은 검증한다(오타로 조용히 폴백되는 것 방지).
        if self.reranker_backend == "remote":
            url = self.reranker_service_url.strip()
            if url:
                parsed = urlparse(url)
                if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                    raise ValueError(
                        "reranker_service_url은 http/https URL 형식이어야 함 (예: http://onramp-reranker:8080)"
                    )
        return self

    @model_validator(mode="after")
    def _check_reranker_onnx(self) -> "Settings":
        # fail-fast: onnx 백엔드면 모델 파일이 실제로 존재해야 기동을 통과시킨다.
        # (빈 경로/오타 경로면 첫 요청에서 vector로 조용히 폴백되므로 기동 단계에서 막는다.)
        if self.reranker_backend == "onnx":
            if not self.reranker_onnx_dir.strip():
                raise ValueError(
                    "reranker_backend='onnx'면 reranker_onnx_dir 필요 (build_reranker_onnx.py 산출물 경로)"
                )
            if not self.reranker_onnx_file.strip():
                raise ValueError("reranker_backend='onnx'면 reranker_onnx_file 필요")
            model_path = Path(self.reranker_onnx_dir, self.reranker_onnx_file)
            if not model_path.is_file():
                raise ValueError(
                    f"reranker_backend='onnx' 모델 파일 없음: {model_path} (scripts/build_reranker_onnx.py 먼저 실행)"
                )
        return self

    @model_validator(mode="after")
    def _check_domain_weights(self) -> "Settings":
        # 대표 도메인(primary)은 추가 도메인(secondary)보다 **엄격히 커야** 우선순위가 보장된다(같으면 변별 없음).
        if self.domain_primary_weight <= self.domain_secondary_weight:
            raise ValueError(
                f"domain_primary_weight({self.domain_primary_weight})는 "
                f"domain_secondary_weight({self.domain_secondary_weight})보다 커야 합니다"
            )
        return self

    retriever_top_k: int = 20  # Qdrant 후보 풀
    retriever_top_n: int = 5  # 리랭킹 후 최종
    snippet_max_chars: int = 500  # SourceDocument content_snippet 길이
    rerank_recency_weight: float = 0.1  # 최신성 가산값 (additive, rerank 순서 우선)
    rerank_recency_half_life_days: int = 180
    # 도메인 필터 모드 — soft 확정(#49 router-in-the-loop: 라우터 33%라 hard/hybrid 붕괴, soft 0.711)
    # soft: 무필터+가산 / hybrid: 저품질 무필터 확장 / hard: 필터만
    retriever_domain_filter_mode: Literal["hard", "hybrid", "soft"] = "soft"
    # 도메인 필터 보정 (임계값은 #49에서 골든셋으로 튜닝)
    # min_score: dense 유사도 임계값 → [0, 1].
    retriever_domain_min_score: float = Field(default=0.45, ge=0.0, le=1.0)
    # Soft 가산(질의 멀티도메인, #61): 문서 단일 domain이 query.domains[0]이면 primary, domains[1:]이면 secondary 가산.
    # primary > secondary 여야 함(대표 도메인 우선, _check_domain_weights로 강제). additive·logit 스케일 → 음수만 금지.
    domain_primary_weight: float = Field(default=0.1, ge=0.0)
    domain_secondary_weight: float = Field(default=0.05, ge=0.0)

    # ── Trust Agent (Evidence Confidence, P1) ──
    trust_max_retries: int = Field(default=1, ge=0)  # 재검색 최대 횟수 (무한루프 방지)
    # 재검색 트리거 τ (calibrate_answerability 보정값; top raw rerank<floor → 재검색)
    # #103 점수 분리: τ 비교가 부스트 합산 점수 → **raw 점수 [0,1]** 기준으로 바뀌면서 재보정.
    # (구 1.0018은 부스트가 섞인 스케일 — sigmoid [0,1]에서 1 초과는 부스트 합산으로만 가능했다.)
    # 114문항·재색인 코퍼스 보정: Youden 최대, precision 1.000 / recall 0.704.
    # eval_retrieval.ANSWERABILITY_FLOOR 와 동일 유지. 골든셋·코퍼스·리랭커 갱신 시 재보정.
    trust_rerank_floor: float = Field(default=0.8681, ge=0.0)
    # ── Trust 4축 재설계 (#108, 설계 4·5장) ──
    # 구 5축 가중치(trust_w_recency/owner/verification/duplication/sensitivity)와
    # trust_min_docs·중립 상수는 제거 — 죽은 축이 overall을 부풀리던 왜곡 제거(설계 1.3).
    trust_min_topics: int = Field(default=2, ge=1)  # coverage 기준 distinct 주제 수
    # strong-single-topic waiver (설계 4.4) — calibrate_answerability 산출 후보 채택:
    # unanswerable raw top1 최대 0.8639 바로 위 + answerable 격차 p75≈0.20 (보수적 시작).
    # 미발동 시 비효율(재검색 1회)로 퇴행할 뿐 오답이 아님 — 안전 마진.
    trust_tau_strong: float = Field(default=0.90, ge=0.0, le=1.0)
    trust_gap_strong: float = Field(default=0.20, ge=0.0)
    trust_adjacent_version_fit: float = Field(default=0.5, ge=0.0, le=1.0)  # match 모드 인접 부분점수
    # per-doc evidence = w_version_fit·fit + w_authority·tier (Answer 인용 우선순위·overall 성분)
    trust_w_version_fit: float = Field(default=0.8, ge=0.0, le=1.0)
    trust_w_authority: float = Field(default=0.2, ge=0.0, le=1.0)
    # overall = w_evidence_mean·mean(per_doc) + w_coverage·coverage + w_residual_dup·(1−잔여dup)
    # (sensitivity는 블렌드 제외 — 게이트 전용. authority 별도 항 없음 — per_doc 경유로 이중 계상 방지)
    trust_w_evidence_mean: float = Field(default=0.50, ge=0.0, le=1.0)
    trust_w_coverage: float = Field(default=0.30, ge=0.0, le=1.0)
    trust_w_residual_dup: float = Field(default=0.20, ge=0.0, le=1.0)
    trust_rewrite_model: str = "gpt-4o-mini"  # 재검색 사다리의 쿼리 재작성용 경량 모델
    # 같은 site·같은 product_version의 다른 주제 top raw 점수 차 < 이 값이면 충돌 의심(gate).
    # 구값 0.05는 부스트 스케일 기준 — raw sigmoid는 관련 문서가 0.95+에 포화 압축되어
    # 상호보완 문서끼리도 0.02~0.03 격차가 일상이라 0.05면 메인라인 질의가 상시 오탐된다
    # (실측: "초기화 컨테이너 디버그" vs "스테이트풀셋 디버깅" 격차 0.023 → CONFLICTING 오탐).
    # 차단성 게이트는 precision 우선 — 사실상 동점(0.005)만 충돌 후보로 본다.
    # 점수 휴리스틱 자체가 약한 근사이며 근본 해결은 P2 내용 기반 모순 감지(설계 5.2).
    trust_conflict_score_gap: float = Field(default=0.005, ge=0.0)
    # [MASKED_*] 마커 수가 이 값이면 sensitivity_risk=1.0 포화. ge=1 — 0/음수면 채점이 무력화됨.
    trust_sensitivity_masked_cap: int = Field(default=5, ge=1)

    # ── 버전 계보 (#94, Trust 재설계 선행 — 라벨 파생 payload + Qdrant facet 계보) ──
    # site별 EOL 버전 목록. env로는 JSON 문자열('{"apache":["2.2"]}')로 덮어쓴다.
    # k8s는 v1.33만 현행 지원 — v1.30 미만 전부 EOL.
    eol_versions: dict[str, list[str]] = Field(
        default_factory=lambda: {"apache": ["2.2"], "kubernetes": ["v1.18", "v1.25", "v1.29"]}
    )
    lineage_cache_ttl_seconds: int = Field(default=300, ge=0)  # 0이면 계보 캐시 비활성

    # ── 랭킹 부스트 (#103, 설계 7장 — per-doc 신뢰 신호의 랭킹 흡수) ──
    # 가산식(기존 recency·domain 부스트와 동일 계약) — 정렬 키(ranking)에만 더하고
    # raw 점수(τ 진단)는 오염시키지 않는다.
    rank_version_weight: float = Field(default=0.1, ge=0.0)  # version_fit 가산 계수
    rank_authority_weight: float = Field(default=0.05, ge=0.0)  # site tier 가산 계수
    # site 권위 등급 — 현 코퍼스는 전부 공식 문서라 변별력 없음(어댑터 자리, 설계 4.3).
    # 사내 Confluence 전환 시 space 등급·verified 라벨이 이 자리를 채운다.
    site_tier: dict[str, float] = Field(
        default_factory=lambda: {"apache": 1.0, "kubernetes": 1.0, "prometheus": 1.0, "datadog": 1.0}
    )
    site_tier_default: float = Field(default=1.0, ge=0.0, le=1.0)  # 미등록 site(내부 문서 등) 중립
    # 다버전 site 목록 — 단일 계보로 보이면 doc_key 정규화 실패 의심 → 보수 캡 적용 대상
    multi_version_sites: list[str] = Field(default_factory=lambda: ["apache", "kubernetes"])
    trust_eol_cap: float = Field(default=0.3, ge=0.0, le=1.0)  # EOL 버전 version_fit 상한
    trust_single_lineage_cap: float = Field(default=0.7, ge=0.0, le=1.0)  # 다버전 site 단일계보 보수 캡

    @field_validator("eol_versions", "site_tier", "multi_version_sites", mode="before")
    @classmethod
    def _parse_json_env(cls, v: object) -> object:
        # env var는 문자열로 들어오므로 JSON 파싱 (dict/list 기본값·직접 주입은 그대로 통과)
        return json.loads(v) if isinstance(v, str) else v

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()
