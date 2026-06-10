from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    app_name: str = "OnRamp API"
    app_version: str = "0.1.0"
    debug: bool = False

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

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "onramp"

    # PostgreSQL
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/onramp"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

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
    # #60: 리랭커 백엔드. "torch"(기본·현행) | "onnx"(int8 경량화, CPU 파드용).
    # onnx 사용 시 scripts/build_reranker_onnx.py 산출물 디렉토리를 reranker_onnx_dir로 지정.
    reranker_backend: Literal["torch", "onnx"] = "torch"
    reranker_onnx_dir: str = ""
    reranker_onnx_file: str = "model_quantized.onnx"

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

    retriever_top_k: int = 20  # Qdrant 후보 풀
    retriever_top_n: int = 5  # 리랭킹 후 최종
    classifier_model: str = "gpt-4o-mini"
    snippet_max_chars: int = 500  # SourceDocument content_snippet 길이
    rerank_recency_weight: float = 0.1  # 최신성 가산값 (additive, rerank 순서 우선)
    rerank_recency_half_life_days: int = 180
    # 도메인 필터 모드 — soft 확정(#49 router-in-the-loop: 라우터 33%라 hard/hybrid 붕괴, soft 0.711)
    # soft: 무필터+가산 / hybrid: 저품질 무필터 확장 / hard: 필터만
    retriever_domain_filter_mode: Literal["hard", "hybrid", "soft"] = "soft"
    # 도메인 필터 보정 (임계값은 #49에서 골든셋으로 튜닝)
    # min_score: dense 유사도 임계값 → [0, 1]. match_weight: rerank 가산값(additive, logit 스케일) → 음수만 금지.
    retriever_domain_min_score: float = Field(default=0.45, ge=0.0, le=1.0)
    retriever_domain_match_weight: float = Field(default=0.1, ge=0.0)

    # ── Trust Agent (Evidence Confidence, P1) ──
    trust_max_retries: int = Field(default=1, ge=0)  # 재검색 최대 횟수 (무한루프 방지)
    # 재검색 트리거 τ (#A calibrate_answerability 보정값; top rerank<floor → 재검색)
    trust_rerank_floor: float = Field(default=0.288, ge=0.0)
    trust_min_docs: int = Field(default=1, ge=0)
    # Evidence Confidence 5축 가중치 (기본 합 1.0).
    #   env로 덮어써 합이 1.0이 아니어도 score_trust()가 wsum으로 나눠 자동 정규화하므로 동작은 정상이다.
    #   단, 가중치 절댓값이 아니라 "상대 비율"로 해석된다는 점에 유의.
    trust_w_recency: float = Field(default=0.30, ge=0.0, le=1.0)
    trust_w_owner: float = Field(default=0.10, ge=0.0, le=1.0)
    trust_w_verification: float = Field(default=0.10, ge=0.0, le=1.0)
    trust_w_duplication: float = Field(default=0.20, ge=0.0, le=1.0)
    trust_w_sensitivity: float = Field(default=0.30, ge=0.0, le=1.0)
    # owner_trust / verification_label: 색인 payload에 소스 없음 → 중립 (track-B 수집 의존성)
    trust_owner_neutral: float = Field(default=1.0, ge=0.0, le=1.0)
    trust_verification_neutral: float = Field(default=1.0, ge=0.0, le=1.0)
    # 서로 다른 page의 top rerank 점수 차 < 이 값이면 충돌 의심(gate)
    trust_conflict_score_gap: float = Field(default=0.05, ge=0.0)
    # [MASKED_*] 마커 수가 이 값이면 sensitivity_risk=1.0 포화. ge=1 — 0/음수면 채점이 무력화됨.
    trust_sensitivity_masked_cap: int = Field(default=5, ge=1)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()
