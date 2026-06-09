from functools import lru_cache
from typing import Literal

from pydantic import Field
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
    retriever_top_k: int = 20  # Qdrant 후보 풀
    retriever_top_n: int = 5  # 리랭킹 후 최종
    classifier_model: str = "gpt-4o-mini"
    snippet_max_chars: int = 500  # SourceDocument content_snippet 길이
    rerank_recency_weight: float = 0.1  # 최신성 가산값 (additive, rerank 순서 우선)
    rerank_recency_half_life_days: int = 180
    # 도메인 필터 모드 (#49 골든셋 A/B로 확정) — hard: 필터만 / hybrid: 저품질 무필터 확장 / soft: 무필터+가산
    retriever_domain_filter_mode: Literal["hard", "hybrid", "soft"] = "hybrid"
    # 도메인 필터 보정 (임계값은 #49에서 골든셋으로 튜닝)
    # min_score: dense 유사도 임계값 → [0, 1]. match_weight: rerank 가산값(additive, logit 스케일) → 음수만 금지.
    retriever_domain_min_score: float = Field(default=0.45, ge=0.0, le=1.0)
    retriever_domain_match_weight: float = Field(default=0.1, ge=0.0)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()
