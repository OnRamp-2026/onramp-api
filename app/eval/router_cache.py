"""라우터 예측 캐시 — A/B를 같은 예측으로 공정하게 돌리기 위한 영속화.

목적: 라우터 LLM은 비결정적이므로, 한 번 생성한 예측을 파일에 저장해 A/B(평가)가
**동일한 예측**을 재사용하게 한다. 캐시는 정답이 아니다 — 모델이 실제로 예측한 결과다
(정답은 골든셋의 사람 검수 `router_domains`).

stale 판정 키(이 중 하나라도 바뀌면 그 qid는 재예측):
    qid + query_sha + requested_model + effective_provider + llm_provider
        + default_model + prompt_sha + schema_version
commit_sha·created_at는 **재현 메타로만** 저장하고 stale 키로 쓰지 않는다.
query 평문은 저장하지 않는다(stale 키는 sha로) — 평문은 사람 검수표에만 둔다.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from app.agents.router.schema import SCHEMA_VERSION
from app.config import Settings
from app.services.llm_selector import resolve_provider

# 오프라인 평가 전용 캐시. **운영 런타임(route_node)은 읽지도 쓰지도 않는다** — 평가 스크립트
# 수동 실행 시에만 생성된다. data/ 가 아니라 .cache/ 아래 둬서 "운영 데이터 아님"을 경로로도 분명히 한다.
CACHE_DIR = Path(".cache/onramp-eval")
DEFAULT_CACHE_PATH = CACHE_DIR / "router_predictions.jsonl"


def sha12(text: str) -> str:
    """sha256 앞 12자리 (질의·프롬프트 식별용)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def prompt_sha() -> str:
    """현재 운영 라우터 시스템 프롬프트의 sha (프롬프트 변경 시 캐시 무효화)."""
    from app.agents.router.prompts import ROUTER_SYSTEM_PROMPT

    return sha12(ROUTER_SYSTEM_PROMPT)


def git_commit_sha() -> str:
    """현재 커밋 SHA (재현 메타). 실패 시 빈 문자열."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5)
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return ""


@dataclass(frozen=True)
class CacheMeta:
    """캐시 stale 판정에 쓰는 모델·계약 메타 (qid·query_sha와 함께 키 구성)."""

    requested_model: str
    effective_provider: str
    llm_provider: str
    default_model: str
    prompt_sha: str
    schema_version: str


def current_meta(requested_model: str, settings: Settings) -> CacheMeta:
    """현재 설정으로 캐시 메타를 만든다. effective_provider는 운영과 동일 로직(resolve_provider) 재사용."""
    return CacheMeta(
        requested_model=requested_model,
        effective_provider=resolve_provider(requested_model, settings),
        llm_provider=(settings.llm_provider or "").strip().lower(),
        default_model=(settings.default_model or "").strip(),
        prompt_sha=prompt_sha(),
        schema_version=SCHEMA_VERSION,
    )


@dataclass(frozen=True)
class PredictionRecord:
    """예측 1건.

    - ``raw_predicted_domains``: confidence 게이팅 **전**(라우터가 실제 분류한 도메인).
      라우터 분류 능력·confidence calibration 측정의 근거.
    - ``predicted_domains``: 게이팅 **후**(검색이 실제 쓰는 값). 운영 결과 측정의 근거.
    둘을 함께 저장해야 "분류가 틀린 것"과 "도메인은 맞지만 저신뢰로 비워진 것"을 구분할 수 있다.
    """

    qid: str
    query_sha: str
    raw_predicted_domains: list[str]
    predicted_domains: list[str]
    confidence: float | None
    use_case: str
    parse_ok: bool
    fallback_reason: str | None
    low_conf_empty: bool
    # ── 메타 (stale 키 구성) ──
    requested_model: str
    effective_provider: str
    llm_provider: str
    default_model: str
    prompt_sha: str
    schema_version: str
    # ── 재현 메타 (stale 키 아님) ──
    commit_sha: str
    created_at: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    def freshness_key(self) -> tuple:
        return _freshness_key(self.qid, self.query_sha, _meta_of(asdict(self)))


def _meta_of(d: dict) -> CacheMeta:
    return CacheMeta(
        requested_model=d["requested_model"],
        effective_provider=d["effective_provider"],
        llm_provider=d["llm_provider"],
        default_model=d["default_model"],
        prompt_sha=d["prompt_sha"],
        schema_version=d["schema_version"],
    )


def _freshness_key(qid: str, query_sha: str, meta: CacheMeta) -> tuple:
    return (
        qid,
        query_sha,
        meta.requested_model,
        meta.effective_provider,
        meta.llm_provider,
        meta.default_model,
        meta.prompt_sha,
        meta.schema_version,
    )


def is_fresh(record: dict, *, query_sha: str, meta: CacheMeta) -> bool:
    """캐시 레코드가 현재 질의·메타와 일치(신선)하는가."""
    try:
        return _freshness_key(record["qid"], record["query_sha"], _meta_of(record)) == _freshness_key(
            record["qid"], query_sha, meta
        )
    except KeyError:
        return False  # 필드 누락된 구형 레코드 → stale 취급


def load_cache(path: Path | str = DEFAULT_CACHE_PATH) -> dict[str, dict]:
    """캐시 JSONL을 qid→레코드 dict로 로드. 없으면 빈 dict. 같은 qid는 마지막 줄이 승리."""
    path = Path(path)
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        rec = json.loads(line)
        out[rec["qid"]] = rec
    return out


def write_cache(records: list[PredictionRecord], path: Path | str = DEFAULT_CACHE_PATH) -> None:
    """레코드를 JSONL로 **원자적**으로 쓴다(.tmp 기록 후 os.replace) — 부분 쓰기 방지."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(r.to_json() + "\n" for r in records), encoding="utf-8")
    os.replace(tmp, path)
