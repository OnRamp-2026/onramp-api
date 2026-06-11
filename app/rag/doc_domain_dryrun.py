"""문서 도메인 분류 dry-run — 페이지를 분류해 검수용 JSONL로 영속 저장 (Step 2, #49).

캐시/해시/DB 없음: JSONL 자체가 결과 보존본이다. 같은 (page_id, page_version, model, prompt_version,
ontology_version) 결과가 이미 있으면 LLM을 재호출하지 않고 재사용한다(page_version은 IngestService 제공값).
rule_fallback은 review_status=pending으로 남고 approved로 자동 승격하지 않는다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.rag.doc_domain_classifier import (
    DOC_CLASSIFIER_PROMPT_VERSION,
    ClassificationResult,
    DocumentDomainClassifier,
)
from app.rag.domains import ONTOLOGY_VERSION

ReuseKey = tuple[str, object, str, str, str]  # (page_id, page_version, model, prompt_version, ontology_version)


@dataclass(frozen=True)
class DryRunPage:
    page_id: str
    version: int | None
    title: str
    masked_markdown: str  # IngestService가 마스킹한 본문 (이 모듈은 마스킹하지 않음)


@dataclass
class DryRunStats:
    classified: int = 0  # LLM 호출로 분류 성공
    reused: int = 0  # 기존 JSONL 결과 재사용
    fallback: int = 0  # LLM 실패 → rule 폴백
    pages: int = 0

    def as_line(self) -> str:
        return (
            f"pages={self.pages}  classified(LLM)={self.classified}  "
            f"reused={self.reused}  rule_fallback={self.fallback}"
        )


def is_reusable(record: dict) -> bool:
    """재사용 가능한 결과인지 — LLM 성공이거나 사람이 승인한 것만. rule_fallback(pending)은 재호출 대상.

    일시적 LLM 장애로 폴백된 문서가 다음 정상 실행에서도 LLM을 못 타는 것을 막는다.
    """
    return record.get("classification_source") == "llm" or record.get("review_status") == "approved"


def record_reuse_key(record: dict) -> ReuseKey:
    return (
        record["page_id"],
        record["page_version"],
        record["classifier_model"],
        record["prompt_version"],
        record["ontology_version"],
    )


def build_record(page: DryRunPage, result: ClassificationResult, *, classifier_model: str) -> dict:
    return {
        "page_id": page.page_id,
        "page_version": page.version,
        "classifier_model": classifier_model,
        "prompt_version": DOC_CLASSIFIER_PROMPT_VERSION,
        "ontology_version": ONTOLOGY_VERSION,
        "title": page.title,
        "primary_domain": result.classification.primary_domain,
        "domains": [evidence.model_dump() for evidence in result.classification.domains],
        "adopted_domains": result.adopted_domains,
        "classification_source": result.source,
        "review_status": "pending",  # rule_fallback도 pending — approved 자동 승격 금지
    }


def _page_identity(record: dict) -> str:
    """검수용 최신 스냅샷의 페이지 식별자 = page_id. (버전/모델/프롬프트/ontology 무관 → 페이지당 1줄)

    과거 결과 이력이 필요하면 별도 이력 파일로 분리한다(이 파일은 '최신 1줄/페이지'를 보장).
    """
    return str(record["page_id"])


def load_existing(path: str | Path) -> dict[ReuseKey, dict]:
    path = Path(path)
    if not path.exists():
        return {}
    existing: dict[ReuseKey, dict] = {}
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                # 사람이 검수 중 편집하다 깨질 수 있으므로 파일·줄 위치를 알려준다
                raise ValueError(f"{path}:{lineno} JSONL 파싱 실패 (검수 중 손상 의심): {exc}") from exc
            try:
                key = record_reuse_key(record)
            except (KeyError, TypeError) as exc:
                # 필수 필드 누락 또는 dict가 아닌 JSON(리스트/숫자 등) — 위치와 함께 알린다
                raise ValueError(f"{path}:{lineno} 필수 필드 누락/구조 손상: {exc}") from exc
            existing[key] = record
    return existing


async def run_dry_run(
    pages: list[DryRunPage],
    classifier: DocumentDomainClassifier,
    *,
    existing: dict[ReuseKey, dict] | None = None,
    force: bool = False,
) -> tuple[list[dict], DryRunStats]:
    """페이지별 1회 분류(또는 기존 결과 재사용). 색인/Qdrant에는 연결하지 않는다."""
    existing = existing or {}
    classifier_model = classifier.settings.classifier_model
    records: list[dict] = []
    stats = DryRunStats(pages=len(pages))
    for page in pages:
        key: ReuseKey = (
            page.page_id,
            page.version,
            classifier_model,
            DOC_CLASSIFIER_PROMPT_VERSION,
            ONTOLOGY_VERSION,
        )
        if not force and key in existing and is_reusable(existing[key]):
            records.append(existing[key])
            stats.reused += 1
            continue
        result = await classifier.classify_page(page_title=page.title, content=page.masked_markdown)
        records.append(build_record(page, result, classifier_model=classifier_model))
        if result.source == "rule_fallback":
            stats.fallback += 1
        else:
            stats.classified += 1
    return records, stats


def merge_records(existing: dict[ReuseKey, dict], new_records: list[dict]) -> list[dict]:
    """기존 전체 결과에 이번 실행 결과를 병합. 이번 대상 밖 페이지(검수본)는 보존하되,
    이번에 분류한 page_id의 **기존 레코드(버전/프롬프트/모델 무관)는 모두 제거**한다 → page_id당 1줄 보장.
    """
    # 이번 실행 내 같은 page_id 중복도 마지막 것만 남긴다(page_id당 1줄 보장)
    new_by_page: dict[str, dict] = {}
    for record in new_records:
        new_by_page[_page_identity(record)] = record
    kept = [record for record in existing.values() if _page_identity(record) not in new_by_page]
    return kept + list(new_by_page.values())


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
