"""Opt-in Agentic Retriever tool loop (#237)."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from qdrant_client.models import ScoredPoint

from app.agents.retriever.tools import SearchToolContext, execute_search_tool
from app.rag.rrf import RankedItem, reciprocal_rank_fusion
from app.services.llm_selector import ToolCall, call_llm_with_tools

logger = logging.getLogger(__name__)

SEARCH_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_dense",
            "description": "개념, 절차, 배경처럼 의미 유사성이 중요한 문서를 검색합니다.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "검색할 구체적인 질의"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_bm25",
            "description": "버전, 명령어, 에러 코드, 고유 명칭처럼 정확한 키워드가 중요한 문서를 검색합니다.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "검색할 구체적인 질의"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]

_SYSTEM_PROMPT = """당신은 사내 지식 검색 Retriever입니다.
질문에 답할 근거를 찾기 위해 search_dense와 search_bm25 중 하나를 선택해 호출하세요.
- 개념, 절차, 배경은 search_dense가 유리합니다.
- 버전, 명령어, 에러 코드, 고유 명칭은 search_bm25가 유리합니다.
- 첫 응답에서는 반드시 가장 적합한 도구 하나만 호출하세요.
- 검색 결과가 질문에 답하기 충분하면 도구 호출 없이 종료하세요.
- 결과가 부족할 때만 다음 응답에서 다른 도구 하나를 호출하거나 검색어를 구체화하세요.
- 한 응답에서 두 도구를 동시에 호출하지 마세요. 전체 검색 호출은 최대 2회입니다.
- tenant와 보안 필터는 서버가 적용합니다. query 외 인자를 만들지 마세요.
충분한 근거를 찾았으면 도구 호출을 멈추세요."""

_MAX_POLICY_ITERATIONS = 2
_MAX_POLICY_TOOL_CALLS = 2


@dataclass(frozen=True)
class AgenticSearchResult:
    hits: list[ScoredPoint]
    metadata: dict[str, Any]


class AgenticRetrievalFallbackError(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip()


def _assistant_message(content: str, calls: list[ToolCall]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": content or None,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)},
            }
            for call in calls
        ],
    }


def _tool_summary(hits: list[ScoredPoint], snippet_chars: int) -> str:
    summary = []
    for hit in hits[:5]:
        payload = hit.payload or {}
        summary.append(
            {
                "chunk_id": str(payload.get("chunk_id") or hit.id),
                "title": str(payload.get("page_title") or ""),
                "score": float(hit.score),
                "snippet": str(payload.get("content") or "")[:snippet_chars],
            }
        )
    return json.dumps({"count": len(hits), "hits": summary}, ensure_ascii=False)


def _fuse_rankings(
    rankings: list[tuple[str, list[ScoredPoint]]],
    *,
    context: SearchToolContext,
) -> list[ScoredPoint]:
    if len(rankings) == 1:
        return rankings[0][1]
    ranked_lists = []
    for source, hits in rankings:
        ranked_lists.append(
            (
                source,
                [
                    RankedItem(
                        id=str((hit.payload or {}).get("chunk_id") or hit.id),
                        score=float(hit.score),
                        payload=hit.payload or {},
                    )
                    for hit in hits
                ],
            )
        )
    fused = reciprocal_rank_fusion(
        ranked_lists,
        k=context.settings.hybrid_rrf_k,
        limit=context.top_k,
    )
    return [
        ScoredPoint(
            id=item.id,
            version=0,
            score=item.score,
            payload={**item.payload, "_agentic_scores": item.source_scores},
        )
        for item in fused
    ]


async def run_agentic_search(
    query: str,
    *,
    model: str,
    context: SearchToolContext,
) -> AgenticSearchResult:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    rankings: list[tuple[str, list[ScoredPoint]]] = []
    seen: set[tuple[str, str, str]] = set()
    calls_metadata: list[dict[str, Any]] = []
    duplicate_calls = 0
    policy_skipped_calls = 0
    total_calls = 0
    filter_key = repr((context.domain, context.filters))
    max_iterations = min(context.settings.retriever_agentic_max_iterations, _MAX_POLICY_ITERATIONS)
    max_tool_calls = min(context.settings.retriever_agentic_max_tool_calls, _MAX_POLICY_TOOL_CALLS)

    for iteration in range(max_iterations):
        response = await call_llm_with_tools(
            messages,
            SEARCH_TOOL_SCHEMAS,
            model=model,
            settings=context.settings,
        )
        messages.append(_assistant_message(response.content, response.tool_calls))
        if not response.tool_calls:
            break

        selected_call = response.tool_calls[0]
        for skipped_call in response.tool_calls[1:]:
            policy_skipped_calls += 1
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": skipped_call.id,
                    "content": json.dumps({"skipped": "one_tool_per_iteration"}, ensure_ascii=False),
                }
            )

        for call in [selected_call]:
            if total_calls >= max_tool_calls:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps({"skipped": "tool_call_limit"}, ensure_ascii=False),
                    }
                )
                continue
            raw_query = call.arguments.get("query")
            normalized = _normalize_query(raw_query) if isinstance(raw_query, str) else ""
            if call.name not in {"search_dense", "search_bm25"} or not normalized:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps({"error": "invalid_tool_call"}, ensure_ascii=False),
                    }
                )
                continue
            dedupe_key = (call.name, normalized.casefold(), filter_key)
            if dedupe_key in seen:
                duplicate_calls += 1
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps({"skipped": "duplicate_call"}, ensure_ascii=False),
                    }
                )
                continue
            seen.add(dedupe_key)
            total_calls += 1
            try:
                hits = await execute_search_tool(call.name, normalized, context=context)
            except Exception as exc:
                logger.warning("Agentic 검색 도구 실패: %s", call.name, exc_info=True)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps({"error": type(exc).__name__}, ensure_ascii=False),
                    }
                )
                continue
            calls_metadata.append(
                {"iteration": iteration + 1, "tool": call.name, "query": normalized, "hit_count": len(hits)}
            )
            if hits:
                rankings.append((f"{call.name}:{len(rankings)}", hits))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": _tool_summary(hits, context.settings.retriever_agentic_tool_snippet_chars),
                }
            )

    if not rankings:
        raise AgenticRetrievalFallbackError("no_search_results")

    return AgenticSearchResult(
        hits=_fuse_rankings(rankings, context=context),
        metadata={
            "calls": calls_metadata,
            "tool_call_count": total_calls,
            "duplicate_calls": duplicate_calls,
            "policy_skipped_calls": policy_skipped_calls,
            "ranking_list_count": len(rankings),
            "rrf_applied": len(rankings) > 1,
        },
    )
