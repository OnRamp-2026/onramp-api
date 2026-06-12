import json
from pathlib import Path

from app.rag.chunker import ControlDocChunker, MarkdownPage, SemanticChunker, child_chunk_to_index_record, write_jsonl


def test_chunker_builds_parent_and_child_chunks_with_context_prefix() -> None:
    markdown = """
# API Runbook

## Restart

Restart the pod when health checks fail.

```bash
kubectl get pod <pod-name>
kubectl describe pod <pod-name>
```

| 상태 | 의미 |
| --- | --- |
| Init:N/M | M개 중 N개 완료 |
"""
    page = MarkdownPage(page_id="123", page_title="API Runbook", markdown=markdown)
    parents, children = SemanticChunker(child_target_tokens=80, child_max_tokens=120).chunk(page)

    assert parents
    assert children
    assert children[0].parent_id == parents[0].parent_id
    assert children[0].page_id == "123"
    assert "문서: API Runbook" in children[0].embedding_text
    assert "도메인:" in children[0].embedding_text
    assert "경로: API Runbook" in children[0].embedding_text
    assert any("블록 유형:" in child.embedding_text for child in children)


def test_chunker_adds_compromise_metadata_for_runbook_content() -> None:
    markdown = """
# API 장애 대응 Runbook

## 원인 분석

CrashLoopBackOff가 발생하면 이벤트와 로그를 확인한다.

```bash
kubectl describe pod api-0
kubectl logs api-0 -c app
```

## 재발 방지

- readiness probe를 검증한다.
- 배포 전 smoke test를 추가한다.
"""
    page = MarkdownPage(page_id="incident", page_title="API 장애 대응 Runbook", markdown=markdown)
    parents, children = SemanticChunker(child_target_tokens=80, child_max_tokens=140).chunk(page)

    assert {parent.section_type for parent in parents} >= {"root_cause", "prevention"}
    assert any(child.domain == "장애대응" for child in children)
    assert any(child.section_type == "root_cause" for child in children)
    assert any(child.has_code for child in children)
    assert any("kubectl describe pod api-0" in child.keywords for child in children if child.keywords)
    assert any("섹션 유형: root_cause" in child.embedding_text for child in children)
    assert any("키워드:" in child.embedding_text for child in children)


def test_control_doc_chunker_preserves_decision_action_and_risk_boundaries() -> None:
    markdown = """
# 결제 정책 회의록

## 결정사항

- Qdrant collection은 문서 domain으로 필터링한다.
- 정책 변경은 다음 배치부터 적용한다.

## 액션아이템

- 담당자: 플랫폼팀
- 기한: 2026-06-03
- 상태: 진행 중

## 리스크

- 블로커: 회의록 문서가 길어지면 결정 맥락이 잘릴 수 있다.
"""
    page = MarkdownPage(page_id="meeting", page_title="결제 정책 회의록", markdown=markdown)
    parents, children = ControlDocChunker(child_target_tokens=80, child_max_tokens=140).chunk(page)

    assert {parent.section_type for parent in parents} >= {"decision", "action_item", "risk"}
    assert any(parent.domain == "기획서" for parent in parents)
    assert any(child.section_type == "action_item" for child in children)
    assert any("플랫폼팀" in child.keywords for child in children if child.keywords)
    assert any("2026-06-03" in child.keywords for child in children if child.keywords)


def test_chunker_keeps_context_before_code_block() -> None:
    markdown = """
# 운영 매뉴얼

## 검증

아래 명령으로 노드 상태를 확인한다.

```bash
kubectl get nodes
kubectl describe node worker-1
```
"""
    page = MarkdownPage(page_id="runbook", page_title="운영 매뉴얼", markdown=markdown)
    _, children = SemanticChunker(child_target_tokens=10, child_max_tokens=35).chunk(page)
    code_child = next(child for child in children if child.has_code)

    assert "아래 명령으로 노드 상태를 확인한다." in code_child.content
    assert code_child.code_languages == ["bash"]


def test_chunker_keeps_context_before_code_block_after_boundary_flush() -> None:
    markdown = """
# 운영 매뉴얼

이 문단은 앞 청크를 채우기 위한 긴 설명이다. 알림을 확인하고 대상 클러스터를 고른다.

아래 명령으로 노드 상태를 확인한다.

```bash
kubectl get nodes
kubectl describe node worker-1
```
"""
    page = MarkdownPage(page_id="flush", page_title="운영 매뉴얼", markdown=markdown)
    _, children = SemanticChunker(child_target_tokens=18, child_max_tokens=45).chunk(page)
    code_child = next(child for child in children if child.has_code)

    assert "아래 명령으로 노드 상태를 확인한다." in code_child.content


def test_chunker_preserves_code_block_without_layout_noise() -> None:
    markdown = """
# Debug

```
kubectl get pod <pod-name>
kubectl describe pod <pod-name>
```
"""
    page = MarkdownPage(page_id="174407", page_title="Debug", markdown=markdown)
    _, children = SemanticChunker(child_target_tokens=20, child_max_tokens=40).chunk(page)
    joined = "\n\n".join(child.content for child in children)

    assert "kubectl get pod <pod-name>" in joined
    assert "wide760" not in joined
    assert "breakoutMode" not in joined
    assert joined.count("```") >= 2


def test_chunker_overlap_preserves_line_breaks() -> None:
    markdown = """
# Overlap

첫 번째 문단은 청크 경계를 만들기 위한 설명이다.

```
kubectl describe pod <pod-name>
kubectl logs <pod-name> -c app
```

두 번째 문단은 이전 청크 일부가 겹쳐 들어와도 줄바꿈이 유지되어야 한다.
"""
    page = MarkdownPage(page_id="overlap", page_title="Overlap", markdown=markdown)
    _, children = SemanticChunker(child_target_tokens=25, child_max_tokens=50, overlap_tokens=20).chunk(page)

    assert len(children) > 1
    assert "```\nkubectl" in "\n\n".join(child.content for child in children)


def test_chunker_keeps_child_token_count_within_max_when_adding_overlap() -> None:
    markdown = """
# Overlap Budget

첫 번째 설명은 다음 청크와 겹쳐 들어갈 수 있는 문장이다.

두 번째 설명은 검색 단위가 될 본문이며 제한 토큰을 넘지 않아야 한다.

세 번째 설명은 이어지는 본문이며 이전 overlap 때문에 상한을 넘으면 안 된다.
"""
    page = MarkdownPage(page_id="budget", page_title="Overlap Budget", markdown=markdown)
    _, children = SemanticChunker(
        child_min_tokens=0,
        child_target_tokens=16,
        child_max_tokens=24,
        overlap_tokens=20,
    ).chunk(page)

    assert len(children) > 1
    assert all(child.token_count <= 24 for child in children)


def test_chunker_overlap_balances_code_fences() -> None:
    markdown = """
# Overlap

설명 문단.

```
line 1
line 2
line 3
line 4
line 5
line 6
```

다음 문단.
"""
    page = MarkdownPage(page_id="fence", page_title="Fence", markdown=markdown)
    _, children = SemanticChunker(child_target_tokens=15, child_max_tokens=25, overlap_tokens=6).chunk(page)

    assert len(children) > 1
    assert all(child.content.count("```") % 2 == 0 for child in children)


def test_chunker_repeats_table_header_when_splitting_large_table() -> None:
    rows = "\n".join(f"| row-{index} | value-{index} |" for index in range(20))
    markdown = f"""
# Table Doc

| 항목 | 값 |
| --- | --- |
{rows}
"""
    page = MarkdownPage(page_id="table", page_title="Table Doc", markdown=markdown)
    _, children = SemanticChunker(child_target_tokens=20, child_max_tokens=35).chunk(page)
    table_children = [child for child in children if "| 항목 | 값 |" in child.content]

    assert len(table_children) > 1
    assert all("| --- | --- |" in child.content for child in table_children)


def test_chunker_preserves_code_language_when_splitting_large_code_block() -> None:
    body = "\n".join(f"kubectl get pod pod-{index}" for index in range(12))
    markdown = f"""
# Large Code

```bash
{body}
```
"""
    page = MarkdownPage(page_id="large-code", page_title="Large Code", markdown=markdown)
    _, children = SemanticChunker(child_target_tokens=18, child_max_tokens=28).chunk(page)
    code_children = [child for child in children if child.has_code]

    assert len(code_children) > 1
    assert all(child.code_languages == ["bash"] for child in code_children)


def test_chunker_does_not_merge_short_parents_beyond_parent_max_tokens() -> None:
    markdown = """
# Parent Max

## 원인

짧은 원인 설명이다.

## 원인 추가

추가 원인 설명도 짧다.

## 원인 보강

보강 설명도 짧다.
"""
    page = MarkdownPage(page_id="parent-max", page_title="Parent Max", markdown=markdown)
    parents, _ = SemanticChunker(child_min_tokens=20, parent_max_tokens=14).chunk(page)

    assert len(parents) > 1
    assert all(parent.token_count <= 14 for parent in parents)


def test_chunker_passes_lineage_meta_to_child_chunks() -> None:
    # 버전 계보 메타 (#94) — MarkdownPage → ChildChunk 관통
    markdown = "# Content Negotiation\n\nApache negotiates the best representation."
    page = MarkdownPage(
        page_id="777",
        page_title="Content Negotiation [a78792-639072]",
        markdown=markdown,
        site="apache",
        product_version="2.2",
        doc_key="apache:content-negotiation",
        is_eol=True,
    )
    _, children = SemanticChunker().chunk(page)

    assert children
    assert all(child.site == "apache" for child in children)
    assert all(child.product_version == "2.2" for child in children)
    assert all(child.doc_key == "apache:content-negotiation" for child in children)
    assert all(child.is_eol is True for child in children)


def test_child_chunk_to_index_record_matches_batch_jsonl_contract() -> None:
    markdown = "# Doc\n\n```bash\nkubectl get pods\n```"
    page = MarkdownPage(
        page_id="doc",
        page_title="Doc",
        markdown=markdown,
        source_url="https://example.atlassian.net/wiki/spaces/OnRamp/pages/doc",
        space_key="OnRamp",
        last_modified="2026-06-01T00:00:00.000+0900",
    )
    _, children = SemanticChunker().chunk(page)

    record = child_chunk_to_index_record(children[0])

    assert record["chunk_id"] == children[0].chunk_id
    assert record["parent_id"] == children[0].parent_id
    assert record["embedding_text"]
    assert record["metadata"]["space_key"] == "OnRamp"
    assert record["metadata"]["heading_path"] == ["Doc"]
    assert record["metadata"]["has_code"] is True
    assert record["metadata"]["chunking_profile"] == ""
    assert record["metadata"]["tags"] == []


def test_write_jsonl_outputs_one_row_per_chunk(tmp_path: Path) -> None:
    markdown = "# Doc\n\nContent for chunking."
    page = MarkdownPage(page_id="doc", page_title="Doc", markdown=markdown)
    _, children = SemanticChunker().chunk(page)
    output = tmp_path / "chunks.jsonl"

    write_jsonl(output, [child_chunk_to_index_record(child) for child in children])

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(children)
    assert rows[0]["page_id"] == "doc"
    assert "metadata" in rows[0]
