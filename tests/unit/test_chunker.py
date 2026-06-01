import json
from pathlib import Path

from app.rag.chunker import MarkdownPage, SemanticChunker, child_chunk_to_index_record, write_jsonl


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
