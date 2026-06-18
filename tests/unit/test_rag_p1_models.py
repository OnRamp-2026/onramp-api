from typing import cast

from sqlalchemy import Table, inspect

from app.db.models import ChatLog, ChunkRegistry, IndexRun, SourceDocument


def test_source_document_uses_tenant_page_primary_key() -> None:
    table = cast(Table, inspect(SourceDocument).local_table)

    assert [column.name for column in table.primary_key.columns] == ["tenant_id", "source", "page_id"]
    assert "tenant_id" in table.c
    assert "source" in table.c
    assert "raw_html_hash" in table.c
    assert "cleaned_markdown_hash" in table.c


def test_chunk_registry_tracks_qdrant_point_and_run() -> None:
    table = cast(Table, inspect(ChunkRegistry).local_table)
    indexes = {str(index.name): tuple(column.name for column in index.columns) for index in table.indexes}
    fk_targets = {
        tuple(element.target_fullname for element in constraint.elements)
        for constraint in table.foreign_key_constraints
    }

    assert indexes["ix_chunk_registry_point_id"] == ("point_id",)
    assert indexes["ix_chunk_registry_run_id"] == ("run_id",)
    assert (
        "source_document.tenant_id",
        "source_document.source",
        "source_document.page_id",
    ) in fk_targets
    assert ("index_run.run_id",) in fk_targets


def test_index_run_has_failure_and_stale_cleanup_counters() -> None:
    table = cast(Table, inspect(IndexRun).local_table)
    indexes = {str(index.name): tuple(column.name for column in index.columns) for index in table.indexes}

    assert "pages_failed" in table.c
    assert "pages_discovered" in table.c
    assert "pages_processed" in table.c
    assert "pages_skipped" in table.c
    assert "trigger" in table.c
    assert "stage" in table.c
    assert "chunks_deleted" in table.c
    assert indexes["ix_index_run_tenant_status"] == ("tenant_id", "status", "created_at")


def test_chat_log_is_time_series_and_source_snapshot_ready() -> None:
    table = cast(Table, inspect(ChatLog).local_table)
    indexes = {str(index.name): tuple(column.name for column in index.columns) for index in table.indexes}

    assert "created_at" in table.c
    assert "sources" in table.c
    assert indexes["ix_chat_log_tenant_created"] == ("tenant_id", "created_at")
    assert indexes["ix_chat_log_domain"] == ("tenant_id", "domain")
