from typing import cast

from sqlalchemy import Table, inspect

from app.db.models import Conversation, Message


def test_conversation_scoped_to_tenant_user_and_sorted_by_updated() -> None:
    table = cast(Table, inspect(Conversation).local_table)
    indexes = {str(index.name): tuple(column.name for column in index.columns) for index in table.indexes}

    assert [column.name for column in table.primary_key.columns] == ["conversation_id"]
    assert "tenant_id" in table.c
    assert "user_id" in table.c
    assert "title" in table.c
    assert indexes["ix_conversation_tenant_user_updated"] == ("tenant_id", "user_id", "updated_at")


def test_message_cascades_from_conversation_and_keeps_answer_snapshot() -> None:
    table = cast(Table, inspect(Message).local_table)
    indexes = {str(index.name): tuple(column.name for column in index.columns) for index in table.indexes}
    fk = next(iter(table.foreign_key_constraints))

    assert [column.name for column in table.primary_key.columns] == ["message_id"]
    assert {"role", "content", "answer", "sources", "domain", "model_used"} <= set(table.c.keys())
    assert fk.elements[0].target_fullname == "conversation.conversation_id"
    assert fk.ondelete == "CASCADE"
    assert indexes["ix_message_conversation_created"] == ("conversation_id", "created_at")
