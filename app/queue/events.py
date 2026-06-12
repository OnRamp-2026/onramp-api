from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel


class StreamEnvelope(BaseModel):
    event_id: str
    event_type: str
    schema_version: Literal["1.0"] = "1.0"
    payload: dict[str, Any]


def encode_envelope(envelope: StreamEnvelope) -> dict[str, str]:
    return {
        "event_id": envelope.event_id,
        "event_type": envelope.event_type,
        "schema_version": envelope.schema_version,
        "payload": json.dumps(envelope.payload, ensure_ascii=False, separators=(",", ":")),
    }
