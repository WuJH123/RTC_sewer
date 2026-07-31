from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FallbackAction:
    facility_id: str
    requested_setting: float | str
    projected_setting: float | str
    expected_actual_setting: float | str
    reason: str
    constraint_reason: str = ""


def selection_input_hash(payload: dict[str, Any]) -> str:
    serializable = _json_safe(payload)
    return hashlib.sha256(json.dumps(serializable, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def serialize_actions(actions: list[FallbackAction]) -> list[dict[str, Any]]:
    return [asdict(action) for action in actions]


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value
