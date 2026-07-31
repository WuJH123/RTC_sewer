from __future__ import annotations

import hashlib
import json
from typing import Any


REQUIRED_CONTROLLER_MEMORY_FIELDS = (
    "native_target",
    "anchor",
    "requested_action",
    "projected_action",
    "target_action",
    "actual_action",
    "override_ttl",
    "fallback_mode",
    "pump_on_duration",
    "pump_off_duration",
    "add350_speed_actual",
    "binary_pump_states",
    "forecast_issue_id",
    "last_measurement_time",
    "decision_time",
    "continuation_policy_id",
)


def controller_memory_hash(memory: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(memory, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def validate_controller_memory(memory: dict[str, Any]) -> list[str]:
    return [field for field in REQUIRED_CONTROLLER_MEMORY_FIELDS if field not in memory]

