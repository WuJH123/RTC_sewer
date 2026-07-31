from __future__ import annotations

from pathlib import Path
from typing import Any

from sewerrtc.contracts.prompt3a import write_json


def write_trajectory_schema(path: str | Path) -> Path:
    schema = {
        "truth_fields": ["node_depth", "node_head", "flooding_rate", "link_flow", "storage_volume", "actual_facility_setting"],
        "controller_visible_fields": ["sr0p15_sensor_observation", "gat_reconstruction", "past_state_history", "operational_forecast", "data_quality", "uncertainty", "ood"],
        "truth_to_controller_forbidden": True,
        "facility_fields": ["native_target", "anchor", "requested", "projected", "target", "actual", "previous_actual", "override_ttl", "release_flag", "rate_limit_status", "dwell_status", "interlock_status"],
    }
    return write_json(Path(path), schema)

