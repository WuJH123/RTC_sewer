from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from sewerrtc.contracts.prompt3a import INP_PATH
from sewerrtc.io.swmm_mutation import mutate_inp_for_event


def duration_from_event_id(event_id: str) -> int:
    match = re.search(r"_D(\d+)_", str(event_id))
    return int(match.group(1)) if match else 0


def build_event_inp_from_plan(row: dict[str, Any], out_dir: str | Path) -> dict[str, Any]:
    out_dir = Path(out_dir)
    event_id = str(row.get("event_id", ""))
    policy_id = str(row.get("policy_id", ""))
    rainfall_path = Path(str(row.get("rainfall_path", "")))
    tail_min = int(float(row.get("tail_min", 180) or 180))
    duration_min = duration_from_event_id(event_id)
    simulation_duration_min = duration_min + tail_min
    strip_controls = policy_id in {"no_control", "executable_passive"}
    event_inp = out_dir / "event_inp" / event_id / f"{event_id}__{policy_id}.inp"
    mutate_inp_for_event(
        INP_PATH,
        rainfall_path,
        event_inp,
        simulation_duration_min=simulation_duration_min,
        strip_controls=strip_controls,
    )
    return {
        "event_inp": str(event_inp),
        "event_id": event_id,
        "policy_id": policy_id,
        "duration_min": duration_min,
        "tail_min": tail_min,
        "simulation_duration_min": simulation_duration_min,
        "strip_controls": strip_controls,
    }
