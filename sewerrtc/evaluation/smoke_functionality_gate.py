from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def event_return_period(event_id: object) -> str:
    match = re.match(r"^(T\d+)", str(event_id))
    return match.group(1) if match else ""


def parse_sequence(value: object) -> np.ndarray | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        arr = np.asarray(ast.literal_eval(text), dtype=float)
    except Exception:
        return None
    return arr if arr.size else None


def _action_ids(control_table: pd.DataFrame) -> list[str]:
    for column in ("actuator_id", "link_id", "id"):
        if column in control_table:
            return control_table[column].astype(str).tolist()
    return []


def _binary_pump_ids(control_table: pd.DataFrame, configured: Iterable[str]) -> list[str]:
    ids = _action_ids(control_table)
    configured_set = {str(x) for x in configured}
    if configured_set:
        return [aid for aid in ids if aid in configured_set]
    if "link_type" not in control_table:
        return []
    return control_table.loc[control_table["link_type"].astype(str).str.lower().eq("pump"), control_table.columns[0]].astype(str).tolist()


def _storage_pairs(control_table: pd.DataFrame) -> dict[str, tuple[list[int], list[int]]]:
    ids = _action_ids(control_table)
    if not ids:
        return {}
    role_col = "asset_role" if "asset_role" in control_table else ("control_semantics" if "control_semantics" in control_table else "")
    storage_col = "storage_node" if "storage_node" in control_table else ("storage_association" if "storage_association" in control_table else "")
    if not role_col or not storage_col:
        return {}
    pairs: dict[str, tuple[list[int], list[int]]] = {}
    for i, row in control_table.reset_index(drop=True).iterrows():
        storage = str(row.get(storage_col, "") or "").strip()
        role = str(row.get(role_col, "") or "").lower()
        if not storage:
            continue
        inlet, outlet = pairs.setdefault(storage, ([], []))
        if "inlet" in role:
            inlet.append(int(i))
        elif "outlet" in role:
            outlet.append(int(i))
    return pairs


def _count_temporal_sequences(hist: pd.DataFrame) -> int:
    count = 0
    for value in hist.get("selected_action_sequence", pd.Series([], dtype=object)):
        seq = parse_sequence(value)
        if seq is not None and seq.ndim == 2 and seq.shape[0] > 1 and np.any(np.abs(np.diff(seq, axis=0)) > 1.0e-7):
            count += 1
    return count


def _count_multiactuator_rows(hist: pd.DataFrame) -> int:
    if "simultaneous_actuator_count" in hist:
        values = pd.to_numeric(hist["simultaneous_actuator_count"], errors="coerce").fillna(0)
        return int((values >= 2).sum())
    count = 0
    for value in hist.get("selected_action_sequence", pd.Series([], dtype=object)):
        seq = parse_sequence(value)
        if seq is not None and seq.ndim == 2 and np.any((np.abs(seq - seq[0:1]) > 1.0e-7).sum(axis=1) >= 2):
            count += 1
    return count


def evaluate_smoke_functionality(
    *,
    run_dir: str | Path,
    control_table: pd.DataFrame,
    required_return_period_groups: dict[str, list[str]],
    binary_pump_ids: Iterable[str],
    require_action_written: bool = True,
    require_temporal_action: bool = True,
    require_simultaneous_action: bool = True,
    forbid_fractional_binary_pumps: bool = True,
    forbid_storage_interlock_violations: bool = True,
) -> dict:
    run_dir = Path(run_dir)
    ids = _action_ids(control_table)
    histories = sorted((run_dir / "proposed").glob("*__controller_history.csv"))
    rows = []
    failures: list[str] = []
    if not histories:
        failures.append("missing_controller_history")
    binary = _binary_pump_ids(control_table, binary_pump_ids)
    binary_indices = [ids.index(aid) for aid in binary if aid in ids]
    storage_pairs = _storage_pairs(control_table)
    action_written_rows = 0
    temporal_rows = 0
    simultaneous_rows = 0
    fractional_pump_rows = 0
    storage_interlock_rows = 0
    event_ids: list[str] = []
    for path in histories:
        hist = pd.read_csv(path)
        event_id = path.name.replace("__controller_history.csv", "")
        event_ids.append(event_id)
        temporal = _count_temporal_sequences(hist)
        simultaneous = _count_multiactuator_rows(hist)
        temporal_rows += temporal
        simultaneous_rows += simultaneous
        for _, row in hist.iterrows():
            first = parse_sequence(row.get("executed_first_action", ""))
            if first is not None and first.ndim == 1 and ids and first.shape[0] == len(ids):
                if np.any(np.abs(first - 1.0) > 1.0e-7):
                    action_written_rows += 1
                if forbid_fractional_binary_pumps and binary_indices:
                    pump_values = first[binary_indices]
                    if np.any((pump_values > 1.0e-7) & (pump_values < 1.0 - 1.0e-7)):
                        fractional_pump_rows += 1
            seq = parse_sequence(row.get("selected_action_sequence", ""))
            if forbid_storage_interlock_violations and seq is not None and seq.ndim == 2:
                for inlet_idx, outlet_idx in storage_pairs.values():
                    if inlet_idx and outlet_idx:
                        inlet_changed = np.any(np.abs(seq[:, inlet_idx] - seq[0:1, inlet_idx]) > 1.0e-7, axis=1)
                        outlet_changed = np.any(np.abs(seq[:, outlet_idx] - seq[0:1, outlet_idx]) > 1.0e-7, axis=1)
                        storage_interlock_rows += int(np.logical_and(inlet_changed, outlet_changed).sum())
        rows.append(
            {
                "event_id": event_id,
                "return_period": event_return_period(event_id),
                "decisions": int(len(hist)),
                "temporal_action_rows": int(temporal),
                "simultaneous_action_rows": int(simultaneous),
            }
        )
    present_periods = {event_return_period(eid) for eid in event_ids}
    group_checks = {}
    for group, periods in required_return_period_groups.items():
        group_checks[group] = bool(present_periods.intersection({str(x) for x in periods}))
        if not group_checks[group]:
            failures.append(f"missing_smoke_return_period_group:{group}")
    if require_action_written and action_written_rows <= 0:
        failures.append("no_executed_action_written")
    if require_temporal_action and temporal_rows <= 0:
        failures.append("no_temporal_action_sequence_observed")
    if require_simultaneous_action and simultaneous_rows <= 0:
        failures.append("no_simultaneous_multi_actuator_action_observed")
    if fractional_pump_rows:
        failures.append(f"fractional_binary_pump_rows:{fractional_pump_rows}")
    if storage_interlock_rows:
        failures.append(f"storage_interlock_violation_rows:{storage_interlock_rows}")
    return {
        "passed": not failures,
        "failures": failures,
        "run_dir": str(run_dir),
        "event_count": int(len(set(event_ids))),
        "return_periods": sorted(present_periods),
        "required_return_period_groups": required_return_period_groups,
        "group_checks": group_checks,
        "action_written_rows": int(action_written_rows),
        "temporal_action_rows": int(temporal_rows),
        "simultaneous_action_rows": int(simultaneous_rows),
        "fractional_binary_pump_rows": int(fractional_pump_rows),
        "storage_interlock_violation_rows": int(storage_interlock_rows),
        "events": rows,
    }
