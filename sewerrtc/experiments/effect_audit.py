from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pandas as pd


def _embedded_value(data: Mapping[str, Any], key: str, row_index: int, default: str) -> str:
    if key not in data:
        return default
    values = data[key]
    try:
        return str(values[row_index])
    except (IndexError, TypeError):
        return default


def resolve_audit_metadata(
    *,
    pair_id: str,
    event_id: str,
    row_index: int,
    candidates: pd.DataFrame,
    data: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve audit metadata without requiring one manifest for every source dataset."""
    embedded_phase = _embedded_value(data, "phase", row_index, "unknown")
    embedded_kind = _embedded_value(data, "candidate_kind", row_index, "unknown")
    embedded_family = _embedded_value(data, "candidate_family", row_index, embedded_kind)
    embedded_checkpoint = _embedded_value(
        data,
        "checkpoint_id",
        row_index,
        f"{event_id}|{embedded_phase}|unknown",
    )
    source_dataset = _embedded_value(data, "source_dataset", row_index, "unknown")
    pair_key = str(pair_id)
    if pair_key not in candidates.index:
        specification = {
            "kind": embedded_kind,
            "mode": embedded_family,
            "actuators": [],
            "metadata_source": "embedded_dataset",
        }
        return {
            "manifest_match": False,
            "phase": embedded_phase,
            "checkpoint_id": embedded_checkpoint,
            "candidate_kind": embedded_kind,
            "candidate_mode": embedded_family,
            "source_dataset": source_dataset,
            "specification": specification,
        }

    record = candidates.loc[pair_key]
    if isinstance(record, pd.DataFrame):
        record = record.iloc[0]
    raw_specification = record.get("executed_action_sequence", "{}")
    specification = (
        json.loads(raw_specification)
        if isinstance(raw_specification, str) and raw_specification.strip()
        else {}
    )
    phase = str(record.get("phase", embedded_phase))
    checkpoint_id = embedded_checkpoint
    if checkpoint_id.endswith("|unknown") and pd.notna(record.get("split_timestamp_fraction")):
        checkpoint_id = f"{event_id}|{phase}|{float(record['split_timestamp_fraction']):.3f}"
    return {
        "manifest_match": True,
        "phase": phase,
        "checkpoint_id": checkpoint_id,
        "candidate_kind": str(specification.get("kind", embedded_kind)),
        "candidate_mode": str(specification.get("mode", embedded_family)),
        "source_dataset": source_dataset,
        "specification": specification,
    }
