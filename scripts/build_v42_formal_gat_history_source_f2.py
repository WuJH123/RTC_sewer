"""Build the Formal F2 causal-history source manifest.

Candidate detail files remain outcome sources. This manifest selects a
separate same-state trajectory covering checkpoint-120..checkpoint and all
13 exact GAT anchors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sewerrtc.v4.formal_f2 import FORMAL_GENERATION_ID, read_table
from sewerrtc.v4.v42_qualification_history_resolver import (
    build_history_index,
    choose_history_compatible_state,
)
from sewerrtc.v4.v42_step1_dataset import _build_usecols, load_graph_assets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_GATES = (
    "training_admission_authorized",
    "raw_independent_oracle_all_pass",
    "same_state_raw_verified",
    "same_forcing_raw_verified",
    "actual_readback_verified",
    "h120_window_complete",
    "kpi_recompute_ok",
)


def _digest(signature: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in ("checkpoint_depth", "rainfall_history", "pre_action_history"):
        digest.update(key.encode("utf-8"))
        digest.update(np.ascontiguousarray(signature[key], dtype=np.float64).tobytes())
    return digest.hexdigest()


def _loader(graph: Any):
    required = _build_usecols(graph.node_ids, graph.facility_ids)
    cache: OrderedDict[str, pd.DataFrame] = OrderedDict()

    def load(path: Path) -> pd.DataFrame:
        key = str(Path(path).resolve())
        if key in cache:
            value = cache.pop(key)
            cache[key] = value
            return value
        header = pd.read_csv(key, nrows=0)
        missing = [column for column in required if column not in header.columns]
        if missing:
            raise KeyError(f"Formal history detail missing columns: {missing[:10]}")
        value = pd.read_csv(key, usecols=required, low_memory=False).loc[:, required]
        cache[key] = value
        while len(cache) > 12:
            cache.popitem(last=False)
        return value

    return load


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--raw-manifest", type=Path, required=True)
    ap.add_argument("--step1-window-manifest", type=Path, required=True)
    ap.add_argument("--output-manifest", type=Path, required=True)
    ap.add_argument("--min-rainfall-groups", type=int, default=69)
    ap.add_argument("--min-checkpoint-min", type=float, default=120.0)
    args = ap.parse_args()

    raw_all = read_table(args.raw_manifest)
    raw = raw_all.copy()
    step1 = read_table(args.step1_window_manifest)
    missing = sorted(set(RAW_GATES) - set(raw.columns))
    if missing:
        raise KeyError(f"Formal raw manifest missing gates: {missing}")
    if not raw[RAW_GATES].fillna(False).astype(bool).all().all():
        raise RuntimeError("Formal history-source builder requires fully admitted raw rows")
    raw = raw[pd.to_numeric(raw["checkpoint_min"], errors="coerce") >= args.min_checkpoint_min].copy()
    if raw.empty:
        raise RuntimeError("no raw states remain after checkpoint gate")

    graph = load_graph_assets(args.project_root)
    history_index = build_history_index(step1)
    load_detail = _loader(graph)
    rows: list[dict[str, Any]] = []
    for rainfall, group in raw.groupby("split_group_key", sort=True):
        chosen = choose_history_compatible_state(
            group,
            history_index=history_index,
            load_detail=load_detail,
            graph=graph,
            required_candidates=3,
            min_checkpoint_min=args.min_checkpoint_min,
        )
        if chosen is None or not chosen.get("compatible"):
            rows.append({
                "formal_generation_id": FORMAL_GENERATION_ID,
                "development_only": False,
                "formal_mainline_authorized": False,
                "rainfall_sha256": str(rainfall),
                "rainfall_group_key": str(rainfall),
                "state_key": "",
                "compatible": False,
                "failure_reason": str((chosen or {}).get("failure_reason", "no_history_compatible_state")),
                "candidate_detail_path": "",
                "history_detail_path": "",
                "checkpoint_min": np.nan,
                "candidate_count": 0,
            })
            continue
        state = chosen["state"]
        first = state.iloc[0]
        history = chosen["history"]
        rows.append({
            "formal_generation_id": FORMAL_GENERATION_ID,
            "development_only": False,
            "formal_mainline_authorized": False,
            "rainfall_sha256": str(rainfall),
            "rainfall_group_key": str(rainfall),
            "event_id": str(first.get("event_id", "")),
            "state_key": str(chosen["state_key"]),
            "checkpoint_min": float(first["checkpoint_min"]),
            "candidate_detail_path": str(first["source_detail_path_candidate"]),
            "history_detail_path": str(history["history_detail_path"]),
            "history_start_min": float(history["history_start_min"]),
            "history_end_min": float(history["history_end_min"]),
            "history_match_level": str(history["history_match_level"]),
            "candidate_pre_action_signature": _digest(history["candidate_pre_action_signature"]),
            "history_pre_action_signature": _digest(history["history_pre_action_signature"]),
            "candidate_count": int(len(state)),
            "compatible": True,
            "failure_reason": "",
        })

    result = pd.DataFrame(rows)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output_manifest, index=False)
    compatible = result[result["compatible"].fillna(False).astype(bool)]
    audit = {
        "formal_generation_id": FORMAL_GENERATION_ID,
        "stage": "formal_f2_history_source_manifest",
        "status": "pass" if compatible["rainfall_group_key"].nunique() >= args.min_rainfall_groups else "fail",
        "development_only": False,
        "formal_mainline_authorized": False,
        "raw_rows_before_checkpoint_gate": int(len(raw_all)),
        "raw_rows_after_checkpoint_gate": int(len(raw)),
        "input_rainfall_groups": int(raw["split_group_key"].nunique()),
        "compatible_rainfall_groups": int(compatible["rainfall_group_key"].nunique()),
        "compatible_states": int(compatible["state_key"].nunique()),
        "failed_groups": int(len(result) - len(compatible)),
        "min_checkpoint_min": float(args.min_checkpoint_min),
        "candidate_count_min": int(compatible["candidate_count"].min()) if not compatible.empty else 0,
        "output_manifest": str(args.output_manifest),
        "failure_examples": result.loc[~result["compatible"].fillna(False).astype(bool)].head(100).to_dict("records"),
    }
    audit_path = args.output_manifest.with_name("FORMAL_F2_HISTORY_SOURCE_AUDIT.json")
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if audit["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
