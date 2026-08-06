"""Recompute historical V4.2 candidates into one canonical experience bank.

No SWMM is started.  Every PFV/TFV label is recomputed from recorded detail.csv
using the shared authoritative metric functions.  Legacy stored KPI labels are
never treated as authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.control.authoritative_control_metrics_v42 import (
    DT_SEC,
    _detail_metrics,
    _prefix,
    action_sha256,
    detail_horizon_metrics,
    rolling_pfv_budget_metric,
)
from sewerrtc.control.experience_bank_v42 import action_bank_row, state_signature
from sewerrtc.v4.v42_priority_contract import get_pfv_core_node_indices
from sewerrtc.v4.v42_trajectory_builder import _load_graph_topology


def _array(value: Any) -> np.ndarray:
    if isinstance(value, str):
        value = json.loads(value)
    return np.asarray(value, dtype=np.float32)


def _first_existing(row: pd.Series, names: list[str]) -> str:
    for name in names:
        if name in row and pd.notna(row[name]) and str(row[name]).strip():
            return str(row[name])
    return ""


def _candidate_action(row: pd.Series) -> np.ndarray:
    for name in ("action_candidate_readback", "candidate_action", "sequence", "candidate_action_json"):
        if name in row and row[name] is not None and not (isinstance(row[name], float) and np.isnan(row[name])):
            try:
                value = _array(row[name])
                if value.ndim == 2:
                    return value
            except Exception:
                pass
    raise ValueError("candidate row has no 2D action sequence")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_parquet_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read only the manifest fields needed by the bank builder.

    The Round2 manifest contains large serialized trajectory columns.  Loading
    all 121 columns defeats the bounded-memory contract even though the bank
    only needs paths, actions, and causal state fields.
    """
    import pyarrow.parquet as pq

    names = set(pq.ParquetFile(path).schema.names)
    missing = [name for name in columns if name not in names]
    if missing:
        raise KeyError(f"{path} missing required columns: {missing}")
    return pd.read_parquet(path, columns=columns)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--state-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relative-margin", type=float, default=0.05)
    parser.add_argument("--absolute-margin-m3", type=float, default=100.0)
    args = parser.parse_args()

    candidate_columns = [
        "state_key", "event_id", "rainfall_sha256", "checkpoint_min",
        "candidate_action_sha256", "action_candidate_readback",
        "source_detail_path_candidate", "source_detail_path_no_control",
        "source_detail_path_dynamic_internal", "candidate_detail_sha256",
    ]
    state_columns = [
        "state_key", "event_id", "rainfall_sha256", "checkpoint_min",
        "history_depth", "rainfall_forecast", "action_hold_previous_readback",
        "source_detail_path_dynamic_internal", "state_source",
        "history_input_contract", "reconstructor_contract",
        "reconstructed_history_contract", "sensor_layout_sha256",
    ]
    candidates = _read_parquet_columns(args.candidate_manifest, candidate_columns)
    states = _read_parquet_columns(args.state_manifest, state_columns)
    if "state_key" not in candidates or "state_key" not in states:
        raise KeyError("candidate and state manifests must contain state_key")
    candidates = candidates.copy()
    candidates["state_key"] = candidates["state_key"].astype(str)
    states = states.copy()
    states["state_key"] = states["state_key"].astype(str)
    states = states.drop_duplicates("state_key", keep="first").set_index("state_key")

    graph = _load_graph_topology(args.project_root)
    node_ids = [str(x) for x in graph["node_ids"]]
    priority_nodes = [node_ids[int(i)] for i in get_pfv_core_node_indices(node_ids)]
    # Keep only a small reference cache.  Candidate detail files are normally
    # unique and must not accumulate into multi-GB resident memory.
    detail_cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
    reference_paths = set(
        str(value)
        for value in pd.concat(
            [candidates["source_detail_path_no_control"], candidates["source_detail_path_dynamic_internal"]],
            ignore_index=True,
        ).dropna().astype(str)
        if value.strip()
    )
    detail_cache_limit = 8
    detail_sha_cache: dict[str, str] = {}
    metric_cache: dict[tuple[str, float], dict[str, float]] = {}

    def detail(path: str) -> pd.DataFrame:
        if path in detail_cache:
            frame = detail_cache.pop(path)
            detail_cache[path] = frame
            return frame
        frame = pd.read_csv(path, low_memory=False)
        if path in reference_paths:
            detail_cache[path] = frame
            while len(detail_cache) > detail_cache_limit:
                detail_cache.popitem(last=False)
        return frame

    def detail_sha(path: str) -> str:
        if path not in detail_sha_cache:
            detail_sha_cache[path] = _sha256(Path(path))
        return detail_sha_cache[path]

    def h120(path: str, checkpoint: float) -> dict[str, float]:
        key = (path, float(checkpoint))
        if key not in metric_cache:
            metric_cache[key] = detail_horizon_metrics(
                detail(path), priority_nodes, checkpoint_min=float(checkpoint), steps=12
            )
        return metric_cache[key]

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    seen: dict[tuple[str, str], str] = {}
    for index, candidate_row in candidates.iterrows():
        state_key = str(candidate_row["state_key"])
        if state_key not in states.index:
            failures.append({"row": int(index), "state_key": state_key, "error": "missing_state_manifest"})
            continue
        state_row = states.loc[state_key]
        try:
            checkpoint = float(candidate_row.get("checkpoint_min", state_row.get("checkpoint_min")))
            action = _candidate_action(candidate_row)
            computed_action_sha = action_sha256(action)
            stored_action_sha = str(candidate_row.get("candidate_action_sha256", "")).strip()
            if stored_action_sha and stored_action_sha != computed_action_sha:
                raise ValueError("stored candidate_action_sha256 differs from canonical action hash")
            action_sha = computed_action_sha
            identity = (state_key, action_sha)
            candidate_path = _first_existing(candidate_row, ["source_detail_path_candidate", "candidate_detail", "detail_path"])
            no_control_path = _first_existing(candidate_row, ["source_detail_path_no_control", "no_control_detail"])
            internal_path = _first_existing(candidate_row, ["source_detail_path_dynamic_internal", "source_detail_path_internal", "dynamic_internal_detail"])
            if not internal_path:
                internal_path = _first_existing(state_row, ["source_detail_path_dynamic_internal", "source_detail_path_internal"])
            if not candidate_path or not no_control_path or not internal_path:
                raise FileNotFoundError("candidate/no-control/internal detail path missing")
            for path in (candidate_path, no_control_path, internal_path):
                if not Path(path).exists():
                    raise FileNotFoundError(path)
            candidate_detail = detail(candidate_path)
            nc_detail = detail(no_control_path)
            candidate_metric = h120(candidate_path, checkpoint)
            nc_metric = h120(no_control_path, checkpoint)
            internal_metric = h120(internal_path, checkpoint)
            candidate_prefix = _prefix(candidate_detail, checkpoint)
            nc_prefix = _prefix(nc_detail, checkpoint)
            candidate_prefix_metric = (
                _detail_metrics(candidate_prefix, priority_nodes, dt_sec=DT_SEC) if len(candidate_prefix) else {"PFV": 0.0}
            )
            nc_prefix_metric = (
                _detail_metrics(nc_prefix, priority_nodes, dt_sec=DT_SEC) if len(nc_prefix) else {"PFV": 0.0}
            )
            candidate_prefix_pfv = float(candidate_prefix_metric["PFV"])
            nc_prefix_pfv = float(nc_prefix_metric["PFV"])
            candidate_composed_pfv = candidate_prefix_pfv + float(candidate_metric["PFV"])
            nc_composed_pfv = nc_prefix_pfv + float(nc_metric["PFV"])
            rolling_metric = rolling_pfv_budget_metric(
                candidate_detail,
                nc_detail,
                priority_nodes=priority_nodes,
                checkpoint_min=checkpoint,
                relative_margin=float(args.relative_margin),
                steps=12,
            )
            reconstructed_rolling = candidate_composed_pfv - (1.0 + float(args.relative_margin)) * nc_composed_pfv
            if abs(float(rolling_metric) - float(reconstructed_rolling)) > 1.0e-6:
                raise RuntimeError("rolling PFV component reconstruction mismatch")
            fingerprint = json.dumps(
                {
                    "candidate_path": str(Path(candidate_path).resolve()),
                    "no_control_path": str(Path(no_control_path).resolve()),
                    "internal_path": str(Path(internal_path).resolve()),
                    "candidate_pfv": float(candidate_metric["PFV"]),
                    "candidate_tfv": float(candidate_metric["TFV"]),
                    "nc_pfv": float(nc_metric["PFV"]),
                    "internal_tfv": float(internal_metric["TFV"]),
                    "rolling_metric": float(rolling_metric),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            if identity in seen:
                if seen[identity] != fingerprint:
                    failures.append({"row": int(index), "state_key": state_key, "error": "duplicate_identity_conflict", "candidate_action_sha256": action_sha})
                continue
            seen[identity] = fingerprint
            tfv_internal = float(internal_metric["TFV"])
            reduction = (
                100.0 * (tfv_internal - float(candidate_metric["TFV"])) / tfv_internal
                if abs(tfv_internal) > 1.0e-9
                else float("nan")
            )
            history_depth = _array(state_row["history_depth"])
            rainfall = _array(state_row["rainfall_forecast"]).reshape(-1)
            hold = _array(state_row["action_hold_previous_readback"])
            current = hold[0] if hold.ndim == 2 else hold.reshape(-1)
            signature = state_signature(
                state_history=history_depth,
                rainfall_forecast=rainfall,
                current_action=current,
            )
            rows.append(
                action_bank_row(
                    state_key=state_key,
                    state_signature_value=signature,
                    candidate_action=action,
                    pfv_candidate_m3=float(candidate_metric["PFV"]),
                    pfv_no_control_m3=float(nc_metric["PFV"]),
                    pfv_budget_metric_m3=float(rolling_metric),
                    pfv_feasible=bool(rolling_metric <= float(args.absolute_margin_m3) + 1.0e-9),
                    tfv_candidate_m3=float(candidate_metric["TFV"]),
                    tfv_internal_m3=tfv_internal,
                    tfv_reduction_pct=float(reduction),
                    extra={
                        "event_id": str(candidate_row.get("event_id", state_row.get("event_id", ""))),
                        "rainfall_sha256": str(candidate_row.get("rainfall_sha256", state_row.get("rainfall_sha256", ""))),
                        "checkpoint_min": checkpoint,
                        "relative_margin_fraction": float(args.relative_margin),
                        "absolute_margin_m3": float(args.absolute_margin_m3),
                        "pfv_candidate_prefix_m3": candidate_prefix_pfv,
                        "pfv_no_control_prefix_m3": nc_prefix_pfv,
                        "pfv_candidate_composed_prefix_plus_h120_m3": candidate_composed_pfv,
                        "pfv_no_control_composed_prefix_plus_h120_m3": nc_composed_pfv,
                        "source_detail_path_candidate": candidate_path,
                        "source_detail_path_no_control": no_control_path,
                        "source_detail_path_dynamic_internal": internal_path,
                        "candidate_detail_sha256": str(candidate_row.get("candidate_detail_sha256", "")).strip(),
                        "no_control_detail_sha256": detail_sha(no_control_path),
                        "dynamic_internal_detail_sha256": detail_sha(internal_path),
                        "candidate_action_sha256_verified": True,
                        "state_source": str(state_row.get("state_source", "")),
                        "history_input_contract": str(state_row.get("history_input_contract", "")),
                        "reconstructor_contract": str(state_row.get("reconstructor_contract", "")),
                        "reconstructed_history_contract": str(state_row.get("reconstructed_history_contract", "")),
                        "sensor_layout_sha256": str(state_row.get("sensor_layout_sha256", "")),
                        "global_peak_candidate": float(candidate_metric["peak_TFV_rate"]),
                        "global_peak_internal": float(internal_metric["peak_TFV_rate"]),
                    },
                )
            )
        except Exception as exc:
            failures.append({"row": int(index), "state_key": state_key, "error": repr(exc)})

    output = pd.DataFrame(rows)
    if output.empty:
        raise RuntimeError("canonical experience bank is empty")
    output = output.sort_values(["state_key", "candidate_action_sha256"], kind="stable").reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() in {".parquet", ".pq"}:
        output.to_parquet(args.output, index=False)
    else:
        output.to_csv(args.output, index=False)
    failure_path = args.output.with_name(args.output.stem + "_FAILURES.json")
    failure_path.write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "contract": "V42_AUTHORITATIVE_EXPERIENCE_BANK_BUILD_V1",
        "new_swmm_started": False,
        "legacy_stored_labels_used": False,
        "metric_source": "detail.csv_shared_authoritative_h120_plus_rolling_pfv",
        "pfv_tradeoff_basis": "realised_prefix_plus_H120_candidate_vs_same_no_control_composition",
        "input_candidate_rows": int(len(candidates)),
        "output_unique_rows": int(len(output)),
        "input_unique_action_identities": int(len(seen)),
        "states": int(output["state_key"].nunique()),
        "pfv_safe_rows": int(output["pfv_feasible"].sum()),
        "tfv_improving_rows": int((output["tfv_reduction_pct"] > 0).sum()),
        "pfv_safe_tfv_improving_rows": int((output["pfv_feasible"] & (output["tfv_reduction_pct"] > 0)).sum()),
        "failures": int(len(failures)),
        "relative_margin_fraction": float(args.relative_margin),
        "absolute_margin_m3": float(args.absolute_margin_m3),
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "candidate_manifest_sha256": _sha256(args.candidate_manifest),
        "state_manifest_sha256": _sha256(args.state_manifest),
    }
    args.output.with_name(args.output.stem + "_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
