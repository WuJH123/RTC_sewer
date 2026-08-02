"""Materialise selected existing V4 cases into the fast Step2 control-core schema.

This is a development-only bridge.  It reads only the cases selected by
``build_v42_fast_core_pool.py`` and their completion/detail files; it does not
perform an exhaustive historical trajectory scan and it never runs SWMM.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.v4.v42_fast_feasibility import (
    _cols,
    _kpis,
    _load_graph_topology,
    _rain,
    _read_core_detail,
    _select,
)
from sewerrtc.v4.v42_priority_contract import get_pfv_core_node_indices
from sewerrtc.v4.v42_trajectory_builder import N_HISTORY_FRAMES, N_HORIZON_STEPS

ROLE_ALIASES = {
    "candidate": ("candidate",),
    "no_control": ("no_control",),
    "dynamic_internal": ("dynamic_internal", "dynamic_internal_rules"),
    # Do not silently treat passive_anchor as Hold-Previous.  Historical
    # passive anchors can converge to a different future action sequence.
    "hold_previous": ("hold_previous",),
}


def _completion_index(output_root: Path) -> dict[str, Path]:
    paths: list[Path] = []
    try:
        result = subprocess.run(
            ["rg", "--files", "-uu", "-g", "completion.json", str(output_root)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode in (0, 1):
            paths = [Path(x.strip()) for x in result.stdout.splitlines() if x.strip()]
    except FileNotFoundError:
        paths = list(output_root.rglob("completion.json"))
    index: dict[str, Path] = {}
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            case_id = str(payload.get("case_id", "")).strip()
            if case_id and case_id not in index:
                index[case_id] = path
        except Exception:
            continue
    return index


def _resolve_detail(completion_path: Path, payload: dict[str, Any], role: str) -> Path:
    branches = payload.get("branches", {})
    for key in ROLE_ALIASES[role]:
        value = branches.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            text = value
        elif isinstance(value, dict):
            text = str(value.get("detail_path") or value.get("path") or value.get("detail") or "")
        else:
            text = ""
        if not text:
            continue
        path = Path(text)
        if path.exists():
            return path
        local = completion_path.parent / path.name
        if local.exists():
            return local
        if not path.is_absolute():
            candidate = completion_path.parent / path
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"{completion_path}: no detail for role={role}")


def _times(checkpoint: float) -> tuple[list[float], list[float]]:
    history = [checkpoint - (N_HISTORY_FRAMES - 1 - i) * 5.0 for i in range(N_HISTORY_FRAMES)]
    future = [checkpoint + (i + 1) * 10.0 for i in range(N_HORIZON_STEPS)]
    return history, future


def _case_uid(state_key: str, case_id: str) -> str:
    return hashlib.sha256(f"{state_key}|{case_id}".encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/project6_dual_reference_v4/final_v4",
    )
    ap.add_argument(
        "--core-pool-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/project6_dual_reference_v4/final_v4/v42_paper/fast_e2e_64plus/core_pool",
    )
    ap.add_argument(
        "--output-manifest",
        type=Path,
        default=PROJECT_ROOT / "outputs/project6_dual_reference_v4/final_v4/v42_paper/fast_e2e_64plus/step2_fast_e2e_core_manifest.parquet",
    )
    ap.add_argument("--audit-output", type=Path, default=None)
    ap.add_argument("--target-groups", type=int, default=88)
    ap.add_argument("--min-groups", type=int, default=64)
    ap.add_argument("--candidates-per-state", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    selected_path = args.core_pool_dir / "FAST_CORE_SELECTED_CASES.parquet"
    split_path = args.core_pool_dir / "FAST_CORE_RAINFALL_GROUPS.csv"
    if not selected_path.exists() or not split_path.exists():
        raise FileNotFoundError("run build_v42_fast_core_pool.py before materialisation")
    selected = pd.read_parquet(selected_path)
    split = pd.read_csv(split_path)
    if selected.empty or split.empty:
        raise ValueError("fast core pool is empty")

    groups = split["rainfall_group_key"].astype(str).tolist()
    ranked = sorted(
        groups,
        key=lambda g: (hashlib.sha256(f"{args.seed}:{g}".encode()).hexdigest(), g),
    )
    chosen_groups = ranked[: min(int(args.target_groups), len(ranked))]
    if len(chosen_groups) < int(args.min_groups):
        raise RuntimeError(
            f"selected core pool has only {len(chosen_groups)} rainfall groups; minimum={args.min_groups}"
        )
    selected = selected[selected["rainfall_group_key"].astype(str).isin(chosen_groups)].copy()

    graph = _load_graph_topology(args.project_root)
    node_ids = [str(x) for x in graph["node_ids"]]
    facility_ids = [str(x) for x in graph["facility_ids"]]
    priority_idx = get_pfv_core_node_indices(node_ids)
    completion_by_case = _completion_index(args.output_root)
    if not completion_by_case:
        raise FileNotFoundError(f"no completion.json records under {args.output_root}")

    detail_cache: dict[str, pd.DataFrame] = {}
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for _, row in selected.iterrows():
        case_id = str(row.get("case_id", "")).strip()
        state_key = str(row.get("counterfactual_state_key", "")).strip()
        try:
            if not case_id or case_id not in completion_by_case:
                raise KeyError(f"completion not found for case_id={case_id!r}")
            checkpoint = float(row["checkpoint_min"])
            history_times, future_times = _times(checkpoint)
            completion_path = completion_by_case[case_id]
            payload = json.loads(completion_path.read_text(encoding="utf-8"))
            paths = {role: _resolve_detail(completion_path, payload, role) for role in ROLE_ALIASES}
            details: dict[str, pd.DataFrame] = {}
            for role, path in paths.items():
                key = str(path.resolve())
                if key not in detail_cache:
                    detail_cache[key] = _read_core_detail(path, node_ids, facility_ids)
                details[role] = detail_cache[key]

            history = _select(details["candidate"], history_times)
            history_depth = _cols(history, "h:", node_ids)
            history_actions = _cols(history, "setting:", facility_ids)

            branches: dict[str, dict[str, np.ndarray]] = {}
            for role, detail in details.items():
                future = _select(detail, future_times)
                branches[role] = {
                    "depth": _cols(future, "h:", node_ids),
                    "flood": _cols(future, "flood:", node_ids),
                    "action": _cols(future, "setting:", facility_ids),
                    "rainfall": _rain(future),
                }
            rainfall = branches["candidate"]["rainfall"]
            for role in ("no_control", "dynamic_internal", "hold_previous"):
                if not np.allclose(rainfall, branches[role]["rainfall"], atol=1e-7, rtol=0.0):
                    raise ValueError(f"future rainfall differs for role={role}")
            pfv, tfv, peak = _kpis(branches, priority_idx)
            rec: dict[str, Any] = {
                "contract_id": "PROJECT6_V42_FAST_CORE_EXISTING_V1",
                "development_only": True,
                "formal_target_domain": False,
                "case_uid": _case_uid(state_key, case_id),
                "case_id": case_id,
                "event_id": str(row.get("event_id", "")),
                "checkpoint_min": checkpoint,
                "state_key": state_key,
                "domain_id": str(row.get("domain_id", "")),
                "split_group_key": str(row["rainfall_group_key"]),
                "history_depth": json.dumps(history_depth.tolist(), allow_nan=False),
                "history_actions_readback": json.dumps(history_actions.tolist(), allow_nan=False),
                # Replaced with a causal forecast by materialize_v42_fast_gat_history.py.
                "rainfall_forecast": json.dumps(rainfall.tolist(), allow_nan=False),
                "pfv_delta": float(pfv),
                "tfv_delta": float(tfv),
                "peak_delta": float(peak),
                "fast_e2e_admission_tier": str(row.get("fast_e2e_admission_tier", "")),
            }
            for role, arrays in branches.items():
                rec[f"action_{role}_readback"] = json.dumps(arrays["action"].tolist(), allow_nan=False)
                rec[f"trajectory_depth_{role}"] = json.dumps(arrays["depth"].tolist(), allow_nan=False)
                rec[f"trajectory_flood_{role}"] = json.dumps(arrays["flood"].tolist(), allow_nan=False)
                rec[f"source_detail_path_{role}"] = str(paths[role])
            records.append(rec)
        except Exception as exc:
            failures.append(
                {
                    "case_id": case_id,
                    "state_key": state_key,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError("no selected V4 core case materialised successfully")
    state_counts = frame.groupby("state_key").size()
    good_states = set(state_counts[state_counts.ge(int(args.candidates_per_state))].index.astype(str))
    frame = frame[frame["state_key"].astype(str).isin(good_states)].copy()
    output_groups = int(frame["split_group_key"].astype(str).nunique()) if not frame.empty else 0
    if output_groups < int(args.min_groups):
        raise RuntimeError(
            f"materialised core retained only {output_groups} rainfall groups with candidate choice; minimum={args.min_groups}"
        )

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output_manifest, index=False)
    audit_path = args.audit_output or args.output_manifest.with_name("step2_fast_core_existing_audit.json")
    audit = {
        "development_only": True,
        "formal_mainline_authorized": False,
        "selected_source_rows": int(len(selected)),
        "accepted_rows_before_state_gate": int(len(records)),
        "output_rows": int(len(frame)),
        "output_states": int(frame["state_key"].nunique()),
        "output_rainfall_groups": output_groups,
        "minimum_required_groups": int(args.min_groups),
        "target_groups": int(args.target_groups),
        "completion_index_size": int(len(completion_by_case)),
        "failed_rows": int(len(failures)),
        "failure_examples": failures[:50],
        "authority": "existing_recorded_SWMM_detail_only_no_new_SWMM",
    }
    audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
