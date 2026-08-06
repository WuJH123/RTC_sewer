"""Freeze a small, read-only state plan for the direct SWMM oracle benchmark."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plan_v42_targeted_candidate_expansion import _prefix_hash, _sha256


REGIME_ORDER = {
    "LOW_LOAD": 0,
    "MODERATE_LOAD": 1,
    "NEAR_CAPACITY": 2,
    "SEVERE_OVERLOAD": 3,
}


def _margin_slice(states: pd.DataFrame, relative: float, absolute: float) -> pd.DataFrame:
    selected = states[
        np.isclose(states["relative_margin_fraction"].astype(float), float(relative))
        & np.isclose(states["absolute_margin_m3"].astype(float), float(absolute))
    ].copy()
    if selected.empty:
        raise RuntimeError(f"missing Pareto margin slice: relative={relative}, absolute={absolute}")
    if selected["state_key"].astype(str).duplicated().any():
        raise RuntimeError("Pareto margin slice contains duplicate state keys")
    return selected


def _prepare(states: pd.DataFrame) -> pd.DataFrame:
    required = {
        "state_key",
        "event_id",
        "rainfall_sha256",
        "load_regime",
        "oracle_tfv_reduction_pct",
        "candidate_count",
        "actual_safe_candidate_count",
        "actual_safe_tfv_improving_count",
        "oracle_candidate_action_sha256",
    }
    missing = sorted(required - set(states.columns))
    if missing:
        raise RuntimeError(f"Pareto state table missing columns: {missing}")
    work = states.copy()
    work["state_key"] = work["state_key"].astype(str)
    work["load_regime"] = work["load_regime"].astype(str)
    work["oracle_tfv_reduction_pct"] = pd.to_numeric(
        work["oracle_tfv_reduction_pct"], errors="coerce"
    )
    for column in (
        "candidate_count",
        "actual_safe_candidate_count",
        "actual_safe_tfv_improving_count",
    ):
        work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0.0)
    return work


def _rank(frame: pd.DataFrame, *, ascending: bool) -> pd.DataFrame:
    return frame.sort_values(
        ["oracle_tfv_reduction_pct", "actual_safe_candidate_count", "candidate_count", "state_key"],
        ascending=[ascending, False, False, True],
        na_position="first" if ascending else "last",
        kind="stable",
    )


def _pick(
    work: pd.DataFrame,
    selected: set[str],
    predicate,
    count: int,
    reason: str,
    *,
    ascending: bool = True,
    fallback_nearest_zero: bool = False,
) -> list[dict[str, object]]:
    preferred = work[predicate(work) & ~work["state_key"].isin(selected)]
    fallback = work[~work["state_key"].isin(selected)]
    if fallback_nearest_zero:
        finite = fallback[fallback["oracle_tfv_reduction_pct"].notna()].copy()
        finite["_distance_to_zero"] = finite["oracle_tfv_reduction_pct"].abs()
        fallback = pd.concat(
            [
                finite.sort_values(
                    ["_distance_to_zero", "actual_safe_candidate_count", "candidate_count", "state_key"],
                    ascending=[True, False, False, True],
                    kind="stable",
                ).drop(columns=["_distance_to_zero"]),
                fallback[fallback["oracle_tfv_reduction_pct"].isna()],
            ]
        )
        chosen = pd.concat([_rank(preferred, ascending=ascending), fallback]).drop_duplicates("state_key")
    else:
        chosen = pd.concat([_rank(preferred, ascending=ascending), _rank(fallback, ascending=ascending)]).drop_duplicates("state_key")
    rows: list[dict[str, object]] = []
    for _, row in chosen.head(int(count)).iterrows():
        item = row.to_dict()
        item["selection_reason"] = reason if row["state_key"] in set(preferred["state_key"]) else f"{reason}_fallback"
        rows.append(item)
        selected.add(str(row["state_key"]))
    if len(rows) != int(count):
        raise RuntimeError(f"cannot select {count} states for {reason}")
    return rows


def _select_direct_states(states: pd.DataFrame) -> pd.DataFrame:
    """Select exactly 3/5/2/2 states using only the frozen Pareto slice."""
    work = _prepare(states)
    selected_keys: set[str] = set()
    selected: list[dict[str, object]] = []

    low = work[work["load_regime"].eq("LOW_LOAD")]
    moderate = work[work["load_regime"].eq("MODERATE_LOAD")]
    near = work[work["load_regime"].eq("NEAR_CAPACITY")]
    severe = work[work["load_regime"].eq("SEVERE_OVERLOAD")]

    selected.extend(_pick(low, selected_keys, lambda x: x["oracle_tfv_reduction_pct"].ge(20.0), 1, "low_positive_ge20", ascending=False))
    selected.extend(_pick(low, selected_keys, lambda x: x["oracle_tfv_reduction_pct"].between(5.0, 15.0, inclusive="both"), 1, "low_typical_5_to_15", ascending=False))
    selected.extend(_pick(low, selected_keys, lambda x: x["oracle_tfv_reduction_pct"].lt(5.0) | x["oracle_tfv_reduction_pct"].isna(), 1, "low_gap", ascending=True))

    selected.extend(_pick(moderate, selected_keys, lambda x: x["oracle_tfv_reduction_pct"].le(0.0) | x["oracle_tfv_reduction_pct"].isna(), 2, "moderate_nonpositive", ascending=True))
    selected.extend(_pick(moderate, selected_keys, lambda x: x["oracle_tfv_reduction_pct"].gt(0.0) & x["oracle_tfv_reduction_pct"].le(10.0), 2, "moderate_low_positive", ascending=True, fallback_nearest_zero=True))
    selected.extend(_pick(moderate, selected_keys, lambda x: x["oracle_tfv_reduction_pct"].notna(), 1, "moderate_best_available", ascending=False))

    selected.extend(_pick(near, selected_keys, lambda x: x["oracle_tfv_reduction_pct"].notna(), 1, "near_capacity_best", ascending=False))
    selected.extend(_pick(near, selected_keys, lambda x: x["oracle_tfv_reduction_pct"].notna(), 1, "near_capacity_gap", ascending=True))

    selected.extend(_pick(severe, selected_keys, lambda x: x["oracle_tfv_reduction_pct"].notna(), 1, "severe_overload_best", ascending=False))
    selected.extend(_pick(severe, selected_keys, lambda x: x["oracle_tfv_reduction_pct"].notna(), 1, "severe_overload_gap", ascending=True))

    plan = pd.DataFrame(selected)
    plan["regime_order"] = plan["load_regime"].map(REGIME_ORDER).fillna(99)
    return plan.sort_values(["regime_order", "selection_reason", "state_key"], kind="stable").drop(columns=["regime_order"]).reset_index(drop=True)


def _load_source(manifest: Path) -> pd.DataFrame:
    columns = [
        "state_key", "event_id", "rainfall_sha256", "checkpoint_min",
        "history_depth", "history_actions_readback", "rainfall_forecast",
        "source_detail_path_no_control", "source_detail_path_dynamic_internal",
        "source_detail_path_hold_previous",
    ]
    source = pd.read_parquet(manifest, columns=columns).drop_duplicates("state_key", keep="first")
    source["state_key"] = source["state_key"].astype(str)
    return source.set_index("state_key")


def _enrich(plan: pd.DataFrame, source: pd.DataFrame, manifest: Path) -> pd.DataFrame:
    counts = pd.read_parquet(manifest, columns=["state_key"]).astype({"state_key": str}).groupby("state_key").size()
    rows: list[dict[str, object]] = []
    for _, row in plan.iterrows():
        state_key = str(row["state_key"])
        if state_key not in source.index:
            raise RuntimeError(f"selected state missing from source manifest: {state_key}")
        src = source.loc[state_key]
        if str(row["event_id"]) != str(src["event_id"]) or str(row["rainfall_sha256"]) != str(src["rainfall_sha256"]):
            raise RuntimeError(f"state identity mismatch: {state_key}")
        payload = row.to_dict()
        payload.update({
            "state_key": state_key,
            "checkpoint_min": float(src["checkpoint_min"]),
            "prefix_state_sha256": _prefix_hash(src),
            "round2_manifest_rows": int(counts.get(state_key, 0)),
            "source_detail_path_no_control": str(src["source_detail_path_no_control"]),
            "source_detail_path_dynamic_internal": str(src["source_detail_path_dynamic_internal"]),
            "source_detail_path_hold_previous": str(src["source_detail_path_hold_previous"]),
            "round2_candidate_count": int(row["candidate_count"]),
            "round2_safe_count": int(row["actual_safe_candidate_count"]),
            "round2_safe_improving_count": int(row["actual_safe_tfv_improving_count"]),
            "round2_best_action_sha256": str(row["oracle_candidate_action_sha256"]),
        })
        rows.append(payload)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states-csv", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--relative-margin", type=float, default=0.05)
    parser.add_argument("--absolute-margin-m3", type=float, default=100.0)
    args = parser.parse_args()

    states = _margin_slice(pd.read_csv(args.states_csv), args.relative_margin, args.absolute_margin_m3)
    plan = _select_direct_states(states)
    if len(plan) != 12 or plan["state_key"].nunique() != 12:
        raise RuntimeError("direct oracle plan must contain 12 unique states")
    expected = {"LOW_LOAD": 3, "MODERATE_LOAD": 5, "NEAR_CAPACITY": 2, "SEVERE_OVERLOAD": 2}
    if plan.groupby("load_regime").size().to_dict() != expected:
        raise RuntimeError("direct oracle plan does not have the required regime counts")

    plan = _enrich(plan, _load_source(args.source_manifest), args.source_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "TRUE_STATE_DIRECT_ORACLE_STATE_PLAN.csv"
    json_path = args.output_dir / "TRUE_STATE_DIRECT_ORACLE_STATE_PLAN.json"
    lock_path = args.output_dir / "TRUE_STATE_DIRECT_ORACLE_STATE_LOCK.json"
    plan.to_csv(csv_path, index=False)
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True).strip()
    plan_sha = _sha256(csv_path)
    payload = {
        "audit_id": "V42_TRUE_STATE_DIRECT_SWMM_ORACLE_STATE_PLAN_V1",
        "development_only": True,
        "online_deployable": False,
        "new_swmm_started": False,
        "selection_source": str(args.states_csv),
        "source_manifest": str(args.source_manifest),
        "relative_margin_fraction": float(args.relative_margin),
        "absolute_margin_m3": float(args.absolute_margin_m3),
        "selected_states": int(len(plan)),
        "selected_by_regime": {str(k): int(v) for k, v in plan["load_regime"].value_counts().sort_index().items()},
        "plan_csv": str(csv_path),
        "plan_sha256": plan_sha,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "selection_uses_direct_results": False,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    lock = {
        "lock_id": "V42_TRUE_STATE_DIRECT_SWMM_ORACLE_STATE_LOCK_V1",
        "git_sha": git_sha,
        "plan_sha256": plan_sha,
        "states_source_sha256": _sha256(args.states_csv),
        "source_manifest_sha256": _sha256(args.source_manifest),
        "relative_margin_fraction": float(args.relative_margin),
        "absolute_margin_m3": float(args.absolute_margin_m3),
        "selected_states": plan[["state_key", "event_id", "rainfall_sha256", "checkpoint_min", "load_regime", "prefix_state_sha256", "selection_reason"]].to_dict(orient="records"),
        "selection_uses_direct_results": False,
        "new_swmm_started": False,
    }
    lock_path.write_text(json.dumps(lock, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
