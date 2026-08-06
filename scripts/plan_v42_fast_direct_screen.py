"""Freeze the bounded 8-state true-state SWMM screening plan.

This is development-only planning.  It reads the frozen Round2 evidence and
does not start SWMM.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.contracts.prompt3a import managed_facility_ids
from scripts.plan_v42_targeted_candidate_expansion import _prefix_hash


REGIMES = ("LOW_LOAD", "MODERATE_LOAD", "NEAR_CAPACITY", "SEVERE_OVERLOAD")
BINARY = {"ADD301.2", "ADD301.3"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value: object) -> np.ndarray:
    return np.asarray(json.loads(value) if isinstance(value, str) else value, dtype=float)


def _json_records(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in frame[columns].to_dict(orient="records"):
        rows.append({key: (None if pd.isna(value) else value) for key, value in item.items()})
    return rows


def _rank(frame: pd.DataFrame, *, ascending: bool = False) -> pd.DataFrame:
    return frame.sort_values(
        ["oracle_tfv_reduction_pct", "actual_safe_candidate_count", "candidate_count", "state_key"],
        ascending=[ascending, False, False, True],
        na_position="last" if not ascending else "first",
        kind="stable",
    )


def _pick(frame: pd.DataFrame, used: set[str], predicate, count: int, reason: str, *, ascending: bool = False) -> list[dict[str, object]]:
    preferred = frame[predicate(frame) & ~frame["state_key"].isin(used)]
    remaining = frame[~frame["state_key"].isin(used)]
    chosen = pd.concat([_rank(preferred, ascending=ascending), _rank(remaining, ascending=ascending)]).drop_duplicates("state_key")
    result: list[dict[str, object]] = []
    for _, row in chosen.head(count).iterrows():
        item = row.to_dict()
        item["selection_reason"] = reason if row["state_key"] in set(preferred["state_key"]) else reason + "_fallback"
        result.append(item)
        used.add(str(row["state_key"]))
    if len(result) != count:
        raise RuntimeError(f"cannot select {count} states for {reason}")
    return result


def select_fast_states(states: pd.DataFrame) -> pd.DataFrame:
    required = {"state_key", "event_id", "rainfall_sha256", "load_regime", "oracle_tfv_reduction_pct", "candidate_count", "actual_safe_candidate_count"}
    missing = sorted(required - set(states.columns))
    if missing:
        raise RuntimeError(f"Pareto table missing columns: {missing}")
    work = states.copy()
    work["state_key"] = work["state_key"].astype(str)
    work["load_regime"] = work["load_regime"].astype(str)
    work["oracle_tfv_reduction_pct"] = pd.to_numeric(work["oracle_tfv_reduction_pct"], errors="coerce")
    for column in ("candidate_count", "actual_safe_candidate_count"):
        work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0)
    used: set[str] = set()
    selected: list[dict[str, object]] = []
    low = work[work.load_regime.eq("LOW_LOAD")]
    moderate = work[work.load_regime.eq("MODERATE_LOAD")]
    near = work[work.load_regime.eq("NEAR_CAPACITY")]
    severe = work[work.load_regime.eq("SEVERE_OVERLOAD")]
    selected += _pick(low, used, lambda x: x.oracle_tfv_reduction_pct.ge(20), 1, "low_positive", ascending=False)
    selected += _pick(low, used, lambda x: x.oracle_tfv_reduction_pct.lt(5) | x.oracle_tfv_reduction_pct.isna(), 1, "low_gap", ascending=True)
    selected += _pick(moderate, used, lambda x: x.oracle_tfv_reduction_pct.lt(0) | x.oracle_tfv_reduction_pct.isna(), 1, "moderate_negative_or_missing", ascending=True)
    selected += _pick(moderate, used, lambda x: x.oracle_tfv_reduction_pct.ge(0) & x.oracle_tfv_reduction_pct.le(5), 1, "moderate_zero_to_five", ascending=True)
    selected += _pick(moderate, used, lambda x: x.oracle_tfv_reduction_pct.notna(), 1, "moderate_best", ascending=False)
    finite_near = near[near.oracle_tfv_reduction_pct.notna()].copy()
    near_target = float(finite_near.oracle_tfv_reduction_pct.median()) if not finite_near.empty else 0.0
    near = near.copy()
    near["_near_distance"] = (near.oracle_tfv_reduction_pct - near_target).abs()
    near_preferred = near[~near.state_key.isin(used)].sort_values(["_near_distance", "state_key"], kind="stable")
    if near_preferred.empty:
        raise RuntimeError("cannot select typical near-capacity state")
    near_row = near_preferred.iloc[0].drop(labels=["_near_distance"]).to_dict()
    near_row["selection_reason"] = "near_typical"
    selected.append(near_row)
    used.add(str(near_row["state_key"]))
    selected += _pick(severe, used, lambda x: x.oracle_tfv_reduction_pct.notna(), 1, "severe_best", ascending=False)
    selected += _pick(severe, used, lambda x: x.oracle_tfv_reduction_pct.notna(), 1, "severe_gap", ascending=True)
    out = pd.DataFrame(selected)
    order = {name: index for index, name in enumerate(REGIMES)}
    out["regime_order"] = out.load_regime.map(order).fillna(99)
    return out.sort_values(["regime_order", "selection_reason", "state_key"], kind="stable").drop(columns="regime_order").reset_index(drop=True)


def _source_rows(manifest: Path) -> pd.DataFrame:
    columns = [
        "state_key", "event_id", "rainfall_sha256", "checkpoint_min", "history_depth", "history_actions_readback", "rainfall_forecast",
        "source_detail_path_no_control", "source_detail_path_dynamic_internal", "source_detail_path_hold_previous",
    ]
    source = pd.read_parquet(manifest, columns=columns).drop_duplicates("state_key", keep="first")
    source["state_key"] = source["state_key"].astype(str)
    return source.set_index("state_key")


def _importance(selected: pd.DataFrame, manifest: pd.DataFrame, pareto_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    facilities = managed_facility_ids()
    by_key = manifest.drop_duplicates(["state_key", "candidate_action_sha256"], keep="first").set_index(["state_key", "candidate_action_sha256"])
    records: list[dict[str, object]] = []
    cooccurrence: list[dict[str, object]] = []
    for state in selected.itertuples(index=False):
        valid = pareto_rows[(pareto_rows.state_key == state.state_key) & pareto_rows.actual_safe.eq(True) & pareto_rows.safe_tfv_improving.eq(True)]
        basis = "safe_tfv_improving"
        if valid.empty:
            valid = pareto_rows[pareto_rows.state_key == state.state_key]
            basis = "all_authoritative_candidates_no_safe_improving"
        changed: list[tuple[str, float, float]] = []
        changed_sets: list[set[str]] = []
        for row in valid.drop_duplicates(["state_key", "candidate_action_sha256"]).itertuples(index=False):
            key = (str(row.state_key), str(row.candidate_action_sha256))
            if key not in by_key.index:
                continue
            manifest_row = by_key.loc[key]
            action = _json(manifest_row.action_candidate_readback)
            history = _json(manifest_row.history_actions_readback)
            current = history[-1] if history.ndim == 2 else history
            first = action[:3] if action.ndim == 2 else action.reshape(1, -1)
            delta = np.max(np.abs(first - current[None, :]), axis=0)
            changed_set = {facilities[index] for index, value in enumerate(delta[: len(facilities)]) if float(value) > 1.0e-6}
            changed_sets.append(changed_set)
            for index, value in enumerate(delta[: len(facilities)]):
                if float(value) > 1.0e-6:
                    changed.append((facilities[index], float(row.tfv_reduction_pct), float(row.pfv_excess_m3)))
        pair_counts = Counter(pair for values in changed_sets for pair in combinations(sorted(values), 2))
        quad_counts = Counter(quad for values in changed_sets for quad in combinations(sorted(values), 4))
        for pair, count in pair_counts.items():
            cooccurrence.append({"state_key": state.state_key, "event_id": state.event_id, "load_regime": state.load_regime, "order": 2, "actuators": "|".join(pair), "count": int(count)})
        for quad, count in quad_counts.items():
            cooccurrence.append({"state_key": state.state_key, "event_id": state.event_id, "load_regime": state.load_regime, "order": 4, "actuators": "|".join(quad), "count": int(count)})
        pair_by_facility = Counter(item for pair, count in pair_counts.items() for item in pair for _ in range(count))
        quad_by_facility = Counter(item for quad, count in quad_counts.items() for item in quad for _ in range(count))
        for facility in facilities:
            values = [(gain, excess) for item, gain, excess in changed if item == facility]
            gains = [x[0] for x in values]
            excess = [x[1] for x in values]
            records.append({
                "state_key": state.state_key,
                "event_id": state.event_id,
                "load_regime": state.load_regime,
                "facility_id": facility,
                "safe_improving_change_count": len(values),
                "frequency": len(values) / max(1, len(valid)),
                "median_tfv_gain_pct": float(np.median(gains)) if gains else np.nan,
                "best_tfv_gain_pct": float(np.max(gains)) if gains else np.nan,
                "median_pfv_excess_m3": float(np.median(excess)) if excess else np.nan,
                "pair_cooccurrence_count": int(pair_by_facility[facility]),
                "quad_cooccurrence_count": int(quad_by_facility[facility]),
                "binary": facility in BINARY,
                "importance_basis": basis,
            })
    out = pd.DataFrame(records)
    out["rank"] = out.groupby("state_key")["frequency"].rank(method="first", ascending=False)
    return out.sort_values(["state_key", "rank", "facility_id"], kind="stable").drop(columns="rank"), pd.DataFrame(cooccurrence)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pareto-rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    states = pd.read_csv(args.states_csv)
    selected = select_fast_states(states)
    expected = {"LOW_LOAD": 2, "MODERATE_LOAD": 3, "NEAR_CAPACITY": 1, "SEVERE_OVERLOAD": 2}
    if selected.groupby("load_regime").size().to_dict() != expected:
        raise RuntimeError(f"unexpected fast plan counts: {selected.groupby('load_regime').size().to_dict()}")
    source = _source_rows(args.manifest)
    for row in selected.itertuples(index=False):
        if row.state_key not in source.index:
            raise RuntimeError(f"selected state missing from manifest: {row.state_key}")
    selected = selected.merge(source.reset_index(), on="state_key", how="left", suffixes=("", "_source"))
    selected["prefix_state_sha256"] = selected.apply(_prefix_hash, axis=1)
    selected["window_start_offset_min"] = 10.0
    selected["plan_scope"] = "FAST_DIRECT_SWMM_CONTROL_POTENTIAL_SCREEN_V1"
    selected["selection_uses_direct_results"] = False
    selected = selected.drop(columns=["history_depth", "history_actions_readback", "rainfall_forecast"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_columns = ["state_key", "candidate_action_sha256", "action_candidate_readback", "history_actions_readback"]
    manifest = pd.read_parquet(args.manifest, columns=manifest_columns)
    pareto = pd.read_csv(args.pareto_rows)
    importance, cooccurrence = _importance(selected, manifest, pareto)
    importance_path = args.output_dir / "AUTHORITATIVE_ACTUATOR_IMPORTANCE.csv"
    importance.to_csv(importance_path, index=False)
    cooccurrence_path = args.output_dir / "AUTHORITATIVE_ACTUATOR_COOCCURRENCE.csv"
    cooccurrence.to_csv(cooccurrence_path, index=False)
    chosen_assets: list[dict[str, object]] = []
    for state_key, group in importance.groupby("state_key", sort=False):
        continuous = group[~group.binary].sort_values(["frequency", "best_tfv_gain_pct", "facility_id"], ascending=[False, False, True], na_position="last", kind="stable").head(6)
        binary = group[group.binary & (group.safe_improving_change_count >= 2)].sort_values(["frequency", "best_tfv_gain_pct", "facility_id"], ascending=[False, False, True], kind="stable")
        chosen_assets.append({
            "state_key": state_key,
            "continuous_actuators": "|".join(continuous.facility_id.astype(str)),
            "binary_actuators": "|".join(binary.facility_id.astype(str)),
            "asset_selection_basis": "top_authoritative_frequency_then_tfv_gain",
        })
    selected = selected.merge(pd.DataFrame(chosen_assets), on="state_key", how="left")
    plan_path = args.output_dir / "FAST_DIRECT_STATE_PLAN.csv"
    selected.to_csv(plan_path, index=False)
    plan_lock = {
        "lock_id": "V42_FAST_DIRECT_STATE_PLAN_LOCK_V1",
        "development_only": True,
        "new_swmm_started": False,
        "plan_sha256": _sha256(plan_path),
        "states_source_sha256": _sha256(args.states_csv),
        "manifest_sha256": _sha256(args.manifest),
        "pareto_rows_sha256": _sha256(args.pareto_rows),
        "selected_states": _json_records(selected, ["state_key", "event_id", "rainfall_sha256", "checkpoint_min", "prefix_state_sha256", "load_regime", "oracle_tfv_reduction_pct", "selection_reason", "window_start_offset_min"]),
        "selection_uses_direct_results": False,
    }
    (args.output_dir / "FAST_DIRECT_STATE_PLAN_LOCK.json").write_text(json.dumps(plan_lock, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"plan": str(plan_path), "plan_sha256": plan_lock["plan_sha256"], "importance": str(importance_path), "cooccurrence": str(cooccurrence_path), "selected": int(len(selected)), "new_swmm_started": False}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
