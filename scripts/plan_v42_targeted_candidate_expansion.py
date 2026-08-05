"""Plan a bounded authoritative candidate-space expansion after PFV/TFV Pareto.

This script is read-only.  It selects high-information states from the
PFV_TFV_PARETO_FRONTIER_STATES/ROWS outputs and writes an auditable SWMM work
plan.  It does not start SWMM and does not modify the control contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


LOAD_ORDER = {
    "LOW_LOAD": 0,
    "MODERATE_LOAD": 1,
    "NEAR_CAPACITY": 2,
    "SEVERE_OVERLOAD": 3,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _prefix_hash(row: pd.Series) -> str:
    payload = {
        "rainfall_sha256": str(row["rainfall_sha256"]),
        "event_id": str(row["event_id"]),
        "checkpoint_min": float(row["checkpoint_min"]),
        "history_depth": _json_value(row["history_depth"]),
        "history_actions_readback": _json_value(row["history_actions_readback"]),
        "rainfall_forecast": _json_value(row["rainfall_forecast"]),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _margin_slice(
    frame: pd.DataFrame, relative_margin: float, absolute_margin_m3: float
) -> pd.DataFrame:
    selected = frame[
        np.isclose(
            frame["relative_margin_fraction"].astype(float),
            float(relative_margin),
        )
        & np.isclose(
            frame["absolute_margin_m3"].astype(float),
            float(absolute_margin_m3),
        )
    ].copy()
    if selected.empty:
        raise RuntimeError(
            "requested PFV margin is absent from Pareto state table: "
            f"delta={relative_margin}, B={absolute_margin_m3}"
        )
    if selected["state_key"].astype(str).duplicated().any():
        raise RuntimeError("Pareto state table has duplicate state rows at one margin")
    return selected


def _select_states(
    states: pd.DataFrame,
    *,
    states_per_regime: int,
    positive_controls: int,
) -> pd.DataFrame:
    work = states.copy()
    work["state_key"] = work["state_key"].astype(str)
    work["load_rank"] = work["load_regime"].map(LOAD_ORDER).fillna(99).astype(int)
    work["oracle_tfv_reduction_pct"] = pd.to_numeric(
        work["oracle_tfv_reduction_pct"], errors="coerce"
    )
    work["candidate_count"] = pd.to_numeric(
        work["candidate_count"], errors="coerce"
    ).fillna(0)
    work["actual_safe_candidate_count"] = pd.to_numeric(
        work["actual_safe_candidate_count"], errors="coerce"
    ).fillna(0)
    work["actual_safe_tfv_improving_count"] = pd.to_numeric(
        work["actual_safe_tfv_improving_count"], errors="coerce"
    ).fillna(0)

    selected_rows: list[dict[str, object]] = []
    selected_keys: set[str] = set()

    def add(frame: pd.DataFrame, reason: str, limit: int) -> None:
        for _, row in frame.head(max(0, int(limit))).iterrows():
            key = str(row["state_key"])
            if key in selected_keys:
                continue
            payload = row.to_dict()
            payload["selection_reason"] = reason
            selected_rows.append(payload)
            selected_keys.add(key)

    # Positive controls reveal which facilities, amplitudes and H3 profiles
    # already produce large authoritative gains.
    positive = work[work["oracle_tfv_reduction_pct"] >= 20.0].sort_values(
        ["oracle_tfv_reduction_pct", "actual_safe_candidate_count"],
        ascending=[False, False],
    )
    add(positive, "positive_control_ge20", positive_controls)

    # Low/moderate states are the strongest test of missing candidate coverage:
    # they should have controllability but currently fail the 20% target.
    for regime in ("LOW_LOAD", "MODERATE_LOAD"):
        subset = work[work["load_regime"] == regime].copy()
        subset["no_improving_candidate"] = (
            subset["actual_safe_tfv_improving_count"] <= 0
        ).astype(int)
        subset["oracle_missing_or_bad"] = (
            subset["oracle_tfv_reduction_pct"].isna()
            | (subset["oracle_tfv_reduction_pct"] < 20.0)
        ).astype(int)
        subset = subset.sort_values(
            [
                "oracle_missing_or_bad",
                "no_improving_candidate",
                "candidate_count",
                "actual_safe_candidate_count",
            ],
            ascending=[False, False, True, False],
        )
        add(subset, f"{regime.lower()}_candidate_gap", states_per_regime)

    # Near/severe states are diagnostic controls.  They establish whether the
    # oracle ceiling contracts under high loading after candidate expansion.
    for regime in ("NEAR_CAPACITY", "SEVERE_OVERLOAD"):
        subset = work[work["load_regime"] == regime].sort_values(
            ["actual_safe_candidate_count", "candidate_count"],
            ascending=[False, True],
        )
        add(subset, f"{regime.lower()}_physical_limit_probe", max(2, states_per_regime // 2))

    result = pd.DataFrame(selected_rows)
    if result.empty:
        raise RuntimeError("no targeted candidate-expansion states were selected")
    result["planned_candidate_budget"] = 384
    result["planned_single_magnitudes"] = "0.05;0.10;0.20"
    result["planned_profiles"] = "constant_h3;early_pulse;ramp_h3;release_h3"
    result["planned_pair_assets"] = 12
    result["planned_quad_assets"] = 12
    result["authoritative_swmm_required"] = True
    result["reuse_references"] = True
    result["new_reference_runs_required"] = False
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states-csv", type=Path, required=True)
    parser.add_argument("--rows-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--relative-margin", type=float, default=0.05)
    parser.add_argument("--absolute-margin-m3", type=float, default=100.0)
    parser.add_argument("--states-per-regime", type=int, default=6)
    parser.add_argument("--positive-controls", type=int, default=4)
    args = parser.parse_args()

    states = pd.read_csv(args.states_csv)
    rows = pd.read_csv(args.rows_csv)
    required_state_columns = {
        "relative_margin_fraction",
        "absolute_margin_m3",
        "state_key",
        "event_id",
        "rainfall_sha256",
        "load_regime",
        "candidate_count",
        "actual_safe_candidate_count",
        "actual_safe_tfv_improving_count",
        "oracle_tfv_reduction_pct",
    }
    missing = sorted(required_state_columns - set(states.columns))
    if missing:
        raise RuntimeError(f"Pareto state table is missing columns: {missing}")
    if "state_key" not in rows.columns or "candidate_action_sha256" not in rows.columns:
        raise RuntimeError("Pareto row table lacks state/action identity columns")

    margin_states = _margin_slice(
        states, args.relative_margin, args.absolute_margin_m3
    )
    plan = _select_states(
        margin_states,
        states_per_regime=args.states_per_regime,
        positive_controls=args.positive_controls,
    )

    action_examples = (
        rows[
            np.isclose(rows["relative_margin_fraction"].astype(float), args.relative_margin)
            & np.isclose(rows["absolute_margin_m3"].astype(float), args.absolute_margin_m3)
            & rows["state_key"].astype(str).isin(plan["state_key"].astype(str))
        ]
        .sort_values(["state_key", "tfv_reduction_pct"], ascending=[True, False])
        .groupby("state_key", sort=True)
        .head(3)
    )
    examples_by_state = {
        str(key): group[
            [
                "candidate_action_sha256",
                "tfv_reduction_pct",
                "pfv_excess_m3",
                "actual_safe",
                "safe_tfv_improving",
            ]
        ].to_dict(orient="records")
        for key, group in action_examples.groupby("state_key", sort=True)
    }
    plan["existing_top_action_examples_json"] = plan["state_key"].astype(str).map(
        lambda value: json.dumps(examples_by_state.get(value, []), separators=(",", ":"))
    )

    default_manifest = (
        PROJECT_ROOT
        / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/step2/FORMAL_F2_STEP2_CONTROL_CORE_MANIFEST.parquet"
    )
    parser_source_manifest = getattr(args, "source_manifest", None)
    source_manifest = Path(parser_source_manifest) if parser_source_manifest else default_manifest
    source = pd.read_parquet(
        source_manifest,
        columns=[
            "state_key",
            "event_id",
            "rainfall_sha256",
            "checkpoint_min",
            "history_depth",
            "history_actions_readback",
            "rainfall_forecast",
        ],
    ).drop_duplicates("state_key", keep="first")
    source = source.set_index(source["state_key"].astype(str))
    plan["checkpoint_min"] = plan["state_key"].astype(str).map(
        lambda value: float(source.loc[value, "checkpoint_min"])
    )
    if plan["checkpoint_min"].isna().any():
        raise RuntimeError("selected state lacks checkpoint metadata in source manifest")
    plan["prefix_state_sha256"] = plan["state_key"].astype(str).map(
        lambda value: _prefix_hash(source.loc[value])
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_dir / "TARGETED_CANDIDATE_EXPANSION_PLAN.csv"
    summary_path = args.output_dir / "TARGETED_CANDIDATE_EXPANSION_PLAN.json"
    plan.to_csv(plan_path, index=False)

    by_reason = {
        str(key): int(value)
        for key, value in plan["selection_reason"].value_counts().sort_index().items()
    }
    by_regime = {
        str(key): int(value)
        for key, value in plan["load_regime"].value_counts().sort_index().items()
    }
    payload = {
        "audit_id": "V42_TARGETED_CANDIDATE_EXPANSION_PLAN_V1",
        "read_only": True,
        "new_swmm_started": False,
        "relative_margin_fraction": float(args.relative_margin),
        "absolute_margin_m3": float(args.absolute_margin_m3),
        "input_states_at_margin": int(len(margin_states)),
        "selected_states": int(len(plan)),
        "selected_by_reason": by_reason,
        "selected_by_load_regime": by_regime,
        "candidate_recipe": {
            "max_candidates_per_state": 384,
            "continuous_magnitudes": [0.05, 0.10, 0.20],
            "profiles": [
                "constant_h3",
                "early_pulse",
                "ramp_h3",
                "release_h3",
            ],
            "binary_ids": ["ADD301.2", "ADD301.3"],
            "pair_asset_pool": 12,
            "quad_asset_pool": 12,
            "tail": "current_readback_H4_H12",
            "reference_reuse": True,
        },
        "next_stage": (
            "run only the planned candidate branches from identical checkpoints, "
            "reuse same-state No-control/Internal/Hold references, then rerun the "
            "authoritative PFV-TFV Pareto audit"
        ),
        "plan_csv": str(plan_path),
        "source_manifest": str(source_manifest),
    }
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), text=True
    ).strip()
    lock = {
        "lock_id": "V42_TARGETED_CANDIDATE_EXPANSION_PLAN_LOCK_V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "plan_sha256": _sha256(plan_path),
        "pareto_source_sha256": {
            "states_csv": _sha256(args.states_csv),
            "rows_csv": _sha256(args.rows_csv),
        },
        "source_manifest": str(source_manifest),
        "relative_margin_fraction": float(args.relative_margin),
        "absolute_margin_m3": float(args.absolute_margin_m3),
        "candidate_recipe": payload["candidate_recipe"],
        "selected_states": plan[
            [
                "state_key",
                "event_id",
                "rainfall_sha256",
                "checkpoint_min",
                "prefix_state_sha256",
                "selection_reason",
            ]
        ].to_dict(orient="records"),
    }
    lock_path = args.output_dir / "TARGETED_CANDIDATE_EXPANSION_LOCK.json"
    lock_path.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    payload["lock_path"] = str(lock_path)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
