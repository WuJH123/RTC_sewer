"""Freeze current-generation F2 Calibration/Locked/Challenge/Test plans.

The split authority is the already-frozen rainfall-group ledger. Historical
labels from earlier Project6 runs are lineage metadata only. The scientific
requirement for this generation is straightforward: model-development rainfall
groups and the held-out evaluation groups are disjoint.

Checkpoint selection uses only rainfall/pre-control opportunity metadata. It
never uses Proposed/baseline outcomes or future hydraulic KPIs. The final stage
keeps the legacy filename/role ``formal_blind`` for pipeline compatibility, but
its evidence semantics are a current-generation held-out test, not a claim that
the rainfall has never appeared in any older Project6 experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.v4.formal_f2 import (
    DEFAULT_COUNTS,
    FORMAL_GENERATION_ID,
    canonical_rain_group,
    read_table,
    text,
)


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _first_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    lookup = {str(c).casefold(): str(c) for c in frame.columns}
    for name in names:
        if name.casefold() in lookup:
            return lookup[name.casefold()]
    return None


def _numeric(
    frame: pd.DataFrame, names: tuple[str, ...], default: float = np.nan
) -> pd.Series:
    col = _first_column(frame, names)
    if col is None:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce")


def _load_opportunity(root: Path) -> tuple[pd.DataFrame, str]:
    candidates = [
        root / "outputs/project6_dual_reference_v4/final_v4/opportunity_pool.csv",
        root
        / "outputs/project6_dual_reference_v4/final_v4/inventory/opportunity_pool.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path, low_memory=False), str(path)
    try:
        import subprocess

        proc = subprocess.run(
            [
                "rg",
                "--files",
                "-uu",
                "-g",
                "opportunity_pool.csv",
                str(root / "outputs/project6_dual_reference_v4/final_v4"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode in (0, 1):
            for line in proc.stdout.splitlines():
                path = Path(line.strip())
                if path.exists():
                    return pd.read_csv(path, low_memory=False), str(path)
    except FileNotFoundError:
        pass
    return pd.DataFrame(), ""


def _rain_map_from_inventory(inventory: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in inventory.to_dict("records"):
        event = text(row.get("event_id", row.get("rainfall_event_id", "")))
        group = canonical_rain_group(row)
        if event and group:
            if event in out and out[event] != group:
                raise RuntimeError(f"event_id maps to multiple rainfall groups: {event}")
            out[event] = group
    return out


def _event_by_rain_unique(rain_by_event: dict[str, str]) -> dict[str, str]:
    grouped: dict[str, list[str]] = {}
    for event, rain in rain_by_event.items():
        grouped.setdefault(rain, []).append(event)
    collision = {rain: values for rain, values in grouped.items() if len(values) > 1}
    if collision:
        raise RuntimeError(
            "rainfall group maps to multiple inventory event IDs; current evaluation requires one authoritative event identity: "
            f"{dict(list(collision.items())[:10])}"
        )
    return {rain: events[0] for rain, events in grouped.items()}


def _phase_label(row: pd.Series) -> str:
    for col in ("phase", "hydraulic_phase", "checkpoint_phase", "rainfall_phase"):
        if col in row.index and text(row[col]):
            return text(row[col]).casefold()
    return ""


def _select_checkpoints(
    frame: pd.DataFrame, n: int, seed: int, rain: str, *, minimum_checkpoint_min: float = 120.0
) -> list[dict[str, Any]]:
    if frame.empty or n <= 0:
        return []
    cp = _numeric(frame, ("checkpoint_min", "elapsed_min", "anchor_min"))
    frame = frame.loc[cp.notna()].copy()
    if frame.empty:
        return []
    frame["_checkpoint"] = cp.loc[frame.index].astype(float)
    frame = frame[frame["_checkpoint"] >= float(minimum_checkpoint_min)].copy()
    if frame.empty:
        return []
    score = _numeric(
        frame,
        ("opportunity_score", "joint_opportunity_score", "control_opportunity_score"),
        0.0,
    ).fillna(0.0)
    active = _numeric(
        frame, ("active_flow_signal", "flood_signal", "storage_signal"), 0.0
    ).fillna(0.0)
    frame["_score"] = score + 0.05 * active
    frame["_phase"] = frame.apply(_phase_label, axis=1)
    chosen: list[int] = []
    for token in ("rising", "peak", "recession", "high", "joint"):
        if len(chosen) >= n:
            break
        sub = frame[frame["_phase"].str.contains(token, na=False)]
        if not sub.empty:
            idx = sub.sort_values(
                ["_score", "_checkpoint"],
                ascending=[False, True],
                kind="mergesort",
            ).index[0]
            if int(idx) not in chosen:
                chosen.append(int(idx))
    remaining = [
        int(i)
        for i in frame.sort_values(
            ["_score", "_checkpoint"],
            ascending=[False, True],
            kind="mergesort",
        ).index
        if int(i) not in chosen
    ]
    remaining = sorted(
        remaining,
        key=lambda i: (
            hashlib.sha256(
                f"f2-eval-state:{seed}:{rain}:{float(frame.loc[i, '_checkpoint']):.6f}".encode()
            ).hexdigest(),
            -float(frame.loc[i, "_score"]),
        ),
    )
    chosen.extend(remaining[: max(0, n - len(chosen))])
    result = []
    for idx in chosen[:n]:
        row = frame.loc[idx]
        checkpoint = float(row["_checkpoint"])
        result.append(
            {
                "checkpoint_min": checkpoint,
                "phase": text(row.get("_phase", "")),
                "opportunity_score": float(row["_score"]),
                "required_history_start_min": checkpoint - 120.0,
                "required_simulation_end_min": checkpoint + 120.0,
                "selection_authority": "pre_control_opportunity_metadata_only",
            }
        )
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument(
        "--ledger",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/prepare/FORMAL_F2_EVENT_LEDGER.csv",
    )
    ap.add_argument(
        "--inventory",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/project6_dual_reference_v4/final_v4/inventory/event_inventory.csv",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/evaluation_plan",
    )
    ap.add_argument("--states-per-event", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ledger = read_table(args.ledger)
    if ledger.empty:
        raise ValueError("Formal F2 event ledger is empty")
    inventory = read_table(args.inventory) if args.inventory.exists() else pd.DataFrame()
    rain_by_event = _rain_map_from_inventory(inventory)
    event_by_rain = _event_by_rain_unique(rain_by_event)
    opportunity, opportunity_path = _load_opportunity(args.project_root)
    opp_event_col = (
        _first_column(opportunity, ("event_id", "rainfall_event_id"))
        if not opportunity.empty
        else None
    )
    opp_rain_col = (
        _first_column(
            opportunity,
            (
                "rainfall_sha256",
                "rainfall_fingerprint",
                "rainfall_group_key",
                "split_group_key",
            ),
        )
        if not opportunity.empty
        else None
    )

    train_groups = set(
        ledger.loc[
            ledger["formal_f2_role"].astype(str).eq("train"), "rainfall_group_key"
        ].astype(str)
    )
    rows: list[dict[str, Any]] = []
    for role in ("calibration", "locked_validation", "challenge", "formal_blind"):
        subset = ledger[ledger["formal_f2_role"].astype(str).eq(role)].copy()
        for _, item in subset.iterrows():
            rain = text(item.get("rainfall_group_key", ""))
            event = text(item.get("inventory_event_id", "")) or event_by_rain.get(rain, "")
            if not rain:
                continue
            if rain in train_groups:
                raise RuntimeError(f"evaluation rainfall overlaps current training: {rain}")
            if role == "formal_blind":
                checkpoints: list[dict[str, Any]] = []
            else:
                query = opportunity.iloc[0:0].copy()
                if not opportunity.empty:
                    if opp_rain_col is not None:
                        query = opportunity[
                            opportunity[opp_rain_col].astype(str).eq(rain)
                        ].copy()
                    if query.empty and event and opp_event_col is not None:
                        query = opportunity[
                            opportunity[opp_event_col].astype(str).eq(event)
                        ].copy()
                checkpoints = _select_checkpoints(
                    query, args.states_per_event, args.seed, rain
                )
            rows.append(
                {
                    "formal_generation_id": FORMAL_GENERATION_ID,
                    "formal_f2_role": role,
                    "evaluation_semantics": "current_generation_rainfall_group_holdout",
                    "rainfall_sha256": rain,
                    "event_id": event,
                    "rainfall_family": text(item.get("rainfall_family", "")),
                    "duration_min": _finite_or_none(item.get("duration_min")),
                    "checkpoints": checkpoints,
                    "current_generation_holdout": True,
                    "model_weight_training_allowed": False,
                    "calibration_parameter_fitting_allowed": role == "calibration",
                    "policy_lock_required_before_evaluation": role
                    in {"locked_validation", "challenge", "formal_blind"},
                    "post_evaluation_exclusion_allowed": False,
                    "selection_uses_control_outcome": False,
                    "historical_status_is_split_gate": False,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_json(
        args.output_dir / "FORMAL_F2_EVALUATION_PLAN.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    for role in ("calibration", "locked_validation", "challenge", "formal_blind"):
        data = [r for r in rows if r["formal_f2_role"] == role]
        (args.output_dir / f"{role}_plan.json").write_text(
            json.dumps(
                {
                    "formal_generation_id": FORMAL_GENERATION_ID,
                    "role": role,
                    "evaluation_semantics": "current_generation_rainfall_group_holdout",
                    "events": data,
                },
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

    counts = {
        role: int((table.formal_f2_role == role).sum())
        for role in ("calibration", "locked_validation", "challenge", "formal_blind")
    }
    checkpoint_counts = {
        role: int(
            sum(
                len(x)
                for x in table.loc[table.formal_f2_role == role, "checkpoints"]
            )
        )
        for role in ("calibration", "locked_validation", "challenge", "formal_blind")
    }
    audit = {
        "formal_generation_id": FORMAL_GENERATION_ID,
        "stage": "formal_f2_evaluation_plan",
        "status": "pass",
        "split_policy": "current_generation_rainfall_group_holdout",
        "historical_status_is_split_gate": False,
        "opportunity_pool_path": opportunity_path,
        "event_counts": counts,
        "checkpoint_counts": checkpoint_counts,
        "blind_has_preselected_control_states": bool(
            any(
                len(x)
                for x in table.loc[
                    table.formal_f2_role == "formal_blind", "checkpoints"
                ]
            )
        ),
        "locked_requires_policy_lock_before_evaluation": bool(
            table.loc[
                table.formal_f2_role == "locked_validation",
                "policy_lock_required_before_evaluation",
            ].all()
        )
        if not table.loc[table.formal_f2_role == "locked_validation"].empty
        else False,
        "current_training_overlap_count": int(
            len(
                train_groups
                & set(
                    table["rainfall_sha256"].astype(str)
                    if not table.empty
                    else []
                )
            )
        ),
        "selection_uses_control_outcome": False,
        "post_evaluation_exclusion_allowed": False,
    }
    required = {
        "calibration": DEFAULT_COUNTS["calibration"],
        "locked_validation": DEFAULT_COUNTS["locked_validation"],
        "challenge": DEFAULT_COUNTS["challenge"],
        "formal_blind": DEFAULT_COUNTS["formal_blind"],
    }
    if (
        any(counts[role] < n for role, n in required.items())
        or audit["blind_has_preselected_control_states"]
        or not audit["locked_requires_policy_lock_before_evaluation"]
        or audit["current_training_overlap_count"] != 0
    ):
        audit["status"] = "fail"
    (args.output_dir / "FORMAL_F2_EVALUATION_PLAN_AUDIT.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, allow_nan=False), flush=True)
    return 0 if audit["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
