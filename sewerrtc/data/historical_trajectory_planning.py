from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _sha1_text(values: Iterable[str]) -> str:
    payload = "\n".join(str(v) for v in values).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _event_policy_from_path(path: Path) -> tuple[str, str]:
    stem = path.stem
    if stem.endswith("_detail"):
        stem = stem[: -len("_detail")]
    event_id, sep, policy_id = stem.rpartition("__")
    if sep:
        return event_id, policy_id
    return stem, "unknown"


def _read_csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        line = handle.readline().strip()
    if not line:
        return []
    return [part.strip().strip('"') for part in line.split(",")]


def _project_label(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    for project in ("project6", "project5", "project4"):
        if project in parts:
            return project
    return "unknown"


def scan_trajectory_roots(
    roots: Iterable[str | Path],
    *,
    canonical_action_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Inventory trajectory detail CSVs without loading full time series.

    The output intentionally separates node and action signatures. GAT state
    reconstruction can mix policies only when the node signature matches;
    action-effect learning additionally needs canonical action coverage.
    """
    canonical = [str(x) for x in (canonical_action_ids or [])]
    canonical_set = set(canonical)
    rows = []
    for raw_root in roots:
        root = Path(raw_root)
        if not root.exists():
            rows.append(
                {
                    "trajectory_root": str(root),
                    "detail_file": "",
                    "exists": False,
                    "can_train_gat": False,
                    "action_learning_use": False,
                    "exclusion_reason": "missing_root",
                }
            )
            continue
        files = sorted(root.glob("*_detail.csv"))
        if not files and root.name != "trajectories":
            files = sorted((root / "trajectories").glob("*_detail.csv"))
        for path in files:
            try:
                cols = _read_csv_header(path)
            except Exception as exc:
                rows.append(
                    {
                        "trajectory_root": str(root),
                        "detail_file": str(path),
                        "exists": True,
                        "can_train_gat": False,
                        "action_learning_use": False,
                        "exclusion_reason": f"unreadable:{exc!r}",
                    }
                )
                continue
            event_id, policy_id = _event_policy_from_path(path)
            h_cols = [c for c in cols if c.startswith("h:")]
            a_cols = [c for c in cols if c.startswith("a:")]
            action_ids = [c.split(":", 1)[1] for c in a_cols]
            action_set = set(action_ids)
            canonical_coverage = (
                float(len(canonical_set.intersection(action_set)) / len(canonical_set))
                if canonical_set
                else 0.0
            )
            rows.append(
                {
                    "source_project": _project_label(path),
                    "trajectory_root": str(root),
                    "detail_file": str(path),
                    "exists": True,
                    "event_id": event_id,
                    "policy_id": policy_id,
                    "node_count": int(len(h_cols)),
                    "action_count": int(len(a_cols)),
                    "node_signature": _sha1_text(h_cols),
                    "action_signature": _sha1_text(a_cols),
                    "canonical_action_coverage": canonical_coverage,
                    "can_train_gat": bool(h_cols),
                    "can_train_action_sequence": bool(canonical_set and canonical_set.issubset(action_set)),
                    "is_no_control_reference": str(policy_id).lower() == "no_control",
                    "evidence_kind": _evidence_kind(path, policy_id),
                    "exclusion_reason": "",
                }
            )
    return pd.DataFrame(rows)


def _evidence_kind(path: Path, policy_id: str) -> str:
    text = str(path).lower()
    if "same_state" in text or "paired_cases" in text or "temporal_joint" in text:
        return "same_state_counterfactual_or_case"
    if str(policy_id).lower() == "no_control":
        return "passive_reference_trajectory"
    return "observational_policy_trajectory"


def build_gat_mixing_plan(inventory: pd.DataFrame, *, base_node_signature: str) -> pd.DataFrame:
    if inventory.empty:
        return inventory.copy()
    plan = inventory.copy()
    plan["gat_use"] = plan["can_train_gat"].fillna(False) & plan["node_signature"].astype(str).eq(str(base_node_signature))
    plan["gat_role"] = plan["policy_id"].map(
        lambda p: "reference_state" if str(p).lower() == "no_control" else "controlled_state"
    )
    plan["gat_exclusion_reason"] = ""
    plan.loc[~plan["can_train_gat"].fillna(False), "gat_exclusion_reason"] = "missing_hydraulic_state_columns"
    plan.loc[
        plan["can_train_gat"].fillna(False) & ~plan["node_signature"].astype(str).eq(str(base_node_signature)),
        "gat_exclusion_reason",
    ] = "node_signature_mismatch"
    return plan


def build_action_learning_plan(
    inventory: pd.DataFrame,
    *,
    canonical_action_ids: Iterable[str],
    horizon_steps: int,
) -> pd.DataFrame:
    canonical = [str(x) for x in canonical_action_ids]
    action_count = len(canonical)
    plan = inventory.copy()
    plan["action_learning_use"] = plan["can_train_action_sequence"].fillna(False)
    plan["action_tensor_shape"] = f"[H,{action_count}]"
    plan["horizon_steps"] = int(horizon_steps)
    plan["effect_label_role"] = plan.apply(_effect_role, axis=1)
    plan["action_learning_exclusion_reason"] = ""
    plan.loc[~plan["can_train_action_sequence"].fillna(False), "action_learning_exclusion_reason"] = (
        "canonical_action_ids_not_fully_covered"
    )
    return plan


def _effect_role(row: pd.Series) -> str:
    kind = str(row.get("evidence_kind", ""))
    policy = str(row.get("policy_id", "")).lower()
    if "same_state" in kind:
        return "same_state_candidate_vs_no_control_effect"
    if policy == "no_control":
        return "reference_dynamics_pretraining"
    return "observational_dynamics_pretraining"


def build_sensor_coverage_plan(ratios: Iterable[float], *, include_priority_nodes: bool = True) -> pd.DataFrame:
    rows = []
    for ratio in ratios:
        value = float(ratio)
        rows.append(
            {
                "sensor_ratio": value,
                "sensor_ratio_label": f"sr{value:.2f}".replace(".", "p"),
                "include_priority_nodes": bool(include_priority_nodes),
                "planned_stage": "gat_eval_or_training",
                "metric_requirements": "full_RMSE,full_NSE,priority_RMSE,priority_NSE,high_risk_period_error",
            }
        )
    return pd.DataFrame(rows)


def canonical_action_ids_from_order(path: str | Path) -> list[str]:
    table = pd.read_csv(path)
    if "actuator_id" not in table:
        raise ValueError(f"canonical action order is missing actuator_id: {path}")
    return table["actuator_id"].astype(str).tolist()


def node_signature_from_cache(path: str | Path) -> str:
    with np.load(path, allow_pickle=True) as data:
        if "node_cols" not in data.files:
            raise ValueError(f"transition cache is missing node_cols: {path}")
        node_cols = [str(x) for x in data["node_cols"].tolist()]
    return _sha1_text(node_cols)
