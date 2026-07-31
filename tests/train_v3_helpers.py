"""Shared fixtures for the Train1600 V3 test suite (torch-free)."""
from __future__ import annotations

import hashlib

import pandas as pd

from sewerrtc.v4.event_splits import (
    build_event_usage_ledger,
    select_train1600_events,
)
from sewerrtc.v4.train1600_v3 import (
    apply_state_feasibility_scorer,
    build_train_round_rotation_v3,
    build_v3_role_plan,
)
from sewerrtc.v4.training_plan import build_train_checkpoint_catalog


FAMILIES = ("frontal", "convective", "typhoon")
RISKS = ("low", "medium", "high")
SPLIT_COUNTS = {
    "train": 48,
    "calibration": 8,
    "locked_validation": 8,
    "reserve": 16,
}

# Frozen online-feature scorer: opportunity_score 10 -> high, 8/9 ->
# boundary, 7 -> fallback-likely; the low_opportunity role always stays low.
SCORER = {
    "scorer_version": "v3",
    "feature": "opportunity_score",
    "online_features_only": True,
    "t_high": 9.5,
    "t_low": 7.5,
}


def make_standard_catalog(num_events: int) -> pd.DataFrame:
    rows = []
    for index in range(num_events):
        event = f"ev{index:03d}"
        sha = hashlib.sha256(event.encode()).hexdigest()
        for checkpoint_index in range(5):
            responsive = checkpoint_index < 4
            elapsed = (
                30 + checkpoint_index * 40
                if responsive
                else 240 + (index % 3) * 120
            )
            rows.append(
                {
                    "event_id": event,
                    "rainfall_sha256": sha,
                    "checkpoint_id": f"{event}_cp{checkpoint_index}",
                    "elapsed_min": elapsed,
                    "checkpoint_min": float(elapsed),
                    "anchor_action_json": "{}",
                    "checkpoint_role": (
                        "responsive" if responsive else "low_opportunity"
                    ),
                    "rainfall_phase": "rising",
                    "opportunity_score": float(10 - checkpoint_index),
                    "event_tier": "standard_4plus",
                    "checkpoint_state_source": "cold_start_prefix_replay",
                    "network_sha256": "net",
                    "config_sha256": "cfg",
                    "source_run_uuid": "uuid",
                    "rainfall_family": FAMILIES[index % 3],
                    "risk_level": RISKS[index % 3],
                }
            )
    return pd.DataFrame(rows)


def make_ledger(catalog: pd.DataFrame) -> pd.DataFrame:
    return build_event_usage_ledger(
        catalog[["event_id", "rainfall_sha256", "event_tier"]],
        scanned_event_ids=set(catalog["event_id"].astype(str)),
    )


def make_plan_chain(num_events: int = 80) -> dict:
    """Full deterministic V3 plan chain on a synthetic 80-event pool."""
    catalog = make_standard_catalog(num_events)
    ledger = make_ledger(catalog)
    selection = select_train1600_events(catalog, ledger, counts=SPLIT_COUNTS)
    train_catalog, reserve_catalog = build_train_checkpoint_catalog(
        catalog, selection
    )
    stratified = apply_state_feasibility_scorer(SCORER, train_catalog)
    role_plan = build_v3_role_plan(stratified)
    rotation = build_train_round_rotation_v3(train_catalog)
    return {
        "catalog": catalog,
        "ledger": ledger,
        "selection": selection,
        "train_catalog": train_catalog,
        "reserve_catalog": reserve_catalog,
        "stratified": stratified,
        "role_plan": role_plan,
        "rotation": rotation,
    }


def make_p3_evidence() -> tuple[dict, dict, dict]:
    """Frozen Gate P3 evidence mirroring the real headline numbers."""
    fallback_states = [
        {"event_id": f"fe{i % 7}", "checkpoint_id": f"fc{i:02d}"}
        for i in range(19)
    ]
    map_audit = {
        "checks": {
            "hard_authenticity_all_true": True,
            "all_catalog_states_classified": True,
            "accounting_closed": True,
            "missing_zero": True,
            "execution_unresolved_zero": True,
            "replay_success_rate_100": True,
            "actual_duplicates_zero": True,
        },
        "class_counts": {
            "joint_feasible_robust": 9,
            "joint_boundary_found": 4,
            "no_joint_found_under_budget": 18,
            "no_pfv_safe_found": 1,
        },
        "recall_report": {
            "event_support": 4,
            "fallback_only_states": fallback_states,
            "candidate_generator_state_recall": 1.0,
            "missed_feasible_states": [],
        },
        "p3_gate": {"unresolved_states": 0},
    }
    gate_verdict = {
        "status": "underpowered_validation",
        "metrics_used": {"ridge_rmse_improvement_vs_zero": 0.191},
        "checks": {"positive_control_replay_100pct": True},
    }
    dataset_audit = {
        "checks": {
            "hard_authenticity_100pct": True,
            "rainfall_sha_split_isolated": True,
            "no_eval_split_search_rows_trainable": True,
        },
        "headline": {"total_samples": 1132},
    }
    return map_audit, gate_verdict, dataset_audit
