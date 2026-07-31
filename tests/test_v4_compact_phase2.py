"""V4.1 Compact rescue Phase-2 tests (spec sections 12, 14-16, 19).

Covers the *brand-new independent* Calibration / Locked evaluation without any
SWMM: the fresh evaluation-split plan/audit on a synthetic Reserve ledger, the
compact calibration (never reads Locked / never updates weights), the read-only
Locked evaluation ops, the Predictive Generalization Gate verdict semantics
against the frozen contract, the one-shot Locked handler protection and the
Phase-2 pipeline wiring.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from v4_model_helpers import make_catalog, make_manifest
from sewerrtc.v4.pipeline import (
    ALL_STAGES,
    PREREQUISITES,
    STAGE_ARTIFACTS,
    build_registry,
)
from sewerrtc.v4.runtime import RuntimeOptions
from sewerrtc.v4.train_v4_loader import build_training_data
from sewerrtc.v4.train_v4_models import ModelConfig
from sewerrtc.v4.v4_compact_model_ops import CompactHeadSpecificModel
from sewerrtc.v4.v4_compact_eval_ops import (
    audit_evaluation_plan,
    calibrate_compact,
    evaluate_compact_locked,
    evaluate_predictive_gate,
    plan_fresh_evaluation_split,
)
from sewerrtc.v4 import pipeline_v4_compact_eval as p2

REPO = Path(__file__).resolve().parents[1]
GATE_CONTRACT = REPO / p2.GATE_CONTRACT_REL
DEAD_ZONES = {"pfv": 1.0, "tfv": 1.0, "peak": 0.001}

PHASE2_STAGES = (
    "PlanV4CompactCalibrationLockedV1",
    "AuditV4CompactEvaluationPlanV1",
    "RunV4CompactCalibrationV1",
    "BuildV4CompactCalibrationV1",
    "AuditV4CompactCalibrationV1",
    "RunV4CompactLockedV1",
    "BuildV4CompactLockedV1",
    "AuditV4CompactLockedV1",
    "CalibrateV4CompactV1",
    "EvaluateV4CompactLockedV1",
    "AuditV4PredictiveGeneralizationGateV1",
)


def _cfg():
    return ModelConfig().light()


def _eval_data(states_per_event: int = 6):
    """Train + fresh v4.1 calibration / locked splits (old splits relabelled)."""
    manifest = make_manifest(states_per_event=states_per_event)
    manifest["split"] = manifest["split"].replace(
        {"calibration": "v4.1_calibration", "locked_validation": "v4.1_locked"}
    )
    return build_training_data(
        manifest, make_catalog(manifest), require_count=None
    )


def _reserve_ledger(n_reserve: int = 16, n_used: int = 6) -> pd.DataFrame:
    rows = []
    for i in range(n_reserve):
        rows.append(
            {
                "event_id": f"rsv{i:02d}",
                "rainfall_sha256": f"{i:064x}",
                "assigned_split": "reserve",
            }
        )
    for i in range(n_used):
        rows.append(
            {
                "event_id": f"used{i:02d}",
                "rainfall_sha256": f"{(100 + i):064x}",
                "assigned_split": "train",
            }
        )
    return pd.DataFrame(rows)


# --- section 12: fresh evaluation split (Reserve only, frozen) ------------

def test_fresh_split_is_reserve_only_frozen_and_disjoint():
    ledger = _reserve_ledger()
    plan = plan_fresh_evaluation_split(ledger)
    assert plan["counts"] == {
        "v4.1_calibration": 4,
        "v4.1_locked": 8,
        "locked_accrual_reserve": 4,
    }
    all_events = [e for events in plan["splits"].values() for e in events]
    assert len(all_events) == len(set(all_events)) == 16
    # selection never reads old Locked and is frozen before any new label.
    assert plan["reads_old_locked_for_selection"] is False
    assert plan["frozen_before_any_new_label"] is True
    audit = audit_evaluation_plan(plan, ledger)
    assert audit["status"] == "pass"
    assert audit["checks"]["all_from_reserve"] is True
    assert audit["checks"]["not_selected_by_old_locked"] is True


def test_fresh_split_order_is_deterministic():
    ledger = _reserve_ledger()
    a = plan_fresh_evaluation_split(ledger)
    b = plan_fresh_evaluation_split(ledger.sample(frac=1.0, random_state=3))
    assert a["frozen_order"] == b["frozen_order"]
    assert a["splits"] == b["splits"]


def test_fresh_split_fails_closed_when_too_few_reserve():
    ledger = _reserve_ledger(n_reserve=10)
    try:
        plan_fresh_evaluation_split(ledger)
    except ValueError as exc:
        assert "reserve" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for insufficient reserve")


# --- section 14: calibrate on NEW calibration only ------------------------

def test_calibrate_never_reads_locked_or_updates_weights():
    data = _eval_data()
    model = CompactHeadSpecificModel(cfg=_cfg(), seeds=(0, 1)).fit(data)
    report = calibrate_compact(
        model, data, cfg=_cfg(), dead_zones=DEAD_ZONES,
        calibration_split="v4.1_calibration",
    )
    assert report["reads_locked"] is False
    assert report["updates_model_weights"] is False
    assert report["split_used"] == "v4.1_calibration"
    assert isinstance(report["disabled_probability_heads"], list)
    # one-sided conformal intervals were fit per continuous head.
    assert set(report["continuous_interval_calibration"]["intervals"]).issubset(
        {"pfv", "tfv", "peak"}
    )


# --- section 16: read-only Locked evaluation ------------------------------

def test_locked_evaluation_is_read_only_and_reports_r2():
    data = _eval_data()
    model = CompactHeadSpecificModel(cfg=_cfg(), seeds=(0, 1)).fit(data)
    calibration = calibrate_compact(
        model, data, cfg=_cfg(), dead_zones=DEAD_ZONES,
        calibration_split="v4.1_calibration",
    )
    report = evaluate_compact_locked(
        model, data, cfg=_cfg(), dead_zones=DEAD_ZONES,
        calibration=calibration, locked_split="v4.1_locked",
    )
    assert report["used_for_tuning"] is False
    assert report["split_used"] == "v4.1_locked"
    assert "pfv" in report["continuous"]
    assert "r2" in report["continuous"]["pfv"]


# --- section 15: Predictive Generalization Gate verdict semantics ---------

def _passing_report(**over):
    def cont(r2, imp):
        return {
            "r2": r2, "mae": 1.0, "beats_mean_baseline": True,
            "mae_improvement_vs_best_simple": imp,
            "sign_accuracy_outside_dead_zone": 0.7,
        }

    def cls(mcc, ap, ba, fsr, pos):
        return {
            "mcc": mcc, "average_precision": ap, "balanced_accuracy": ba,
            "false_safe_rate": fsr, "class_support": {"positive": pos}, "n": 20,
            "probability_head_disabled": False,
        }

    report = {
        "continuous": {
            "pfv": cont(0.5, 0.2), "tfv": cont(0.4, 0.2), "peak": cont(0.3, 0.05),
        },
        "classification": {
            "pfv_safe": cls(0.4, 0.8, 0.75, 0.10, 10),
            "peak_noninferior": cls(0.3, 0.7, 0.70, 0.15, 8),
        },
        "decision": {"top_k_feasible_recall": {"5": 0.9}, "states_with_feasible": 8},
    }
    report.update(over)
    return report


def test_gate_contract_keys_align_with_scorer():
    contract = json.loads(GATE_CONTRACT.read_text(encoding="utf-8"))
    assert contract["continuous"]["min_heads_improving"] == 2
    assert contract["classification"]["max_false_safe_rate"] == 0.20
    assert contract["ranking_and_decision"]["min_feasible_states"] == 5


def test_gate_verdict_pass():
    contract = json.loads(GATE_CONTRACT.read_text(encoding="utf-8"))
    verdict = evaluate_predictive_gate(_passing_report(), contract)
    assert verdict["status"] == "pass"
    assert verdict["authorizes_closed_loop"] is True


def test_gate_verdict_underpowered_triggers_accrual_not_change():
    contract = json.loads(GATE_CONTRACT.read_text(encoding="utf-8"))
    report = _passing_report(
        decision={"top_k_feasible_recall": {"5": 0.9}, "states_with_feasible": 2}
    )
    verdict = evaluate_predictive_gate(report, contract)
    assert verdict["status"] == "underpowered"
    assert verdict["authorizes_closed_loop"] is False


def test_gate_verdict_scientific_fail_on_negative_r2():
    contract = json.loads(GATE_CONTRACT.read_text(encoding="utf-8"))
    report = _passing_report()
    report["continuous"]["pfv"]["r2"] = -0.5
    verdict = evaluate_predictive_gate(report, contract)
    assert verdict["status"] == "scientific_fail"
    assert verdict["authorizes_closed_loop"] is False


# --- section 12/16: handlers (one-shot protection, plan freeze) -----------

def _handlers(tmp_path: Path):
    return p2.build_v4_compact_phase2_handlers(
        project_root=REPO, output_root=tmp_path / "out", config={}
    )


def test_plan_and_audit_handlers_materialize_executable_plans(tmp_path: Path, monkeypatch):
    out = tmp_path / "out"
    ledger_path = out / p2.EVENT_LEDGER_REL
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = _reserve_ledger()
    ledger.to_csv(ledger_path, index=False)
    reserve_rows = []
    for _, event in ledger[ledger["assigned_split"] == "reserve"].iterrows():
        for checkpoint in range(5):
            reserve_rows.append(
                {
                    "event_id": event["event_id"],
                    "rainfall_sha256": event["rainfall_sha256"],
                    "checkpoint_id": f"cp{checkpoint}",
                    "checkpoint_role": "responsive",
                    "checkpoint_min": checkpoint * 10,
                    "split": "reserve",
                    "predicted_stratum": "predicted_high_feasibility",
                    "anchor_action_json": "[]",
                }
            )
    reserve_path = out / "train1600_v3/planning/train_reserve_catalog_v3.csv"
    reserve_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(reserve_rows).to_csv(reserve_path, index=False)
    peak_path = out / "peak_boundary/peak_boundary_anchor_library.csv"
    peak_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"anchor_id": []}).to_csv(peak_path, index=False)

    def fake_materialize(primary, *_args, **_kwargs):
        rows = primary.copy()
        rows["sample_id"] = rows["case_id"]
        return rows, pd.DataFrame()

    def fake_branch_plan(candidates, **_kwargs):
        return pd.DataFrame(
            [
                {"sample_id": sample, "branch_role": role}
                for sample in candidates["sample_id"]
                for role in ("candidate", "no_control", "dynamic_internal_rules", "hold_previous")
            ]
        )

    monkeypatch.setattr(
        p2,
        "_facility_inputs",
        lambda *_: {"facility_ids": [], "facility_semantics": pd.DataFrame()},
    )
    monkeypatch.setattr(p2, "materialize_pilot_candidates", fake_materialize)
    monkeypatch.setattr(p2, "build_pilot_branch_plan", fake_branch_plan)
    handlers = _handlers(tmp_path)
    plan_result = handlers["PlanV4CompactCalibrationLockedV1"](
        RuntimeOptions(stage="PlanV4CompactCalibrationLockedV1")
    )
    assert plan_result.status == "pass"
    assert (out / p2.V4C_PLAN_FREEZE_REL).exists()
    assert (out / p2.V4C_LOCKED_PLAN_REL).exists()
    assert (out / p2.V4C_CAL_RUN_PLAN_REL).exists()
    assert (out / p2.V4C_LOCKED_RUN_PLAN_REL).exists()
    branch_plan = pd.read_csv(out / p2.V4C_BRANCH_PLAN_REL)
    assert len(branch_plan) == 4 * (100 + 200)
    assert not branch_plan.duplicated(["sample_id", "branch_role"]).any()
    consumption = json.loads(
        (out / p2.V4C_OLD_CONSUMPTION_REL).read_text(encoding="utf-8")
    )
    assert consumption["old_locked"]["eligible_for_v1_official_evaluation"] is False
    audit_result = handlers["AuditV4CompactEvaluationPlanV1"](
        RuntimeOptions(stage="AuditV4CompactEvaluationPlanV1")
    )
    assert audit_result.status == "pass"


def test_locked_one_shot_refuses_when_intent_exists(tmp_path: Path):
    out = tmp_path / "out"
    intent = out / p2.V4C_LOCKED_INTENT_REL
    intent.parent.mkdir(parents=True, exist_ok=True)
    intent.write_text("{}", encoding="utf-8")
    handlers = _handlers(tmp_path)
    result = handlers["EvaluateV4CompactLockedV1"](
        RuntimeOptions(stage="EvaluateV4CompactLockedV1")
    )
    assert result.status == "blocked"
    assert result.evidence["reason"] == "locked_evaluation_already_executed"


def test_locked_handler_missing_inputs_does_not_write_intent(tmp_path: Path):
    out = tmp_path / "out"
    handlers = _handlers(tmp_path)
    result = handlers["EvaluateV4CompactLockedV1"](
        RuntimeOptions(stage="EvaluateV4CompactLockedV1")
    )
    assert result.status == "incomplete"
    # one-shot must not be burned when inputs are absent.
    assert not (out / p2.V4C_LOCKED_INTENT_REL).exists()


# --- section 19: Phase-2 pipeline wiring ----------------------------------

def test_phase2_stages_registered_wired_and_have_artifacts():
    for stage in PHASE2_STAGES:
        assert stage in ALL_STAGES
        assert stage in STAGE_ARTIFACTS
        assert stage in PREREQUISITES
    # gate audit gates on the one-shot locked evaluation.
    assert PREREQUISITES["AuditV4PredictiveGeneralizationGateV1"] == (
        "EvaluateV4CompactLockedV1",
    )


def test_phase2_registry_builds_all_handlers(tmp_path: Path):
    registry = build_registry(
        project_root=REPO, output_root=tmp_path / "out", config={}
    )
    for stage in PHASE2_STAGES:
        result = registry.run(stage, RuntimeOptions(stage=stage))
        # Every Phase-2 stage is registered; with no upstream artifacts it
        # blocks on prerequisites / missing inputs, never crashes.
        assert result.exit_code != 0
