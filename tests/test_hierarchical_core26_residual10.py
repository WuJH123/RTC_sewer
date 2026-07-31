from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _actuators() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "actuator_id": [f"C{i:02d}" for i in range(26)] + [f"R{i:02d}" for i in range(10)],
            "is_legacy_v8": [True] * 26 + [False] * 10,
            "link_type": ["orifice"] * 36,
            "storage_control_type": [""] * 36,
        }
    )


def test_strict_preflight_rejects_false_residual_gate(tmp_path: Path) -> None:
    from sewerrtc.control.hierarchical_core26_residual10 import build_strict_preflight

    core_path = tmp_path / "core.json"
    core_path.write_text(json.dumps({"templates": []}), encoding="utf-8")
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"pt")
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"validation_gate_passed": False}), encoding="utf-8")
    uncertainty_path = tmp_path / "uncertainty.json"
    uncertainty_path.write_text("{}", encoding="utf-8")
    guard_path = tmp_path / "guard.csv"
    guard_path.write_text("actuator_id,allowed\nR00,true\n", encoding="utf-8")

    report = build_strict_preflight(
        cfg={
            "controller": {
                "mode": "hierarchical_core26_residual10",
                "temporal_joint": {
                    "prediction_horizon_min": 120,
                    "move_horizon_min": 30,
                    "safety": {"event_pfv_budget_enabled": True},
                    "hierarchical": {
                        "core26_policy_path": str(core_path),
                        "residual10_model_path": str(model_path),
                        "residual10_model_report": str(report_path),
                        "uncertainty_model_path": str(uncertainty_path),
                        "empirical_guard_path": str(guard_path),
                        "residual_actuator_ids": [f"R{i:02d}" for i in range(10)],
                    },
                },
            }
        },
        actuators=_actuators(),
        project_root=tmp_path,
    )

    assert report["passed"] is False
    assert "residual_model_gate_passed" in report["failed_checks"]


def test_residual_candidates_do_not_modify_core26_columns() -> None:
    from sewerrtc.control.hierarchical_core26_residual10 import assert_residual_only_changes_residual_columns

    ids = [f"C{i:02d}" for i in range(26)] + [f"R{i:02d}" for i in range(10)]
    core = np.full((6, 36), 0.5, dtype=np.float32)
    residual = core.copy()
    residual[:, ids.index("R03")] += 0.1

    assert_residual_only_changes_residual_columns(
        core,
        residual,
        canonical_action_ids=ids,
        residual_actuator_ids=[f"R{i:02d}" for i in range(10)],
    )

    bad = residual.copy()
    bad[0, ids.index("C05")] += 0.1
    try:
        assert_residual_only_changes_residual_columns(
            core,
            bad,
            canonical_action_ids=ids,
            residual_actuator_ids=[f"R{i:02d}" for i in range(10)],
        )
    except AssertionError as exc:
        assert "core26_modified" in str(exc)
    else:
        raise AssertionError("expected a core26 modification to be rejected")


def test_event_pfv_budget_persists_across_decisions() -> None:
    from sewerrtc.control.temporal_joint_36_controller import TemporalJoint36Controller
    from sewerrtc.control.temporal_joint_candidate_search import TemporalJointCandidateConfig
    from sewerrtc.control.temporal_joint_safety import JointCandidatePrediction, JointSafetyConfig

    class FakePredictor:
        def predict_many(self, **kwargs):  # pragma: no cover - this test uses budget helpers only
            raise AssertionError("not used")

    controller = TemporalJoint36Controller(
        actuators=_actuators(),
        predictor=FakePredictor(),
        candidate_config=TemporalJointCandidateConfig(horizon_steps=6),
        safety_config=JointSafetyConfig(
            event_pfv_budget_enabled=True,
            event_pfv_abs_margin_m3=100.0,
            event_pfv_rel_margin=0.005,
            uncertainty_z=0.0,
        ),
        prediction_horizon_steps=24,
    )

    initial = controller._event_budget_state(reference_pfv=10_000.0)
    assert initial["event_pfv_budget_remaining"] == 100.0

    controller._commit_event_pfv_cost(
        JointCandidatePrediction(
            "safe_but_costly", 48.0, -1000.0, -1.0, 0.0, 0.0, 0.0, 1, 0.1, 0
        )
    )
    after_commit = controller._event_budget_state(reference_pfv=10_000.0)
    assert after_commit["event_pfv_budget_committed_cost"] == 48.0
    assert after_commit["event_pfv_budget_remaining"] == 52.0

    controller.decision_index = 1
    controller._advance_event_budget()
    next_step = controller._event_budget_state(reference_pfv=10_000.0)
    assert 0.0 < next_step["event_pfv_budget_realized_cost"] < 48.0
    assert next_step["event_pfv_budget_committed_cost"] < 48.0
    assert next_step["event_pfv_budget_remaining"] == 52.0

    controller._commit_event_pfv_cost(
        JointCandidatePrediction("no_control", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0)
    )
    no_control_state = controller._event_budget_state(reference_pfv=10_000.0)
    assert no_control_state["event_pfv_budget_remaining"] == next_step["event_pfv_budget_remaining"]
