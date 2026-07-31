from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sewerrtc.v4.v42_fasttrack import (
    CONTRACT_ID,
    diagnose_learning_curve,
    evaluate_stage_payload,
    select_fasttrack_core,
)


def _fixture_tables(tmp_path: Path) -> tuple[Path, Path]:
    physical_rows = []
    case_rows = []
    for event_idx in range(10):
        for case_idx in range(2):
            ids = []
            for role_idx, role in enumerate(
                ("candidate", "no_control", "dynamic_internal", "hold_previous")
            ):
                pid = f"p-{event_idx}-{case_idx}-{role_idx}"
                ids.append(pid)
                physical_rows.append(
                    {
                        "physical_identity_sha256": pid,
                        "detail_path": str(tmp_path / f"{pid}.csv"),
                        "branch_role": role,
                    }
                )
            case_rows.append(
                {
                    "case_uid": f"case-{event_idx}-{case_idx}",
                    "event_id": f"event-{event_idx}",
                    "rainfall_sha256": f"rain-{event_idx}",
                    "checkpoint_min": 100.0 + 10.0 * case_idx,
                    "branch_physical_ids": json.dumps(ids),
                    "four_reference_complete": True,
                    "core_trajectory_targets": True,
                    "classification": "PARTIAL_AUX_REUSE",
                    "source_role": "development",
                    "domain_id": "target_no_dwf",
                }
            )
    physical = tmp_path / "physical.parquet"
    cases = tmp_path / "cases.csv"
    pd.DataFrame(physical_rows).to_parquet(physical, index=False)
    pd.DataFrame(case_rows).to_csv(cases, index=False)
    return physical, cases


def test_fasttrack_core_is_small_and_rainfall_isolated(tmp_path: Path) -> None:
    physical, cases = _fixture_tables(tmp_path)
    result = select_fasttrack_core(
        physical_inventory=physical,
        case_inventory=cases,
        output_dir=tmp_path / "core",
        max_events=4,
        cases_per_event=1,
        seed=7,
    )
    selected = pd.read_csv(result.case_manifest)
    selected_physical = pd.read_parquet(result.physical_manifest)
    assert result.selected_events == 4
    assert result.selected_cases == 4
    assert len(selected_physical) == 16
    assert selected.groupby("fasttrack_group")["fasttrack_split"].nunique().max() == 1


def test_learning_curve_distinguishes_data_limited_from_plateau() -> None:
    data_limited = diagnose_learning_curve(
        [
            {"train_groups": 4, "train_score": 0.90, "val_score": 0.45},
            {"train_groups": 8, "train_score": 0.91, "val_score": 0.53},
            {"train_groups": 12, "train_score": 0.91, "val_score": 0.59},
        ],
        score_key="val_score",
        pass_threshold=0.70,
    )
    plateau = diagnose_learning_curve(
        [
            {"train_groups": 4, "train_score": 0.48, "val_score": 0.45},
            {"train_groups": 8, "train_score": 0.49, "val_score": 0.46},
            {"train_groups": 12, "train_score": 0.49, "val_score": 0.46},
        ],
        score_key="val_score",
        pass_threshold=0.70,
    )
    assert data_limited == "data_limited_expand_targeted_evidence"
    assert plateau == "model_or_target_limited_do_not_bulk_expand"


def test_step2_gate_is_development_only_and_fail_closed() -> None:
    payload = {
        "contract_id": CONTRACT_ID,
        "stage": "step2_surrogate",
        "status": "pass",
        "development_only": True,
        "formal_authorization": False,
        "metrics": {
            "pfv_direction_accuracy": 0.75,
            "tfv_direction_accuracy": 0.68,
            "peak_direction_accuracy": 0.66,
            "safe_candidate_recall": 0.76,
            "false_safe_rate": 0.10,
        },
    }
    decision = evaluate_stage_payload("step2_surrogate", payload)
    assert decision.passed

    unsafe = dict(payload)
    unsafe["formal_authorization"] = True
    decision = evaluate_stage_payload("step2_surrogate", unsafe)
    assert not decision.passed
    assert "fasttrack_must_not_authorize_formal" in decision.reasons
