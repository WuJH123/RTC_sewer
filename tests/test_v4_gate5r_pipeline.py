from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sewerrtc.prompt3.gate5r_pipeline import (
    EXIT_BLOCKED,
    EXIT_PASS,
    EXIT_SCIENTIFIC_FAIL,
    accounting_is_closed,
    audit_contract_values,
    build_pilot_plan,
    build_formal_1600_plan,
    classify_candidate_result,
    output_roots_are_isolated,
    pending_case_ids,
    post_decision_readback_mask,
    select_pending_plan,
    reference_cache_key,
    gate_exit_code,
    action_authority_reference_name,
    rebuild_run_manifest_from_completions,
    reference_cache_is_ready,
    safe_repeat_noise_ranges,
    canary_gate_status,
    confirmed_flat_fraction_is_in_range,
)


def test_contract_audit_requires_1600_ten_minute_control_and_k8() -> None:
    audit = audit_contract_values(
        recovery_contract={
            "control_step_sec": 300,
            "control_interval_sec": 600,
            "prediction_horizon_min": 120,
            "prediction_horizon_steps": 12,
            "max_k": 8,
            "reference_roles": {
                "PFV": {"primary": "No-control"},
                "TFV": {"primary": "Dynamic Internal"},
                "Peak": {"primary": "Dynamic Internal"},
            },
            "kpi_definitions": {"Peak": {"unit": "m3/s"}},
        },
        dataset_contract={"version": "1.2.0"},
        v4_config={"v4": {"aug1": {"effective_target": 1600}, "adaptive_k": {"max_k": 8}}},
        facility_ids=[f"f{i}" for i in range(36)],
    )

    assert audit["status"] == "pass"
    assert all(audit["checks"].values())


def test_contract_audit_blocks_lowered_sample_target() -> None:
    audit = audit_contract_values(
        recovery_contract={
            "control_step_sec": 300,
            "control_interval_sec": 600,
            "prediction_horizon_min": 120,
            "prediction_horizon_steps": 12,
            "max_k": 8,
            "reference_roles": {
                "PFV": {"primary": "No-control"},
                "TFV": {"primary": "Dynamic Internal"},
                "Peak": {"primary": "Dynamic Internal"},
            },
            "kpi_definitions": {"Peak": {"unit": "m3/s"}},
        },
        dataset_contract={"version": "1.2.0"},
        v4_config={"v4": {"aug1": {"effective_target": 1000}, "adaptive_k": {"max_k": 8}}},
        facility_ids=[f"f{i}" for i in range(36)],
    )

    assert audit["status"] == "blocked"
    assert not audit["checks"]["accepted_target_1600"]


def test_gate_exit_codes_are_fail_closed() -> None:
    assert gate_exit_code("pass") == EXIT_PASS
    assert gate_exit_code("blocked") == EXIT_BLOCKED
    assert gate_exit_code("scientific_fail") == EXIT_SCIENTIFIC_FAIL


def test_accounting_identity_is_exact() -> None:
    assert accounting_is_closed(10, 4, 2, 3, 1)
    assert not accounting_is_closed(10, 4, 2, 3, 0)


def test_formal_plan_has_1600_cases_and_event_level_splits() -> None:
    events = pd.DataFrame(
        {
            "event_id": [f"event-{index:03d}" for index in range(80)],
            "eligible": [True] * 80,
        }
    )

    plan, partition = build_formal_1600_plan(events, seed=20260726)

    assert len(plan) == 1600
    assert partition["split"].value_counts().to_dict() == {
        "train": 48,
        "model_validation": 8,
        "challenge": 8,
        "reserve": 16,
    }
    assert not plan.groupby("event_id")["split"].nunique().gt(1).any()
    assert plan.groupby("event_id").size().eq(25).all()


def test_science_gate_keeps_zero_margin_separate_from_material_benefit() -> None:
    neutral = classify_candidate_result(0.0, -1.0, 0.0, action_cost=1.0)
    material = classify_candidate_result(0.0, -30.0, 0.0, action_cost=10.0)

    assert neutral["joint_noninferior"]
    assert not neutral["materially_beneficial"]
    assert material["materially_beneficial"]


def test_pilot_plan_requires_four_responsive_and_one_flat_probe_per_event() -> None:
    rows = []
    for event_index in range(8):
        for checkpoint_index in range(4):
            rows.append(
                {
                    "event_id": f"event-{event_index}",
                    "checkpoint_min": checkpoint_index * 10.0,
                    "opportunity_class": "responsive",
                    "opportunity_score": 1.0 - checkpoint_index / 10,
                }
            )
        rows.append(
            {
                "event_id": f"event-{event_index}",
                "checkpoint_min": 50.0,
                "opportunity_class": "flat",
                "opportunity_score": 0.0,
            }
        )

    plan = build_pilot_plan(pd.DataFrame(rows))

    assert len(plan) == 40
    assert plan.groupby("event_id").size().eq(5).all()
    assert (
        plan.groupby("event_id")["checkpoint_role"]
        .apply(lambda values: (values == "flat_action_probe").sum())
        .eq(1)
        .all()
    )


def test_resume_and_reference_cache_are_stable_and_output_isolated(tmp_path: Path) -> None:
    first = reference_cache_key("event-a", 60.0, "contract")
    second = reference_cache_key("event-a", 60.0, "contract")
    changed = reference_cache_key("event-a", 70.0, "contract")

    assert first == second
    assert first != changed
    assert pending_case_ids(["a", "b", "c"], ["a", "c"]) == ["b"]
    assert output_roots_are_isolated(
        tmp_path / "gate5r_informative_v1", tmp_path / "legacy_gate5"
    )


def test_resume_limit_selects_next_pending_cases_not_plan_prefix() -> None:
    plan = pd.DataFrame({"case_id": ["a", "b", "c", "d"]})

    first = select_pending_plan(plan, completed_case_ids=set(), limit=2)
    second = select_pending_plan(plan, completed_case_ids={"a", "b"}, limit=2)

    assert first["case_id"].tolist() == ["a", "b"]
    assert second["case_id"].tolist() == ["c", "d"]


def test_action_authority_uses_frozen_hold_anchor_not_dynamic_internal() -> None:
    assert action_authority_reference_name() == "hold_previous"


def test_empty_accepted_sample_audit_has_finite_repeat_ranges() -> None:
    assert safe_repeat_noise_ranges(pd.DataFrame()) == {
        "pfv_m3": 0.0,
        "tfv_m3": 0.0,
        "peak_m3s": 0.0,
    }
    assert canary_gate_status(pass_gate=False, accepted=0) == "incomplete"
    assert gate_exit_code(
        canary_gate_status(pass_gate=False, accepted=0)
    ) == 3


def test_readback_mask_uses_stable_midpoint_after_native_rule_evaluation() -> None:
    elapsed = pd.Series([360.0, 365.0, 370.0, 375.0, 475.0, 480.0])

    mask = post_decision_readback_mask(
        elapsed,
        checkpoint_min=360.0,
        decision_interval_min=10.0,
        sample_interval_min=5.0,
        horizon_min=120.0,
    )

    assert elapsed[mask].tolist() == [365.0, 375.0, 475.0]


def test_canary_flat_control_is_measured_over_accepted_samples() -> None:
    samples = pd.DataFrame(
        {"confirmed_flat": [True] * 8 + [False] * 72}
    )

    assert confirmed_flat_fraction_is_in_range(samples, 0.10, 0.20)
    assert not confirmed_flat_fraction_is_in_range(samples, 0.11, 0.20)


def test_reference_cache_ready_requires_matching_contract_and_all_branches(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "reference"
    cache.mkdir()
    (cache / "completion.json").write_text(
        json.dumps({"status": "pass", "reference_contract_hash": "contract-a"}),
        encoding="utf-8",
    )
    for name in (
        "no_control_detail.csv",
        "dynamic_internal_rules_detail.csv",
        "hold_previous_detail.csv",
    ):
        (cache / name).write_text("elapsed_min\n0\n", encoding="utf-8")

    assert reference_cache_is_ready(cache, "contract-a")
    assert not reference_cache_is_ready(cache, "contract-b")
    (cache / "hold_previous_detail.csv").unlink()
    assert not reference_cache_is_ready(cache, "contract-a")


def test_resume_rebuilds_manifest_from_every_completion(tmp_path: Path) -> None:
    plan = pd.DataFrame({"case_id": ["a", "b", "c"]})
    run_root = tmp_path / "runs"
    for case_id in ("a", "b"):
        case_dir = run_root / case_id
        case_dir.mkdir(parents=True)
        (case_dir / "completion.json").write_text(
            json.dumps(
                {
                    "status": "pass",
                    "case_id": case_id,
                    "reference_contract_hash": "v3",
                }
            ),
            encoding="utf-8",
        )

    rebuilt = rebuild_run_manifest_from_completions(plan, run_root)
    current = rebuild_run_manifest_from_completions(
        plan, run_root, reference_contract_hash="v3"
    )
    stale = rebuild_run_manifest_from_completions(
        plan, run_root, reference_contract_hash="v4"
    )

    assert rebuilt["case_id"].tolist() == ["a", "b"]
    assert rebuilt["status"].tolist() == ["accepted", "accepted"]
    assert current["case_id"].tolist() == ["a", "b"]
    assert stale.empty
