from __future__ import annotations

import pandas as pd

from sewerrtc.v4.partial_audit import (
    HARD_AUTHENTICITY_COLUMNS,
    applicable_release_level,
    build_partial_bundle,
    progressive_release_gate,
)


def make_plan(total: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": [f"case{i}" for i in range(total)],
            "event_id": [f"e{i % 8}" for i in range(total)],
            "checkpoint_id": [f"c{i % 40}" for i in range(total)],
        }
    )


def make_completion(index: int, **overrides) -> dict:
    row = {
        "case_id": f"case{index}",
        "status": "pass",
        "branch_role": "candidate",
        "event_id": f"e{index % 8}",
        "checkpoint_id": f"c{index % 40}",
        "checkpoint_role": "responsive",
        "candidate_family": f"fam{index % 3}",
        "K": 2,
        "is_noop": False,
        "requested_schedule_sha256": f"req{index}",
        "actual_schedule_sha256": f"act{index}",
        "actual_action_distance": 1.0,
        "local_response_magnitude": 0.5,
        "delta_pfv_h120_vs_no_control": -5.0 - index,
        "delta_tfv_h120_vs_dynamic_internal": 10.0 + index,
        "delta_peak_h120_vs_dynamic_internal": 0.01 * index,
        "pfv_safe": index % 5 != 0,
        "tfv_improved": True,
        "peak_noninferior": True,
        "neutral": False,
        "joint_noninferior": index % 2 == 0,
        "materially_beneficial": False,
        "output_isolated": True,
    }
    row.update({column: True for column in HARD_AUTHENTICITY_COLUMNS})
    row.update(overrides)
    return row


def bundle_of(total_plan: int, completed: int, **overrides) -> dict:
    completions = pd.DataFrame(
        [make_completion(i, **overrides) for i in range(completed)]
    )
    return build_partial_bundle(make_plan(total_plan), completions)


def test_release_levels_map_1_16_40() -> None:
    assert applicable_release_level(0) == -1
    assert applicable_release_level(1) == 0
    assert applicable_release_level(15) == 0
    assert applicable_release_level(16) == 1
    assert applicable_release_level(39) == 1
    assert applicable_release_level(40) == 2


def test_level0_single_case_gate_passes_then_authorises_16() -> None:
    bundle = bundle_of(400, 1)
    gate = progressive_release_gate(bundle, level=0)

    assert gate["status"] == "pass"
    assert gate["checks"]["single_case_completed"]
    assert gate["checks"]["output_isolated"]
    assert gate["partial_only"] is True
    assert gate["full_gate_pass"] is False


def test_level0_blocks_when_single_case_rejected() -> None:
    bundle = bundle_of(400, 1, is_noop=True)
    gate = progressive_release_gate(bundle, level=0)

    assert gate["status"] == "blocked"
    assert not gate["checks"]["single_case_accepted"]


def test_level1_requires_16_authentic_and_two_families() -> None:
    bundle = bundle_of(400, 16)
    gate = progressive_release_gate(bundle, level=1)

    assert gate["status"] == "pass"
    assert gate["checks"]["actual_duplicates_zero"]
    assert gate["checks"]["candidate_families_ge_2"]
    assert gate["checks"]["deltas_not_all_constant"]


def test_level1_blocks_on_broken_process_pool_evidence() -> None:
    bundle = bundle_of(400, 16)
    gate = progressive_release_gate(
        bundle, level=1, evidence={"broken_process_pool": True}
    )

    assert gate["status"] == "blocked"
    assert not gate["checks"]["no_broken_process_pool"]


def test_level2_pilot40_gate_enforces_share_and_noise_rules() -> None:
    bundle = bundle_of(400, 40)
    gate = progressive_release_gate(
        bundle,
        level=2,
        config={
            "noise_floor": {
                "delta_pfv_h120_vs_no_control": 1.0,
                "delta_tfv_h120_vs_dynamic_internal": 1.0,
                "delta_peak_h120_vs_dynamic_internal": 0.001,
            }
        },
    )

    assert gate["status"] == "pass"
    assert gate["checks"]["responsive_local_response_rate_ge_70"]
    assert gate["checks"]["two_deltas_exceed_noise"]
    assert gate["checks"]["max_family_share_le_50"]
    assert gate["checks"]["max_event_share_le_25"]
    assert gate["checks"]["not_all_joint_beneficial"]
    assert gate["checks"]["not_all_unsafe"]


def test_level2_blocks_when_single_event_dominates() -> None:
    completions = pd.DataFrame(
        [
            make_completion(i, event_id="e0" if i < 20 else f"e{i % 8}")
            for i in range(40)
        ]
    )
    bundle = build_partial_bundle(make_plan(400), completions)
    gate = progressive_release_gate(bundle, level=2)

    assert not gate["checks"]["max_event_share_le_25"]
    assert gate["status"] == "blocked"


def test_any_level_blocks_on_hard_authenticity_violation() -> None:
    bundle = bundle_of(400, 16, no_hotstart=False)
    for level in (0, 1, 2):
        gate = progressive_release_gate(bundle, level=level)
        assert gate["status"] == "blocked"
        assert not gate["checks"]["hard_authenticity_clean"]
