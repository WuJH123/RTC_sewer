from __future__ import annotations

import json

import pandas as pd

from scripts.prepare_v42_formal_f2 import _historical_contamination, _reserved
from sewerrtc.v4.formal_f2 import (
    ACCEPTANCE_GATE_COLUMNS,
    build_event_ledger,
    explicit_step1_roles,
    formal_step2_metadata_pool,
    source_acceptance_mask,
    split_overlap_matrix,
)


def _accepted(event: str, rain: str, state: str, action: str) -> dict:
    row = {
        "event_id": event,
        "rainfall_sha256": rain,
        "prefix_state_hash": state,
        "candidate_action_sha": action,
        "case_id": f"{event}-{action}",
    }
    for col in ACCEPTANCE_GATE_COLUMNS:
        row[col] = True
    return row


def test_formal_f2_group_splits_are_rainfall_isolated() -> None:
    source = pd.DataFrame([
        {"source_id": "train1600_v3", "rainfall_group_key": f"r{i}", "formal_step2_allowed": True,
         "step2_accepted_from_manifest": True, "raw_readmission_required": False}
        for i in range(70)
    ])
    inventory = pd.DataFrame([{"event_id": f"u{i}", "rainfall_sha256": f"u{i}"} for i in range(80)])
    ledger = build_event_ledger(source, inventory=inventory, seed=42)
    assert all(v == 0 for v in split_overlap_matrix(ledger).values())
    assert (ledger.formal_f2_role == "train").sum() == 70
    assert (ledger.formal_f2_role == "formal_blind").sum() == 24


def test_explicit_step1_roles_do_not_depend_on_domain_id() -> None:
    source = pd.DataFrame([
        {"source_id": "train1600_v3", "rainfall_group_key": f"r{i}", "formal_step2_allowed": True,
         "step2_accepted_from_manifest": True, "raw_readmission_required": False}
        for i in range(80)
    ])
    ledger = build_event_ledger(source, inventory=pd.DataFrame(), seed=42)
    windows = pd.DataFrame([
        {"split_group_key": f"r{i}", "detail_path": f"d{i}.csv", "anchor_min": 120.0,
         "physical_identity_sha256": f"p{i}", "domain_id": "legacy_unknown"}
        for i in range(80)
    ])
    out = explicit_step1_roles(windows, ledger, validation_fraction=0.15, split_seed=42)
    assert (out.step1_domain_role == "target_formal").all()
    assert (out.formal_split == "train").sum() == 68
    assert (out.formal_split == "validation").sum() == 12


def test_pilot_v3_admission_requires_training_flag_and_current_gates() -> None:
    good = _accepted("e1", "r1", "s1", "a1")
    bad = _accepted("e2", "r2", "s2", "a2")
    good["eligible_for_training"] = True
    bad["eligible_for_training"] = False
    mask = source_acceptance_mask(pd.DataFrame([good, bad]), "pilot_v3", {"step2_admission": "pilot_v3_training"})
    assert mask.tolist() == [True, False]


def test_pilot_v3_flag_without_current_gates_is_not_authorized() -> None:
    frame = pd.DataFrame([{"event_id": "e", "rainfall_sha256": "r", "eligible_for_training": True}])
    mask = source_acceptance_mask(frame, "pilot_v3", {"step2_admission": "pilot_v3_training"})
    assert mask.tolist() == [False]


def test_step2_metadata_deduplicates_same_rain_state_action() -> None:
    source = pd.DataFrame([
        {"source_id": "train1600_v3", "source_manifest": "a.csv", "source_manifest_sha256": "x",
         "source_row_number": i, "case_id": f"case{i}", "event_id": "e", "rainfall_group_key": "r",
         "checkpoint_min": 120.0, "state_key": "s", "action_key": "a", "formal_step2_allowed": True,
         "step2_accepted_from_manifest": True, "raw_readmission_required": False}
        for i in range(2)
    ])
    ledger = build_event_ledger(source, inventory=pd.DataFrame(), seed=42)
    out = formal_step2_metadata_pool(source, ledger)
    assert len(out) == 1
    assert out.iloc[0].formal_f2_role == "train"


def test_raw_readmission_keeps_rows_without_stale_state_action_identity() -> None:
    source = pd.DataFrame([
        {"source_id": "peak_boundary", "source_manifest": "peak.csv", "source_manifest_sha256": "x",
         "source_row_number": 7, "case_id": "peak-case-7", "event_id": "e", "rainfall_group_key": "r",
         "checkpoint_min": 180.0, "state_key": "", "action_key": "", "formal_step2_allowed": True,
         "step2_accepted_from_manifest": False, "raw_readmission_required": True}
    ])
    ledger = build_event_ledger(source, inventory=pd.DataFrame(), seed=42)
    out = formal_step2_metadata_pool(source, ledger)
    assert len(out) == 1
    assert out.iloc[0].raw_readmission_pending
    assert not out.iloc[0].training_admission_authorized


def test_step1_eligibility_alone_does_not_mark_rainfall_historically_consumed() -> None:
    rows = pd.DataFrame([
        {
            "source_id": "opportunity_pool",
            "rainfall_group_key": "untouched-rain",
            "historically_revealed": False,
            "formal_step1_allowed": True,
            "formal_step2_allowed": False,
            "step2_accepted_from_manifest": False,
            "raw_readmission_required": False,
        },
        {
            "source_id": "old_revealed",
            "rainfall_group_key": "revealed-rain",
            "historically_revealed": True,
            "formal_step1_allowed": True,
            "formal_step2_allowed": False,
            "step2_accepted_from_manifest": False,
            "raw_readmission_required": False,
        },
    ])
    out = _historical_contamination(rows)
    assert set(out.rainfall_group_key.astype(str)) == {"revealed-rain"}


def test_step2_training_population_is_contamination_even_if_registry_flag_is_false() -> None:
    rows = pd.DataFrame([
        {
            "source_id": "future_training_source",
            "rainfall_group_key": "train-rain",
            "historically_revealed": False,
            "formal_step1_allowed": False,
            "formal_step2_allowed": True,
            "step2_accepted_from_manifest": True,
            "raw_readmission_required": False,
        }
    ])
    out = _historical_contamination(rows)
    assert set(out.rainfall_group_key.astype(str)) == {"train-rain"}


def test_reserved_event_ids_are_mapped_to_rainfall_groups_via_event_inventory(tmp_path) -> None:
    root = tmp_path
    rain_dir = root / "outputs/rainfall_library_v8_storage_variablepump"
    rain_dir.mkdir(parents=True)
    (rain_dir / "rainfall_event_table.formal_adapter.json").write_text(
        json.dumps({"split": "formal_blind_v33", "event_ids": ["blind-1"]}), encoding="utf-8"
    )
    # Deliberately omit a rainfall SHA in the legacy table; the regression is
    # that the event inventory must still recover the reserved rainfall group.
    pd.DataFrame([{"event_id": "blind-1", "duration_min": 300}]).to_csv(
        rain_dir / "rainfall_event_table.csv", index=False
    )
    inventory = pd.DataFrame([{"event_id": "blind-1", "rainfall_sha256": "reserved-rain-sha"}])
    events, groups, audit = _reserved(root, pd.DataFrame(), inventory)
    assert events == {"blind-1"}
    assert groups == {"reserved-rain-sha"}
    assert audit["reserved_rainfall_group_count"] == 1
