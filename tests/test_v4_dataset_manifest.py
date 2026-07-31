from __future__ import annotations

import pandas as pd
import pytest

from sewerrtc.v4.dataset import build_branch_manifest, build_sample_manifest
from sewerrtc.v4.manifests import (
    accounting_summary,
    deduplicate_actual_schedules,
    sample_key,
    validate_sample_contract,
)


BRANCHES = (
    "candidate",
    "no_control",
    "dynamic_internal_rules",
    "hold_previous",
)


def make_branch_records(
    sample_ids: list[str],
    *,
    drop_reference_for: set[str] | None = None,
    actual_shas: dict[str, str] | None = None,
) -> pd.DataFrame:
    drop_reference_for = drop_reference_for or set()
    actual_shas = actual_shas or {}
    rows = []
    for sample_id in sample_ids:
        for branch in BRANCHES:
            if branch == "hold_previous" and sample_id in drop_reference_for:
                continue
            rows.append(
                {
                    "sample_id": sample_id,
                    "event_id": "e",
                    "checkpoint_id": "c",
                    "branch_role": branch,
                    "actual_schedule_sha256": actual_shas.get(
                        sample_id, f"act-{sample_id}"
                    ),
                }
            )
    return pd.DataFrame(rows)


def test_sample_key_uses_event_checkpoint_and_actual_schedule() -> None:
    assert sample_key("e", "c", "a") == "e|c|a"


def test_manifest_accounting_closes_and_actual_schedules_are_deduplicated() -> None:
    plan = pd.DataFrame(
        {
            "event_id": ["e", "e"],
            "checkpoint_id": ["c", "c"],
            "actual_schedule_sha256": ["a", "a"],
            "status": ["accepted", "accepted"],
        }
    )
    unique, rejected = deduplicate_actual_schedules(plan)
    audit = accounting_summary(2, accepted=len(unique), rejected=len(rejected))

    assert len(unique) == 1
    assert len(rejected) == 1
    assert audit["accounting_closed"]


def test_sample_contract_requires_four_branches_and_separate_action_stages() -> None:
    sample = {
        "event_id": "e",
        "checkpoint_id": "c",
        "checkpoint_state_sha256": "s",
        "actual_schedule_sha256": "a",
        "branches": [
            "candidate",
            "no_control",
            "dynamic_internal_rules",
            "hold_previous",
        ],
        "requested_schedule_path": "requested.npy",
        "projected_schedule_path": "projected.npy",
        "written_schedule_path": "written.npy",
        "target_schedule_path": "target.npy",
        "current_schedule_path": "current.npy",
        "readback_schedule_path": "readback.npy",
        "candidate_matches_reference": False,
        "no_op": False,
    }

    assert validate_sample_contract(sample)["status"] == "pass"
    sample["branches"] = ["candidate"]
    assert validate_sample_contract(sample)["status"] == "blocked"


def test_branch_manifest_marks_references_and_rejects_unknown_roles() -> None:
    manifest = build_branch_manifest(make_branch_records(["s1"]))

    assert len(manifest) == 4
    references = manifest[manifest["is_reference_branch"]]
    assert set(references["branch_role"]) == set(BRANCHES[1:])

    bad = make_branch_records(["s1"])
    bad.loc[0, "branch_role"] = "oracle"
    with pytest.raises(ValueError, match="unknown branch roles"):
        build_branch_manifest(bad)


def test_sample_manifest_never_counts_references_and_flags_duplicates() -> None:
    records = make_branch_records(
        ["s1", "s2", "s3"],
        # s2 repeats s1's actual schedule; s3 misses a reference branch.
        drop_reference_for={"s3"},
        actual_shas={"s2": "act-s1"},
    )
    manifest = build_branch_manifest(records)
    samples, duplicates, missing = build_sample_manifest(manifest)

    # Reference branches contribute evidence but never count as samples.
    assert len(samples) == 1
    assert samples["sample_id"].tolist() == ["s1"]
    assert samples["reference_branches_counted_as_samples"].eq(0).all()
    assert duplicates["sample_id"].tolist() == ["s2"]
    assert duplicates["rejection_reason"].eq("duplicate_actual_schedule").all()
    assert missing["sample_id"].tolist() == ["s3"]
    assert missing["missing_reason"].eq("reference_branch_incomplete").all()


# --- Pilot400 dataset accounting and audit columns -------------------------


def _write_pilot_branch_detail(directory, post_flood_rate, actual_value):
    import numpy as np

    elapsed = np.arange(0.0, 185.0, 5.0)
    pre = elapsed <= 60.0
    frame = {"elapsed_min": elapsed}
    frame["flood:NODEP"] = np.where(pre, 0.01, post_flood_rate)
    frame["tfv_rate_m3s"] = np.where(pre, 0.01, post_flood_rate)
    for facility in ("ADD301.2", "ADD301.3", "add350.1"):
        frame[f"flow:{facility}"] = np.where(pre, 0.5, 0.5 + post_flood_rate)
        frame[f"storage_volume:{facility}"] = np.where(pre, 10.0, 12.0)
        setting = np.where(pre, 0.0, actual_value)
        frame[f"requested_setting:{facility}"] = setting
        frame[f"target_setting:{facility}"] = setting
        frame[f"actual_setting:{facility}"] = setting
        frame[f"readback_setting:{facility}"] = setting
        frame[f"a:{facility}"] = setting
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "detail.csv"
    pd.DataFrame(frame).to_csv(path, index=False)
    return path


def test_pilot_dataset_accounting_closes_with_confirmed_missing(
    tmp_path,
) -> None:
    import json

    from sewerrtc.v4.pilot_reducers import build_pilot_dataset

    def plan_row(sample_id):
        return {
            "sample_id": sample_id,
            "event_id": "e0",
            "checkpoint_id": "e0_c0",
            "checkpoint_min": 60.0,
            "k_target": 2,
            "rainfall_sha256": "rainsha",
            "binary_semantics_ok": True,
            "rate_limit_ok": True,
            "dwell_ok": True,
            "interlock_ok": True,
            "projected_schedule_json": json.dumps([[1.0, 1.0, 1.0]]),
            "anchor_schedule_json": json.dumps([[0.0, 0.0, 0.0]]),
        }

    candidate_plan = pd.DataFrame([plan_row("s1"), plan_row("s2")])
    branch_plan = pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "case_id": f"{sample_id}__{branch}",
                "branch_role": branch,
            }
            for sample_id in ("s1", "s2")
            for branch in BRANCHES
        ]
    )
    rates = {
        "candidate": (0.02, 1.0),
        "no_control": (0.05, 0.0),
        "dynamic_internal_rules": (0.03, 0.0),
        "hold_previous": (0.01, 0.0),
    }
    completions = pd.DataFrame(
        [
            {
                "case_id": f"s1__{branch}",
                "sample_id": "s1",
                "branch": branch,
                "status": "pass",
                "detail_path": str(
                    _write_pilot_branch_detail(
                        tmp_path / f"s1__{branch}", rate, actual
                    )
                ),
                "rainfall_sha256": "rainsha",
                "input_sha": "insha",
                "runner_kwargs": json.dumps({"inp_path": "x.inp"}),
                "result": {
                    "hotstart_used": False,
                    "use_hotstart_call_count": 0,
                    "save_hotstart_call_count": 0,
                },
            }
            for branch, (rate, actual) in rates.items()
        ]
    )

    dataset = build_pilot_dataset(
        candidate_plan,
        branch_plan,
        completions,
        priority_nodes=["NODEP"],
        facility_ids=["ADD301.2", "ADD301.3", "add350.1"],
        scientific_margin={"pfv_m3": 0.0, "tfv_m3": 0.0, "peak_m3s": 0.0},
        dead_zone={"pfv_m3": 0.0, "tfv_m3": 0.0, "peak_m3s": 0.0},
    )

    accounting = dataset["accounting"]
    # planned == accepted + rejected + pending + missing with pending == 0:
    # the full-scope build confirms the unrun sample as missing.
    assert accounting["planned"] == 2
    assert accounting["accepted"] == 1
    assert accounting["pending"] == 0
    assert accounting["missing"] == 1
    assert accounting["accounting_closed"] is True
    assert len(dataset["pending"]) == 0
    assert dataset["missing_confirmed"]["sample_id"].tolist() == ["s2"]
    # Flat-state and ranking labels are attached to accepted samples.
    assert "flat_state" in dataset["sample_manifest"]


def test_pilot_audit_columns_are_recomputed_from_dataset_values() -> None:
    from sewerrtc.v4.pipeline import LOCAL_RESPONSE_FLOOR, _add_pilot_audit_columns

    config = {"thresholds": {"scientific_margin": {"tfv_m3": 0.0}}}
    samples = pd.DataFrame(
        {
            "local_response_magnitude": [LOCAL_RESPONSE_FLOOR * 2, 0.0],
            "flat_state": [False, True],
            "delta_tfv_h120_vs_dynamic_internal": [-1.0, 2.0],
        }
    )

    result = _add_pilot_audit_columns(samples, config)

    assert result["locally_responsive"].tolist() == [True, False]
    assert result["confirmed_flat"].tolist() == [False, True]
    assert result["tfv_noninferior"].tolist() == [True, False]

    empty = _add_pilot_audit_columns(samples.head(0), config)
    for column in ("locally_responsive", "confirmed_flat", "tfv_noninferior"):
        assert column in empty
