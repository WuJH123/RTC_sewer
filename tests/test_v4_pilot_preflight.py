"""Tests for the Pilot preflight gate (spec section VIII fixes)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from sewerrtc.v4.preflight import preflight_checks


GOOD_EVIDENCE = {
    "writer_lock_free": True,
    "reference_cache_clean": True,
    "active_conflicting_pids": [],
    "torch_free_override": True,
}

KEY_FIELDS = (
    "candidate_projection_present",
    "requested_schedule_duplicates_zero",
    "k_le_8",
    "engineering_constraint_columns_present",
)


def _good_plan() -> pd.DataFrame:
    rows = []
    for sample_index in range(4):
        sample_id = f"e0_c0__s{sample_index}"
        for branch in (
            "candidate",
            "no_control",
            "dynamic_internal_rules",
            "hold_previous",
        ):
            rows.append(
                {
                    "case_id": f"{sample_id}__{branch}",
                    "sample_id": sample_id,
                    "branch": branch,
                    "event_id": "e0",
                    "checkpoint_id": "e0_c0",
                    "split": "pilot_train",
                    "requested_schedule_sha256": f"req{sample_index}",
                    "projected_schedule_sha256": f"proj{sample_index}",
                    "k_target": 2,
                    "binary_semantics_ok": True,
                    "rate_limit_ok": True,
                    "dwell_ok": True,
                    "interlock_ok": True,
                }
            )
    return pd.DataFrame(rows)


def _run(plan: pd.DataFrame, tmp_path: Path, evidence: dict) -> dict:
    return preflight_checks(
        plan,
        workers=1,
        output_root=tmp_path,
        input_sha="insha",
        evidence=evidence,
        minimum_free_bytes=1,
        probe_torch=False,
    )


def test_good_pilot_plan_passes_all_four_key_fields(tmp_path: Path) -> None:
    report = _run(_good_plan(), tmp_path, GOOD_EVIDENCE)

    assert report["status"] == "pass"
    assert report["exit_code"] == 0
    for field in KEY_FIELDS:
        assert report["checks"][field] is True, field
    assert report["requested_schedule_duplicates"] == 0


def test_reference_rows_sharing_requested_sha_are_not_duplicates(
    tmp_path: Path,
) -> None:
    # The three reference branches legitimately reuse the sample's requested
    # SHA; only candidate-row duplicates may block the run.
    plan = _good_plan()
    report = _run(plan, tmp_path, GOOD_EVIDENCE)
    assert report["checks"]["requested_schedule_duplicates_zero"] is True

    # A genuine candidate duplicate must block.
    duplicated = pd.concat(
        [plan, plan[plan["branch"].eq("candidate")].head(1)],
        ignore_index=True,
    )
    blocked = _run(duplicated, tmp_path, GOOD_EVIDENCE)
    assert blocked["status"] == "blocked"
    assert blocked["checks"]["requested_schedule_duplicates_zero"] is False


def test_missing_projection_column_blocks(tmp_path: Path) -> None:
    plan = _good_plan().drop(columns=["projected_schedule_sha256"])
    report = _run(plan, tmp_path, GOOD_EVIDENCE)

    assert report["status"] == "blocked"
    assert report["checks"]["candidate_projection_present"] is False


def test_k_above_8_blocks(tmp_path: Path) -> None:
    plan = _good_plan()
    plan.loc[0, "k_target"] = 9
    report = _run(plan, tmp_path, GOOD_EVIDENCE)

    assert report["status"] == "blocked"
    assert report["checks"]["k_le_8"] is False


def test_false_engineering_constraint_blocks(tmp_path: Path) -> None:
    plan = _good_plan()
    plan.loc[0, "dwell_ok"] = False
    report = _run(plan, tmp_path, GOOD_EVIDENCE)

    assert report["status"] == "blocked"
    assert report["checks"]["engineering_constraint_columns_present"] is False


def test_missing_runtime_evidence_fails_closed(tmp_path: Path) -> None:
    # No writer-lock / reference-cache / pid evidence: never a pass.
    report = _run(_good_plan(), tmp_path, {"torch_free_override": True})

    assert report["status"] == "blocked"
    assert report["checks"]["writer_lock_free"] is False
    assert report["checks"]["reference_cache_clean"] is False
    assert report["checks"]["no_conflicting_active_pids"] is False
