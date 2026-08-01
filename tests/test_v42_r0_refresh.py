from __future__ import annotations

from pathlib import Path

import pandas as pd

from sewerrtc.v4.v42_r0_refresh import (
    AUX_CANDIDATE_THEN_INTERNAL,
    AUX_CANDIDATE_THEN_PASSIVE,
    _discovery_index,
    _logical_key,
    _formal_role_from_filename,
    discover_formal_existing_details,
)


def test_pfvfirst_continuations_are_auxiliary_not_canonical_references():
    assert _formal_role_from_filename(Path("candidate.csv")) == "candidate"
    assert (
        _formal_role_from_filename(Path("candidate_then_internal.csv"))
        == AUX_CANDIDATE_THEN_INTERNAL
    )
    assert (
        _formal_role_from_filename(Path("candidate_then_passive.csv"))
        == AUX_CANDIDATE_THEN_PASSIVE
    )
    assert _formal_role_from_filename(Path("dynamic_internal_detail.csv")) == "dynamic_internal"
    assert _formal_role_from_filename(Path("hold_previous_detail.csv")) == "hold_previous"


def test_formal_discovery_keeps_pfvfirst_files_but_not_as_fake_four_references(tmp_path: Path):
    details = tmp_path / "pfvfirst" / "round0" / "dryrun_runtime" / "abc" / "details"
    details.mkdir(parents=True)
    for name in (
        "candidate.csv",
        "candidate_then_internal.csv",
        "candidate_then_passive.csv",
    ):
        (details / name).write_text("elapsed_min,rainfall_mm_h\n0,0\n", encoding="utf-8")
    found = discover_formal_existing_details(tmp_path)
    assert len(found) == 3
    assert set(found["branch_role"]) == {
        "candidate",
        AUX_CANDIDATE_THEN_INTERNAL,
        AUX_CANDIDATE_THEN_PASSIVE,
    }
    assert "dynamic_internal" not in set(found["branch_role"])
    assert "hold_previous" not in set(found["branch_role"])


def test_discovery_fingerprint_changes_when_population_changes(tmp_path: Path):
    details = tmp_path / "runs" / "a"
    details.mkdir(parents=True)
    first = details / "candidate_detail.csv"
    first.write_text("elapsed_min\n0\n", encoding="utf-8")
    d1 = discover_formal_existing_details(tmp_path)
    _, sha1 = _discovery_index(d1)

    second_dir = tmp_path / "runs" / "b"
    second_dir.mkdir(parents=True)
    (second_dir / "no_control_detail.csv").write_text("elapsed_min\n0\n", encoding="utf-8")
    d2 = discover_formal_existing_details(tmp_path)
    _, sha2 = _discovery_index(d2)
    assert len(d2) == len(d1) + 1
    assert sha2 != sha1


def test_logical_key_normalizes_missing_metadata(tmp_path: Path):
    detail = tmp_path / "detail.csv"
    detail.write_text("elapsed_min\n0\n", encoding="utf-8")
    a = pd.Series(
        {
            "detail_path": str(detail),
            "case_id": "c",
            "event_id": "e",
            "checkpoint_min": None,
            "network_sha256": None,
            "rainfall_sha256": float("nan"),
            "branch_role": "candidate",
            "completion_path": None,
        }
    )
    b = a.copy()
    b["checkpoint_min"] = float("nan")
    b["network_sha256"] = float("nan")
    assert _logical_key(a) == _logical_key(b)
