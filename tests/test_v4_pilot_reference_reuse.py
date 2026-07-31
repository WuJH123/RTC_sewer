"""Tests for pilot reference-cache reuse and sample-task accounting."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from sewerrtc.v4 import pilot_run
from sewerrtc.v4.pilot_run import (
    attach_branch_context,
    ensure_reference_cache,
    expand_pilot_completions,
    run_pilot_sample,
)

IDENTITY = {
    "network_sha256": "net",
    "config_sha256": "cfg",
    "contract_sha256": "con",
    "checkpoint_state_sha256": "state",
}


def _stub_runner(calls: list[str]):
    def stub_run_prepared_case(row, paths):
        directory = Path(paths["directory"])
        directory.mkdir(parents=True, exist_ok=True)
        detail = directory / "detail.csv"
        detail.write_text("elapsed_min,x\n0,1\n", encoding="utf-8")
        calls.append(str(row.get("case_id", row.get("sample_id", ""))))
        return {
            "runner_function": "run_swmm_fixed_action",
            "result": {
                "hotstart_used": False,
                "use_hotstart_call_count": 0,
                "save_hotstart_call_count": 0,
            },
            "detail_path": str(detail),
        }

    return stub_run_prepared_case


def _reference_branch_rows(sample_id: str) -> dict[str, dict]:
    return {
        branch: {
            "branch_role": branch,
            "case_id": f"{sample_id}__{branch}",
            "runner_function": "run_swmm_fixed_action",
            "runner_kwargs": json.dumps({"inp_path": "x.inp"}),
        }
        for branch in (
            "candidate",
            "no_control",
            "dynamic_internal_rules",
            "hold_previous",
        )
    }


def test_reference_cache_single_writer_then_zero_cost_readers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        pilot_run, "run_prepared_case", _stub_runner(calls)
    )

    payload, first = ensure_reference_cache(
        tmp_path,
        "e0",
        "e0_c0",
        identity=IDENTITY,
        branch_rows=_reference_branch_rows("s1"),
    )
    _, second = ensure_reference_cache(
        tmp_path,
        "e0",
        "e0_c0",
        identity=IDENTITY,
        branch_rows=_reference_branch_rows("s2"),
    )

    assert first == 3
    assert second == 0
    assert len(calls) == 3  # exactly one physical run per reference branch
    assert isinstance(payload, dict)


def _sample_row(sample_id: str, reference_root: Path) -> dict:
    return {
        "sample_id": sample_id,
        "case_id": f"{sample_id}__candidate",
        "event_id": "e0",
        "checkpoint_id": "e0_c0",
        "runner_function": "run_swmm_fixed_action",
        "runner_kwargs": json.dumps({"inp_path": "x.inp"}),
        "branch_rows_json": json.dumps(
            list(_reference_branch_rows(sample_id).values())
        ),
        "reference_root": str(reference_root),
        "reference_identity_json": json.dumps(IDENTITY, sort_keys=True),
    }


def test_run_pilot_sample_reuses_same_state_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        pilot_run, "run_prepared_case", _stub_runner(calls)
    )
    reference_root = tmp_path / "references_root"

    outcomes = []
    for sample_id in ("s1", "s2"):
        case_dir = tmp_path / "runs" / f"{sample_id}__candidate"
        outcomes.append(
            run_pilot_sample(
                _sample_row(sample_id, reference_root),
                {"directory": str(case_dir)},
            )
        )

    # First sample pays 3 reference runs + 1 candidate; the same-state
    # second sample reuses the cache and pays exactly 1 candidate run.
    assert outcomes[0]["physical_swmm_runs"] == 4
    assert outcomes[1]["physical_swmm_runs"] == 1
    assert len(calls) == 5
    for outcome in outcomes:
        assert set(outcome["branches"]) == {
            "candidate",
            "no_control",
            "dynamic_internal_rules",
            "hold_previous",
        }
        for info in outcome["branches"].values():
            assert Path(str(info["detail_path"])).exists()


def test_expand_pilot_completions_yields_four_branch_rows(
    tmp_path: Path,
) -> None:
    branches = {
        branch: {
            "status": "pass",
            "detail_path": f"{branch}/detail.csv",
            "result": {"hotstart_used": False},
            "runner_kwargs": json.dumps({"inp_path": "x.inp"}),
        }
        for branch in (
            "candidate",
            "no_control",
            "dynamic_internal_rules",
            "hold_previous",
        )
    }
    completions = pd.DataFrame(
        [
            {
                "case_id": "s1",
                "sample_id": "s1",
                "status": "pass",
                "branches": branches,
                "rainfall_sha256": "rain",
                "input_sha": "in",
            },
            {
                "case_id": "s2",
                "sample_id": "s2",
                "status": "failed",
                "branches": branches,
                "rainfall_sha256": "rain",
                "input_sha": "in",
            },
            # Foreign / broken marker without a branches payload: skipped.
            {"case_id": "s3", "sample_id": "s3", "status": "pass"},
        ]
    )

    expanded = expand_pilot_completions(completions)

    assert len(expanded) == 8
    s1 = expanded[expanded["sample_id"] == "s1"]
    assert sorted(s1["case_id"]) == sorted(
        f"s1__{branch}" for branch in branches
    )
    assert s1["status"].eq("pass").all()
    # A failed sample poisons all of its branch rows.
    s2 = expanded[expanded["sample_id"] == "s2"]
    assert s2["status"].eq("failed").all()
    assert "s3" not in set(expanded["sample_id"])


def test_attach_branch_context_requires_four_branch_rows(
    tmp_path: Path,
) -> None:
    candidate_plan = pd.DataFrame(
        [{"sample_id": "s1", "case_id": "s1__candidate"}]
    )
    branch_plan = pd.DataFrame(
        [
            row
            for row in _reference_branch_rows("s1").values()
            if row["branch_role"] != "hold_previous"
        ]
    )
    branch_plan["sample_id"] = "s1"

    with pytest.raises(ValueError, match="missing four-branch plan rows"):
        attach_branch_context(
            candidate_plan, branch_plan, reference_root=tmp_path
        )
