"""Tests for the zero-SWMM Peak Boundary restamp (spec section I)."""
from __future__ import annotations

import ast
import inspect
import io
import json
from pathlib import Path

import pandas as pd

from sewerrtc.v4 import peak_restamp
from sewerrtc.v4.peak_restamp import (
    LABEL_COLUMNS,
    _network_matches_contract_anchor,
    _sha256_file,
    compare_with_frozen,
    restamp_peak_boundary_evidence,
)


N_SAMPLES = 60
BRANCHES = (
    "candidate",
    "no_control",
    "dynamic_internal_rules",
    "hold_previous",
)
TOLERANCE = {"pfv_m3": 0.0, "tfv_m3": 0.0, "peak_m3s": 0.0}


def _samples() -> pd.DataFrame:
    rows = []
    for index in range(N_SAMPLES):
        rows.append(
            {
                "sample_id": f"s{index:02d}",
                "actual_schedule_sha256": f"actual{index:02d}",
                # Exact binary fractions so the frozen CSV round-trips
                # bit-identically under the all-zero tolerance.
                "delta_pfv_h120_vs_no_control": -1.0 - index,
                "delta_tfv_h120_vs_dynamic_internal": -0.5 - index,
                "delta_peak_h120_vs_dynamic_internal": -index / 64.0,
                "pfv_safe": index % 3 != 0,
                "tfv_improved": index % 2 == 0,
                "peak_noninferior": index >= 42,
                "joint_noninferior": index % 5 == 0,
                "materially_beneficial": index % 7 == 0,
                "neutral": False,
                "hard_negative_type": "pfv_safe_peak" if index < 33 else "",
                "state_hash_match": True,
                "readback_ok": True,
            }
        )
    return pd.DataFrame(rows)


def _run_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": f"s{index:02d}",
                "branch": branch,
                "status": "pass",
            }
            for index in range(N_SAMPLES)
            for branch in BRANCHES
        ]
    )


def _audit() -> dict:
    return {
        "status": "pass",
        "peak_degraded": 42,
        "pfv_safe_peak_hard_negative": 33,
    }


def _write_frozen(archive_dir: Path, output_root: Path) -> None:
    frozen = archive_dir / "peak_boundary"
    frozen.mkdir(parents=True, exist_ok=True)
    _samples().to_csv(frozen / "sample_manifest.csv", index=False)
    _run_manifest().to_csv(frozen / "run_manifest.csv", index=False)
    (frozen / "peak_boundary_audit.json").write_text(
        json.dumps(_audit()), encoding="utf-8"
    )
    live = output_root / "peak_boundary"
    live.mkdir(parents=True, exist_ok=True)
    _run_manifest().to_csv(live / "run_manifest.csv", index=False)


def test_identical_rebuild_matches_frozen_evidence(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    output_root = tmp_path / "out"
    _write_frozen(archive, output_root)

    equal, report = compare_with_frozen(
        _samples(), _audit(), output_root, archive, tolerance=TOLERANCE
    )

    assert equal is True
    assert report["mismatches"] == []
    checks = report["checks"]
    assert checks["sample_count"] == 60
    assert checks["branch_association_count"] == 240
    assert checks["actual_schedule_duplicates"] == 0
    # The frozen scientific counts survive the restamp unchanged.
    assert checks["peak_degraded"] == 42
    assert checks["pfv_safe_peak_hard_negative"] == 33


def test_frozen_csv_is_read_with_round_trip_float_precision(
    tmp_path: Path,
) -> None:
    # Regression: the default lossy C parser reads this real frozen value
    # one ULP off, which spuriously broke the all-zero tolerance.
    tricky = 9.552434241388141
    csv_text = f"x\n{tricky!r}\n"
    lossy = float(
        pd.read_csv(io.StringIO(csv_text))["x"].iloc[0]
    )
    exact = float(
        pd.read_csv(io.StringIO(csv_text), float_precision="round_trip")[
            "x"
        ].iloc[0]
    )
    assert exact == tricky
    assert lossy != tricky  # precondition: the value is parser-sensitive

    archive = tmp_path / "archive"
    output_root = tmp_path / "out"
    _write_frozen(archive, output_root)
    frozen_path = archive / "peak_boundary" / "sample_manifest.csv"
    frozen = pd.read_csv(frozen_path, float_precision="round_trip")
    frozen.loc[0, "delta_pfv_h120_vs_no_control"] = tricky
    frozen.to_csv(frozen_path, index=False)
    rebuilt = _samples()
    rebuilt.loc[0, "delta_pfv_h120_vs_no_control"] = tricky

    equal, report = compare_with_frozen(
        rebuilt, _audit(), output_root, archive, tolerance=TOLERANCE
    )

    assert equal is True, report["mismatches"]


def test_delta_drift_beyond_frozen_tolerance_fails_closed(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive"
    output_root = tmp_path / "out"
    _write_frozen(archive, output_root)
    drifted = _samples()
    drifted.loc[0, "delta_pfv_h120_vs_no_control"] += 1e-6

    equal, report = compare_with_frozen(
        drifted, _audit(), output_root, archive, tolerance=TOLERANCE
    )

    assert equal is False
    assert (
        "delta_pfv_h120_vs_no_control_exceeds_tolerance"
        in report["mismatches"]
    )


def test_label_flip_or_count_change_is_detected(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    output_root = tmp_path / "out"
    _write_frozen(archive, output_root)

    flipped = _samples()
    flipped.loc[0, "pfv_safe"] = not bool(flipped.loc[0, "pfv_safe"])
    equal, report = compare_with_frozen(
        flipped, _audit(), output_root, archive, tolerance=TOLERANCE
    )
    assert equal is False
    assert "label_pfv_safe_differ" in report["mismatches"]

    bad_audit = dict(_audit(), peak_degraded=41)
    equal, report = compare_with_frozen(
        _samples(), bad_audit, output_root, archive, tolerance=TOLERANCE
    )
    assert equal is False
    assert "peak_degraded_count_differ" in report["mismatches"]

    lost = dict(_audit(), pfv_safe_peak_hard_negative=32)
    equal, report = compare_with_frozen(
        _samples(), lost, output_root, archive, tolerance=TOLERANCE
    )
    assert equal is False
    assert (
        "pfv_safe_peak_hard_negative_count_differ" in report["mismatches"]
    )


def test_restamp_module_never_launches_swmm() -> None:
    # Spec section I: the restamp must rebuild purely from cached branch
    # outputs; statically prove that no SWMM entry point is imported or
    # referenced in code (docstrings may mention the prohibition itself).
    tree = ast.parse(inspect.getsource(peak_restamp))
    referenced: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            referenced.add(node.id)
        elif isinstance(node, ast.Attribute):
            referenced.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            referenced.update(alias.name for alias in node.names)
            referenced.add(str(node.module))
        elif isinstance(node, ast.Import):
            referenced.update(alias.name for alias in node.names)
    for forbidden in (
        "run_prepared_case",
        "run_swmm_fixed_action",
        "run_swmm_dynamic_internal",
        "run_parallel_cases",
        "pyswmm",
        "pyswmm_runner",
    ):
        assert forbidden not in referenced, forbidden


def test_restamp_without_frozen_archive_is_blocked(tmp_path: Path) -> None:
    payload = restamp_peak_boundary_evidence(
        tmp_path, tmp_path / "out", {"project": {}, "thresholds": {}}
    )

    assert payload["status"] == "blocked"
    assert payload["reason"] == "frozen_archive_missing"


def test_all_seven_label_columns_are_compared() -> None:
    assert set(LABEL_COLUMNS) == {
        "pfv_safe",
        "tfv_improved",
        "peak_noninferior",
        "joint_noninferior",
        "materially_beneficial",
        "neutral",
        "hard_negative_type",
    }


def _network_fixture(tmp_path: Path) -> tuple[Path, dict, pd.DataFrame, Path]:
    """Project with a network file, a contract anchor, and a legacy plan."""
    project_root = tmp_path / "proj"
    (project_root / "data").mkdir(parents=True)
    network_path = project_root / "data" / "network.inp"
    network_path.write_text("[TITLE]\nfixture network\n", encoding="utf-8")
    contract_path = project_root / "contract.json"
    contract_path.write_text(
        json.dumps({"network_sha256": _sha256_file(network_path)}),
        encoding="utf-8",
    )
    config = {
        "project": {"network": "data/network.inp", "contract": "contract.json"}
    }
    # Legacy frozen plan shape: no network_sha256 column, network identity
    # only via runner_kwargs.inp_path.
    plan = pd.DataFrame(
        {
            "case_id": ["a", "b"],
            "runner_kwargs": [
                json.dumps({"inp_path": str(network_path)}),
                json.dumps({"inp_path": str(network_path)}),
            ],
        }
    )
    return project_root, config, plan, network_path


def test_legacy_plan_without_network_column_passes_via_contract_anchor(
    tmp_path: Path,
) -> None:
    project_root, config, plan, network_path = _network_fixture(tmp_path)

    assert _network_matches_contract_anchor(
        project_root, config, plan, network_path
    )


def test_contract_anchor_mismatch_or_ambiguous_inp_path_fails_closed(
    tmp_path: Path,
) -> None:
    project_root, config, plan, network_path = _network_fixture(tmp_path)

    # Live network drifted away from the contract anchor.
    network_path.write_text("[TITLE]\nmutated network\n", encoding="utf-8")
    assert not _network_matches_contract_anchor(
        project_root, config, plan, network_path
    )
    network_path.write_text("[TITLE]\nfixture network\n", encoding="utf-8")

    # Plan rows disagreeing on inp_path must fail closed.
    ambiguous = plan.copy()
    ambiguous.loc[1, "runner_kwargs"] = json.dumps(
        {"inp_path": str(project_root / "data" / "other.inp")}
    )
    assert not _network_matches_contract_anchor(
        project_root, config, ambiguous, network_path
    )

    # Missing contract reference must fail closed.
    assert not _network_matches_contract_anchor(
        project_root, {"project": {"network": "data/network.inp"}}, plan,
        network_path,
    )

