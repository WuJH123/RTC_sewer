from __future__ import annotations

import csv
import json
from pathlib import Path

from sewerrtc.state.state_clone_equivalence import evaluate_state_clone_gate


ROOT = Path(__file__).resolve().parents[1]
CLONE = ROOT / "sewerrtc" / "state" / "state_clone_equivalence.py"
RUNNER = ROOT / "scripts" / "project6_runs" / "RUN_PROJECT6_PFVFIRST_DUALFALLBACK_10MIN_V3.ps1"
BASELINE_RUNNER = ROOT / "sewerrtc" / "simulation" / "pyswmm_runner.py"
BASELINE_SCRIPT = ROOT / "scripts" / "160_generate_baseline_trajectories.py"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_runtime_executed_false_cannot_pass(tmp_path: Path) -> None:
    (tmp_path / "state_clone_report.json").write_text(
        json.dumps({"runtime_executed": False, "eligible_checkpoint_count": 18, "executed_checkpoint_count": 18}),
        encoding="utf-8",
    )
    _write_csv(tmp_path / "state_clone_equivalence.csv", [{"checkpoint_id": "c1", "metric": "PFV", "status": "pass"}])
    _write_csv(tmp_path / "state_clone_controller_memory_audit.csv", [{"checkpoint_id": "c1", "status": "pass"}])
    _write_csv(tmp_path / "state_clone_timeline_audit.csv", [{"checkpoint_id": "c1", "status": "pass"}])
    (tmp_path / "state_clone_numerical_noise.json").write_text(json.dumps({"empirically_measured": True}), encoding="utf-8")

    code, outputs = evaluate_state_clone_gate(tmp_path)

    assert code == 3
    gate = json.loads(outputs["gate"].read_text(encoding="utf-8"))
    assert gate["formal_same_state_unlock_allowed"] is False
    assert "runtime_not_executed" in gate["blocking_reasons"]


def test_executed_but_failed_metrics_returns_gate_failure(tmp_path: Path) -> None:
    (tmp_path / "state_clone_report.json").write_text(
        json.dumps({"runtime_executed": True, "eligible_checkpoint_count": 18, "executed_checkpoint_count": 18}),
        encoding="utf-8",
    )
    _write_csv(tmp_path / "state_clone_equivalence.csv", [{"checkpoint_id": "c1", "metric": "PFV", "status": "failed_gate"}])
    _write_csv(tmp_path / "state_clone_controller_memory_audit.csv", [{"checkpoint_id": "c1", "status": "pass"}])
    _write_csv(tmp_path / "state_clone_timeline_audit.csv", [{"checkpoint_id": "c1", "status": "pass"}])
    (tmp_path / "state_clone_numerical_noise.json").write_text(json.dumps({"empirically_measured": True}), encoding="utf-8")

    code, outputs = evaluate_state_clone_gate(tmp_path)

    assert code == 5
    gate = json.loads(outputs["gate"].read_text(encoding="utf-8"))
    assert gate["status"] == "failed_gate"


def test_state_clone_module_contains_real_runtime_executor_not_plan_only() -> None:
    text = CLONE.read_text(encoding="utf-8")
    assert "try_use_hotstart" in text
    assert "from pyswmm import Links, Nodes, RainGages, Simulation" in text
    assert "state_clone_timeline_audit.csv" in text
    assert "state_clone_controller_memory_audit.csv" in text
    assert "requires_real_swmm_hotstart_equivalence_run" in text
    assert "write_state_clone_equivalence_plan" not in text


def test_baseline_detail_materializes_head_and_all_facility_columns() -> None:
    text = BASELINE_RUNNER.read_text(encoding="utf-8")
    assert 'row[f"head:{nid}"]' in text
    assert "all_requested_actuator_ids" in text
    assert 'row.setdefault(f"a:{aid}", np.nan)' in text
    assert 'row.setdefault(f"setting:{aid}", np.nan)' in text


def test_baseline_generator_uses_36_facility_contract_not_10_asset_table_only() -> None:
    text = BASELINE_SCRIPT.read_text(encoding="utf-8")
    assert "project6_v3_facility_semantics_36.csv" in text
    assert "project6_v8_storage_retrofit_control_enabled_ids.txt" in text
    assert "_load_baseline_actuators()" in text


def test_runner_supports_smoke_full_clone_modes_without_smoke_marker() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert '[string]$StateCloneMode = "full"' in text
    assert '[int]$MaxCheckpoints = 0' in text
    assert '"--mode", $StateCloneMode' in text
    assert 'if ($StateCloneMode -eq "smoke")' in text
    assert 'Status "runtime_partial"' in text
