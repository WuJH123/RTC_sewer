from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def _load_micro_module():
    path = Path(__file__).parents[1] / "scripts" / "run_v42_qualification_micro.py"
    spec = importlib.util.spec_from_file_location("qualification_micro", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load qualification micro module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_common_inp_resolves_shared_reference_when_candidate_parent_lacks_inp(tmp_path: Path) -> None:
    micro = _load_micro_module()
    reference_dir = tmp_path / "reference"
    candidate_dir = tmp_path / "candidate"
    reference_dir.mkdir()
    candidate_dir.mkdir()
    inp = reference_dir / "case.inp"
    inp.write_text("[TITLE]\nqualification micro\n", encoding="utf-8")

    resolved = micro._resolve_common_inp([candidate_dir / "detail.csv", reference_dir / "detail.csv"])

    assert resolved == inp.resolve()


def test_common_inp_rejects_conflicting_network_hashes(tmp_path: Path) -> None:
    micro = _load_micro_module()
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "case.inp").write_text("first", encoding="utf-8")
    (second / "case.inp").write_text("second", encoding="utf-8")

    with pytest.raises(RuntimeError, match="network input hash mismatch"):
        micro._resolve_common_inp([first / "detail.csv", second / "detail.csv"])


def test_ledger_reuse_requires_matching_hashes(tmp_path: Path) -> None:
    micro = _load_micro_module()
    ledger = tmp_path / "ledger.csv"
    output = tmp_path / "detail.csv"
    output.write_text("elapsed_min\n5\n", encoding="utf-8")
    ledger.write_text(
        "event_id,strategy,status,input_sha256,model_sha256,policy_sha256,detail_path\n"
        f"e1,No-control,pass,in,model,policy,{output}\n",
        encoding="utf-8",
    )

    assert micro._ledger_reusable(ledger, "e1", "No-control", "in", "model", "policy")
    assert not micro._ledger_reusable(ledger, "e1", "No-control", "different", "model", "policy")


def test_json_safe_does_not_turn_nonfinite_diagnostics_into_zero() -> None:
    micro = _load_micro_module()
    value = micro._json_safe({"nan": np.nan, "inf": np.inf, "array": np.asarray([1.0, np.nan])})
    assert value == {"nan": None, "inf": None, "array": [1.0, None]}


def test_micro_reconciles_first_pass_audit(tmp_path: Path) -> None:
    micro = _load_micro_module()
    audit_path = tmp_path / "QUALIFICATION_FIRST_PASS_AUDIT.json"
    audit_path.write_text(
        json.dumps({"stage_status": {"01_core": "PASS_REUSABLE"}}),
        encoding="utf-8",
    )
    summary = {"event_rows": 70, "potential_go": False}
    status = {
        "stage_status": {"13_micro": "PASS_REUSABLE"},
        "scientific_performance_status": "provisional",
    }
    micro._update_first_pass_audit(tmp_path, summary, status)
    result = json.loads(audit_path.read_text(encoding="utf-8"))
    assert result["next_stage"] is None
    assert result["micro_summary"] == summary
    assert result["micro_stage_status"] == status
