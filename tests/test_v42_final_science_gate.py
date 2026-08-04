from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from sewerrtc.v4.v42_final_science_gate import (
    EXPECTED_STRATEGIES,
    audit_final_scientific_outcomes,
)
from sewerrtc.v4.v42_priority_contract import PFV_CORE_8_IDS


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _detail(path: Path, *, priority_flood_rate: float, total_extra_rate: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {f"flood:{node}": 0.0 for node in PFV_CORE_8_IDS}
    row[f"flood:{PFV_CORE_8_IDS[0]}"] = priority_flood_rate
    row["flood:NON_PRIORITY"] = total_extra_rate
    row["elapsed_min"] = 5.0
    pd.DataFrame([row]).to_csv(path, index=False)


def _build_final_fixture(root: Path, *, violating_event: str | None = None) -> None:
    paper = root / "v42_paper"
    formal = paper / "formal_f2"
    evidence = paper / "formal_blind/evidence.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps({"status": "pass"}), encoding="utf-8")

    rows = []
    for event_no in range(24):
        event = f"event_{event_no:02d}"
        for strategy in EXPECTED_STRATEGIES:
            detail = formal / "paper_execution/formal_blind" / event / strategy / "detail.csv"
            if strategy == "No-control":
                rate = 1.0
            elif strategy == "Proposed":
                # PFV over one 300 s row: no-control = 300 m3.
                # Allowed proposed = 100 + 1.05*300 = 415 m3 => 1.3833 m3/s.
                rate = 1.5 if event == violating_event else 1.2
            else:
                rate = 1.1
            _detail(detail, priority_flood_rate=rate)
            rows.append(
                {
                    "role": "formal_blind",
                    "event_id": event,
                    "strategy": strategy,
                    "status": "pass",
                    "authority": "authoritative_swmm",
                    "detail_path": str(detail),
                    "detail_sha256": _sha(detail),
                }
            )
    ledger = formal / "paper_execution/FORMAL_EXECUTION_LEDGER.csv"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(ledger, index=False)


def test_final_science_gate_passes_when_every_event_respects_pfv_budget(tmp_path: Path) -> None:
    _build_final_fixture(tmp_path)
    result = audit_final_scientific_outcomes(tmp_path)
    assert result["status"] == "pass"
    assert result["scientific_constraint_pass"] is True
    assert result["PFV_violation_count"] == 0
    assert result["event_count"] == 24


def test_final_science_gate_fails_on_one_authoritative_pfv_violation(tmp_path: Path) -> None:
    _build_final_fixture(tmp_path, violating_event="event_07")
    result = audit_final_scientific_outcomes(tmp_path)
    assert result["status"] == "fail"
    assert result["scientific_constraint_pass"] is False
    assert result["PFV_violation_count"] == 1
    assert "final_PFV_hard_constraint_violations:1" in result["reasons"]
