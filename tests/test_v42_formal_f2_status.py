from __future__ import annotations

import json
from pathlib import Path

from scripts.run_v42_formal_f2 import PFV_SAFETY_STATISTIC, _status


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_status_reads_current_pfv_only_calibration_filename(tmp_path: Path) -> None:
    formal = (
        tmp_path
        / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2"
    )
    _write_json(
        formal / "calibration/PFV_ONLY_SAFETY_CALIBRATION.json",
        {
            "status": "pass",
            "safety_calibrated": True,
            "pfv_safety_statistic": PFV_SAFETY_STATISTIC,
            "pfv_predicted_safe_count": 3,
        },
    )

    payload = _status(tmp_path)

    assert payload["step2_calibration"] is not None
    assert payload["step2_calibration"]["pfv_safety_statistic"] == PFV_SAFETY_STATISTIC
    assert not any(
        reason.startswith("step2_current_calibration")
        for reason in payload["calibration_reasons"]
    )


def test_status_rejects_legacy_or_empty_pfv_calibration(tmp_path: Path) -> None:
    formal = (
        tmp_path
        / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2"
    )
    _write_json(
        formal / "calibration/PFV_ONLY_SAFETY_CALIBRATION.json",
        {
            "status": "pass",
            "safety_calibrated": True,
            "pfv_safety_statistic": "legacy_candidate_minus_no_control",
            "pfv_predicted_safe_count": 0,
        },
    )

    payload = _status(tmp_path)
    assert payload["calibration_chain_pass"] is False
    assert "step2_current_calibration_wrong_pfv_safety_statistic" in payload[
        "calibration_reasons"
    ]
