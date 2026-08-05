from __future__ import annotations

import pandas as pd

from sewerrtc.v4.v42_formal_runtime import _audit_baseline_contract


def _detail(command: float, readback: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a:ADD301.2": [command],
            "setting:ADD301.2": [readback],
        }
    )


def test_no_control_contract_uses_actual_setting_readback(tmp_path) -> None:
    inp = tmp_path / "network.inp"
    inp.write_text("[CONTROLS]\nRULE1 LINK L1 SETTING = 1\n", encoding="utf-8")

    result = _audit_baseline_contract(_detail(1.0, 1.0), "No-control", inp)

    assert result["readback_finite"] is True
    assert result["target_write_verified"] is True
    assert result["physical_setting_verified"] is True
    assert result["baseline_contract_pass"] is True


def test_baseline_contract_rejects_target_write_mismatch(tmp_path) -> None:
    inp = tmp_path / "network.inp"
    inp.write_text("[CONTROLS]\nRULE1 LINK L1 SETTING = 1\n", encoding="utf-8")

    result = _audit_baseline_contract(_detail(1.0, 0.5), "No-control", inp)

    assert result["readback_finite"] is True
    assert result["target_write_verified"] is False
    assert result["physical_setting_verified"] is False
    assert result["baseline_contract_pass"] is False


def test_internal_records_native_rule_contract(tmp_path) -> None:
    inp = tmp_path / "network.inp"
    inp.write_text("[CONTROLS]\nRULE1 LINK L1 SETTING = 1\n", encoding="utf-8")

    result = _audit_baseline_contract(_detail(0.3, 0.3), "Internal", inp)

    assert result["target_write_verified"] is None
    assert result["internal_native_rules_preserved"] is True
    assert result["native_rule_count"] == 1
    assert result["native_rule_contract_sha256"]
    assert result["baseline_contract_pass"] is True
