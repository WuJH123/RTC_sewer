from pathlib import Path

import json

from sewerrtc.v4.contracts import audit_final_contract


ROOT = Path(__file__).resolve().parents[1]


def test_final_contract_freezes_core_runtime_and_reference_roles() -> None:
    contract_path = ROOT / "docs/contracts/PROJECT6_V4_FINAL_PIPELINE_CONTRACT.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    audit = audit_final_contract(contract, ROOT)

    assert audit["status"] == "pass"
    assert contract["network_variant"] == "rainfall_only_no_dwf"
    assert contract["state_record_step_sec"] == 300
    assert contract["control_interval_sec"] == 600
    assert contract["history_frames"] == 13
    assert contract["horizon_steps"] == 12
    assert contract["max_active_changes"] == 8
    assert contract["references"] == {
        "PFV": "no_control",
        "TFV": "dynamic_internal_rules",
        "Peak": "dynamic_internal_rules",
        "necessity": "hold_previous",
    }


def test_final_contract_records_full_and_physical_network_hashes() -> None:
    contract = json.loads(
        (ROOT / "docs/contracts/PROJECT6_V4_FINAL_PIPELINE_CONTRACT.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(contract["network_sha256"]) == 64
    assert len(contract["physical_network_sha256"]) == 64
    assert contract["active_dwf_flow_rows"] == 0


def test_final_contract_verifies_canonical_order_and_facility_semantics() -> None:
    contract = json.loads(
        (ROOT / "docs/contracts/PROJECT6_V4_FINAL_PIPELINE_CONTRACT.json").read_text(
            encoding="utf-8"
        )
    )
    audit = audit_final_contract(contract, ROOT)

    assert audit["canonical_facility_order"]["count"] == 36
    assert audit["facility_semantics"]["count"] == 36
    assert audit["checks"]["canonical_order_sha256"]
    assert audit["checks"]["facility_semantics_sha256"]
    assert audit["checks"]["binary_semantics"]
    assert audit["checks"]["variable_speed_semantics"]
