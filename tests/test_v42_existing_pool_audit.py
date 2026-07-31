from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sewerrtc.v4.v42_existing_pool_audit import (
    ReuseClassification,
    _classify_case,
    _identity_sha,
    _infer_domain_id,
    _infer_role_from_filename,
    _source_role,
    discover_existing_details,
)
from sewerrtc.v4.v42_outfall_recovery import (
    reconstruct_outfall_flow,
    reconstruction_candidates,
    validate_outfall_reconstruction,
)
from sewerrtc.v4.v42_reusable_pool import build_reusable_paper_pool


def _case_group(*, domain: str = "target_no_dwf", outfall: bool = False, four: bool = True) -> pd.DataFrame:
    roles = ["candidate", "no_control", "dynamic_internal", "hold_previous"] if four else ["candidate"]
    rows = []
    for idx, role in enumerate(roles):
        rows.append(
            {
                "case_id": "E1__300__c1",
                "event_id": "E1",
                "rainfall_sha256": "rain",
                "checkpoint_min": 300.0,
                "network_sha256": "net",
                "domain_id": domain,
                "source_experiment": "exp",
                "source_role": "development",
                "branch_role": role,
                "formal_all_target_complete": bool(outfall),
                "core_trajectory_complete": True,
                "missing_outfall_only": not outfall,
                "outfall_reconstruction_candidate": not outfall,
                "completion_status": "pass",
                "depth_semantics": "h_is_depth_head_is_hydraulic_head",
                "physical_identity_sha256": f"p{idx}",
                "available_node_depth": True,
                "available_node_flooding_rate": True,
            }
        )
    return pd.DataFrame(rows)


def test_four_branch_all_target_is_full_reuse():
    record = _classify_case(_case_group(outfall=True))
    assert record.classification is ReuseClassification.FULL_REUSE
    assert record.four_reference_complete


def test_missing_outfall_is_partial_not_full_or_rerun():
    record = _classify_case(_case_group(outfall=False))
    assert record.classification is ReuseClassification.PARTIAL_AUX_REUSE
    assert record.outfall_only_blocker
    assert record.outfall_reconstruction_candidate


def test_dwf_core_trajectory_is_source_domain_reuse():
    record = _classify_case(_case_group(domain="source_dwf", outfall=False))
    assert record.classification is ReuseClassification.SOURCE_DOMAIN_REUSE


def test_incomplete_four_reference_can_still_be_auxiliary():
    record = _classify_case(_case_group(outfall=False, four=False))
    assert record.classification is ReuseClassification.PARTIAL_AUX_REUSE
    assert not record.four_reference_complete


def test_physical_identity_includes_action_sha():
    common = dict(
        detail_sha="d",
        network_sha="n",
        rainfall_sha="r",
        checkpoint_min=300.0,
        branch_role="candidate",
        domain_id="target_no_dwf",
    )
    assert _identity_sha(action_sha="a1", **common) != _identity_sha(action_sha="a2", **common)


def test_same_physical_identity_is_stable_for_duplicate_lineage():
    kwargs = dict(
        detail_sha="d",
        network_sha="n",
        rainfall_sha="r",
        checkpoint_min=300.0,
        branch_role="no_control",
        action_sha="a",
        domain_id="target_no_dwf",
    )
    assert _identity_sha(**kwargs) == _identity_sha(**kwargs)


def test_role_inference_covers_legacy_names():
    assert _infer_role_from_filename(Path("dynamic_internal_rules_detail.csv")) == "dynamic_internal"
    assert _infer_role_from_filename(Path("hold_previous_detail.csv")) == "hold_previous"
    assert _infer_role_from_filename(Path("candidate_detail.csv")) == "candidate"


def test_old_formal_is_consumed_not_fresh_reserved():
    assert _source_role(Path("outputs/closed_loop_paired_no_controls/formal_blind/candidate_detail.csv")) == "consumed_development"
    assert _source_role(Path("outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_blind/candidate_detail.csv")) == "reserved_evaluation"


def test_dwf_domain_is_not_automatically_invalid(tmp_path: Path):
    path = tmp_path / "historical_dwf" / "candidate_detail.csv"
    path.parent.mkdir()
    path.write_text("elapsed_min\n0\n", encoding="utf-8")
    assert _infer_domain_id(path, network_sha="old", active_network_sha="active") == "source_dwf"


def test_discovery_includes_orphan_detail_without_completion(tmp_path: Path):
    orphan = tmp_path / "pfvfirst" / "round0" / "candidate_detail.csv"
    orphan.parent.mkdir(parents=True)
    orphan.write_text("elapsed_min\n0\n", encoding="utf-8")
    found = discover_existing_details(tmp_path)
    assert len(found) == 1
    assert found.iloc[0]["branch_role"] == "candidate"
    assert found.iloc[0]["completion_path"] is None


def _write_minimal_inp(path: Path) -> None:
    path.write_text(
        """[JUNCTIONS]\nJ1 0 5\n[OUTFALLS]\nO1 0 FREE\n[CONDUITS]\nC1 J1 O1 10 0.01 0 0 0 0\n""",
        encoding="utf-8",
    )


def test_outfall_candidate_requires_complete_incoming_link_set(tmp_path: Path):
    inp = tmp_path / "m.inp"
    _write_minimal_inp(inp)
    detail = tmp_path / "detail.csv"
    pd.DataFrame({"flow:C1": [1.0, 2.0]}).to_csv(detail, index=False)
    assert reconstruction_candidates(detail, inp_path=inp) == {"O1": ["C1"]}


def test_outfall_validation_requires_explicit_recorder_column(tmp_path: Path):
    inp = tmp_path / "m.inp"
    _write_minimal_inp(inp)
    detail = tmp_path / "detail.csv"
    pd.DataFrame({"flow:C1": [1.0, 2.0]}).to_csv(detail, index=False)
    with pytest.raises(KeyError):
        validate_outfall_reconstruction(detail, inp_path=inp)


def test_validated_outfall_reconstruction_is_exact_on_fixture(tmp_path: Path):
    inp = tmp_path / "m.inp"
    _write_minimal_inp(inp)
    detail = tmp_path / "detail.csv"
    pd.DataFrame(
        {
            "flow:C1": [1.0, 2.0, 3.0],
            "outfall_flow:O1": [1.0, 2.0, 3.0],
        }
    ).to_csv(detail, index=False)
    result = validate_outfall_reconstruction(detail, inp_path=inp)
    assert result.status == "pass"
    reconstructed = reconstruct_outfall_flow(detail, inp_path=inp, validated_result=result)
    np.testing.assert_allclose(reconstructed["outfall_flow:O1"], [1.0, 2.0, 3.0])


def test_reusable_pool_masks_missing_outfall_without_zero_imputation(tmp_path: Path):
    physical = pd.DataFrame(
        [
            {
                "physical_identity_sha256": "p1",
                "source_role": "development",
                "domain_id": "target_no_dwf",
                "available_node_depth": True,
                "available_node_flooding_rate": True,
                "available_storage_volume": True,
                "available_managed_facility_flow": True,
                "available_outfall_flow": False,
                "available_readback_setting": True,
                "available_rainfall": True,
                "available_history_complete": True,
                "available_horizon_complete": True,
                "available_outfall_reconstruction_candidate": True,
            }
        ]
    )
    cases = pd.DataFrame(
        [
            {
                "case_uid": "c1",
                "classification": "PARTIAL_AUX_REUSE",
                "source_role": "development",
                "domain_id": "target_no_dwf",
                "four_reference_complete": True,
                "core_trajectory_targets": True,
                "full_reuse_targets": False,
            }
        ]
    )
    physical_path = tmp_path / "physical.csv"
    case_path = tmp_path / "cases.csv"
    physical.to_csv(physical_path, index=False)
    cases.to_csv(case_path, index=False)
    result = build_reusable_paper_pool(
        physical_inventory=physical_path,
        case_inventory=case_path,
        output_physical_manifest=tmp_path / "reuse.csv",
        output_case_manifest=tmp_path / "reuse_cases.csv",
        audit_output=tmp_path / "audit.json",
    )
    reuse = pd.read_csv(result.physical_manifest_path)
    assert bool(reuse.loc[0, "eligible_dynamics_pretrain"])
    assert not bool(reuse.loc[0, "mask_outfall_flow"])
    assert bool(reuse.loc[0, "outfall_requires_validation_before_reconstruction"])
    assert "outfall_flow" not in reuse.columns
    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert audit["missing_targets_are_imputed"] is False


def test_reusable_pool_has_no_fixed_1600_cap(tmp_path: Path):
    physical = pd.DataFrame(
        [
            {
                "physical_identity_sha256": f"p{i}",
                "source_role": "development",
                "domain_id": "target_no_dwf",
                "available_node_depth": True,
                "available_node_flooding_rate": True,
                "available_storage_volume": True,
                "available_managed_facility_flow": True,
                "available_outfall_flow": False,
                "available_readback_setting": True,
                "available_rainfall": True,
                "available_history_complete": True,
                "available_horizon_complete": True,
                "available_outfall_reconstruction_candidate": False,
            }
            for i in range(1601)
        ]
    )
    cases = pd.DataFrame(
        [
            {
                "case_uid": "c1",
                "classification": "PARTIAL_AUX_REUSE",
                "source_role": "development",
                "domain_id": "target_no_dwf",
                "four_reference_complete": True,
                "core_trajectory_targets": True,
                "full_reuse_targets": False,
            }
        ]
    )
    p = tmp_path / "p.csv"
    c = tmp_path / "c.csv"
    physical.to_csv(p, index=False)
    cases.to_csv(c, index=False)
    result = build_reusable_paper_pool(
        physical_inventory=p,
        case_inventory=c,
        output_physical_manifest=tmp_path / "reuse.csv",
        output_case_manifest=tmp_path / "reuse_cases.csv",
        audit_output=tmp_path / "audit.json",
    )
    assert result.physical_row_count == 1601
