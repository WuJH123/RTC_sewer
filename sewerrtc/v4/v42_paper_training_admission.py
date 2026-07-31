"""Admission gate for formal V4.2 trajectory-surrogate training.

The old V4.2 trainer is not scientifically authoritative for the paper line
because it regresses independent KPI heads and does not require the full
hydraulic target contract.  Formal paper training is therefore permitted only
when two independent prerequisites are present:

1. the raw 5-minute SWMM Independent Oracle passes every admitted sample; and
2. every raw branch detail file contains all trajectory targets required by the
   trajectory-first hydraulic model, including explicit outfall flow.

This module makes that stop condition machine-readable.  It does not silently
upgrade stored-trajectory oracle evidence or partially covered raw detail data.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PAPER_CONTRACT_ID = "PROJECT6_V42_PAPER_WORKFLOW_V1"
FORMAL_MODEL_LINE = "v42_trajectory_first_multi_reference"


@dataclass(frozen=True)
class TrainingAdmission:
    authorized: bool
    reasons: tuple[str, ...]
    oracle_summary_sha256: str | None
    hydraulic_target_audit_sha256: str | None
    expected_sample_count: int
    admitted_sample_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract_id": PAPER_CONTRACT_ID,
            "model_line": FORMAL_MODEL_LINE,
            "authorized": self.authorized,
            "reasons": list(self.reasons),
            "expected_sample_count": self.expected_sample_count,
            "admitted_sample_count": self.admitted_sample_count,
            "oracle_summary_sha256": self.oracle_summary_sha256,
            "hydraulic_target_audit_sha256": self.hydraulic_target_audit_sha256,
            "legacy_v42_kpi_head_trainer_authorized": False,
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def audit_training_admission(
    *,
    independent_oracle_summary: str | Path,
    hydraulic_target_audit: str | Path,
    expected_sample_count: int = 1200,
) -> TrainingAdmission:
    """Require raw-oracle truth and complete hydraulic target coverage."""
    oracle_path = Path(independent_oracle_summary)
    target_path = Path(hydraulic_target_audit)
    reasons: list[str] = []
    oracle_sha: str | None = None
    target_sha: str | None = None
    admitted = 0

    if not oracle_path.exists():
        reasons.append("raw_independent_oracle_summary_missing")
        oracle: dict[str, Any] = {}
    else:
        oracle_sha = _sha256(oracle_path)
        try:
            oracle = _read_json(oracle_path)
        except Exception as exc:
            oracle = {}
            reasons.append(f"raw_independent_oracle_summary_unreadable:{type(exc).__name__}")

    if oracle:
        if oracle.get("audit_mode") != "raw":
            reasons.append("stored_oracle_cannot_authorize_formal_training")
        try:
            row_count = int(oracle.get("row_count", -1))
            pass_count = int(oracle.get("pass_count", -1))
            fail_count = int(oracle.get("fail_count", -1))
        except (TypeError, ValueError):
            row_count = pass_count = fail_count = -1
            reasons.append("oracle_counts_invalid")
        if row_count != int(expected_sample_count):
            reasons.append(
                f"oracle_row_count_mismatch:{row_count}!={int(expected_sample_count)}"
            )
        if pass_count != row_count or fail_count != 0 or oracle.get("all_pass") is not True:
            reasons.append("raw_independent_oracle_not_all_pass")
        if oracle.get("expected_count") not in (None, int(expected_sample_count)):
            reasons.append("oracle_expected_count_contract_mismatch")

    if not target_path.exists():
        reasons.append("hydraulic_target_audit_missing")
        target: dict[str, Any] = {}
    else:
        target_sha = _sha256(target_path)
        try:
            target = _read_json(target_path)
        except Exception as exc:
            target = {}
            reasons.append(f"hydraulic_target_audit_unreadable:{type(exc).__name__}")

    if target:
        try:
            detail_count = int(target.get("detail_count", -1))
            complete_count = int(target.get("formal_complete_count", -1))
        except (TypeError, ValueError):
            detail_count = complete_count = -1
            reasons.append("hydraulic_target_counts_invalid")
        if target.get("contract") not in (None, PAPER_CONTRACT_ID):
            reasons.append("hydraulic_target_audit_wrong_contract")
        if target.get("formal_complete") is not True:
            reasons.append("hydraulic_target_coverage_incomplete")
        if detail_count <= 0 or complete_count != detail_count:
            reasons.append("not_all_detail_files_have_formal_targets")
        required = set(target.get("required_target_groups", []))
        expected_groups = {
            "node_depth",
            "node_flooding_rate",
            "storage_volume",
            "managed_facility_flow",
            "outfall_flow",
        }
        if not expected_groups.issubset(required):
            reasons.append("hydraulic_target_group_contract_incomplete")
        admitted = min(
            int(expected_sample_count),
            max(0, complete_count),
        )

    return TrainingAdmission(
        authorized=not reasons,
        reasons=tuple(reasons),
        oracle_summary_sha256=oracle_sha,
        hydraulic_target_audit_sha256=target_sha,
        expected_sample_count=int(expected_sample_count),
        admitted_sample_count=int(admitted),
    )


def write_training_admission(
    *,
    output_path: str | Path,
    admission: TrainingAdmission,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(admission.as_dict(), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return path


def assert_formal_training_authorized(admission: TrainingAdmission) -> None:
    if not admission.authorized:
        raise RuntimeError(
            "Formal V4.2 paper training is not authorized: "
            + ";".join(admission.reasons)
        )
