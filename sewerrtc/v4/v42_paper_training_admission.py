"""Admission gate for formal V4.2 trajectory-surrogate training.

Formal training is authorized only when the raw Independent Oracle passes every
admitted sample and the audited raw hydraulic targets are complete.  The gate is
population based: it does not assume that the final evidence pool contains a
fixed 1200/1600 rows.
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
    expected_sample_count: int | None
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
            "fixed_sample_quota_required": False,
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
    expected_sample_count: int | None = None,
) -> TrainingAdmission:
    """Require raw-oracle truth and complete hydraulic target coverage.

    ``expected_sample_count`` is optional and exists only for a deliberately
    frozen experiment.  The project-wide R0 pool must not be forced back to a
    historical Train1200/Train1600 quota.
    """
    oracle_path = Path(independent_oracle_summary)
    target_path = Path(hydraulic_target_audit)
    reasons: list[str] = []
    oracle_sha: str | None = None
    target_sha: str | None = None
    admitted = 0
    oracle_row_count = -1

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
            oracle_row_count = int(oracle.get("row_count", -1))
            pass_count = int(oracle.get("pass_count", -1))
            fail_count = int(oracle.get("fail_count", -1))
        except (TypeError, ValueError):
            oracle_row_count = pass_count = fail_count = -1
            reasons.append("oracle_counts_invalid")
        if oracle_row_count <= 0:
            reasons.append("raw_independent_oracle_empty")
        if expected_sample_count is not None and oracle_row_count != int(expected_sample_count):
            reasons.append(
                f"oracle_row_count_mismatch:{oracle_row_count}!={int(expected_sample_count)}"
            )
        if (
            pass_count != oracle_row_count
            or fail_count != 0
            or oracle.get("all_pass") is not True
        ):
            reasons.append("raw_independent_oracle_not_all_pass")
        declared_expected = oracle.get("expected_count")
        if (
            expected_sample_count is not None
            and declared_expected not in (None, int(expected_sample_count))
        ):
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

        # When both independent audits carry an explicit population lineage,
        # they must match.  Absence remains backward-compatible until the R0
        # materializer emits this field everywhere.
        oracle_lineage = str(oracle.get("sample_lineage_sha256", "")) if oracle else ""
        target_lineage = str(target.get("sample_lineage_sha256", ""))
        if oracle_lineage and target_lineage and oracle_lineage != target_lineage:
            reasons.append("oracle_target_population_lineage_mismatch")

    if not reasons and oracle_row_count > 0:
        admitted = int(oracle_row_count)

    return TrainingAdmission(
        authorized=not reasons,
        reasons=tuple(reasons),
        oracle_summary_sha256=oracle_sha,
        hydraulic_target_audit_sha256=target_sha,
        expected_sample_count=(
            None if expected_sample_count is None else int(expected_sample_count)
        ),
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
