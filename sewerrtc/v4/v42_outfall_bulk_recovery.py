"""Bulk sidecar extraction of outfall flow from validated incoming-link sums.

This is a post-validation utility.  It never edits historical `detail.csv` and
never promotes structural candidates without a previously passing independent
validation report created from a detail file that contains explicit outfall
recorder columns.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sewerrtc.v4.v42_outfall_recovery import (
    OutfallValidationResult,
    OutfallValidationRow,
    reconstruct_outfall_flow,
)


def _load_validation(path: Path) -> OutfallValidationResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = tuple(OutfallValidationRow(**row) for row in payload.get("rows", []))
    return OutfallValidationResult(
        detail_path=str(payload.get("detail_path", "")),
        inp_path=str(payload.get("inp_path", "")),
        atol_m3s=float(payload.get("atol_m3s", 0.0)),
        rtol=float(payload.get("rtol", 0.0)),
        status=str(payload.get("status", "fail")),
        rows=rows,
    )


def recover_outfall_sidecars(
    *,
    physical_inventory: str | Path,
    validation_json: str | Path,
    inp_path: str | Path,
    sidecar_dir: str | Path,
    output_manifest: str | Path,
) -> pd.DataFrame:
    physical_path = Path(physical_inventory)
    physical = pd.read_parquet(physical_path) if physical_path.suffix.lower() == ".parquet" else pd.read_csv(physical_path)
    validation = _load_validation(Path(validation_json))
    if validation.status != "pass":
        raise RuntimeError("outfall recovery validation has not passed")
    required = {
        "physical_identity_sha256",
        "detail_path",
        "available_outfall_flow",
        "available_outfall_reconstruction_candidate",
    }
    if not required.issubset(physical.columns):
        raise KeyError(f"physical inventory missing columns: {sorted(required - set(physical.columns))}")
    sidecar_dir = Path(sidecar_dir)
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for row in physical.itertuples(index=False):
        if bool(getattr(row, "available_outfall_flow")):
            rows.append(
                {
                    "physical_identity_sha256": str(row.physical_identity_sha256),
                    "status": "explicit_already_available",
                    "sidecar_path": "",
                    "source": "explicit_detail_column",
                }
            )
            continue
        if not bool(getattr(row, "available_outfall_reconstruction_candidate")):
            rows.append(
                {
                    "physical_identity_sha256": str(row.physical_identity_sha256),
                    "status": "not_recoverable_from_recorded_links",
                    "sidecar_path": "",
                    "source": "missing_complete_incoming_link_set",
                }
            )
            continue
        detail_path = Path(str(row.detail_path))
        try:
            reconstructed = reconstruct_outfall_flow(
                detail_path,
                inp_path=inp_path,
                validated_result=validation,
            )
            header = pd.read_csv(detail_path, nrows=0)
            if "elapsed_min" not in header.columns:
                raise KeyError("detail missing elapsed_min")
            elapsed = pd.read_csv(detail_path, usecols=["elapsed_min"])
            if len(elapsed) != len(reconstructed):
                raise ValueError("reconstructed outfall length mismatch")
            sidecar = pd.concat([elapsed.reset_index(drop=True), reconstructed.reset_index(drop=True)], axis=1)
            target = sidecar_dir / f"{row.physical_identity_sha256}_outfall.parquet"
            sidecar.to_parquet(target, index=False)
            rows.append(
                {
                    "physical_identity_sha256": str(row.physical_identity_sha256),
                    "status": "recovered_validated",
                    "sidecar_path": str(target),
                    "source": "validated_sum_of_all_incoming_link_flows",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "physical_identity_sha256": str(row.physical_identity_sha256),
                    "status": "recovery_failed",
                    "sidecar_path": "",
                    "source": f"{type(exc).__name__}: {exc}",
                }
            )
    result = pd.DataFrame(rows)
    output_manifest = Path(output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    if output_manifest.suffix.lower() == ".parquet":
        result.to_parquet(output_manifest, index=False)
    else:
        result.to_csv(output_manifest, index=False)
    return result
