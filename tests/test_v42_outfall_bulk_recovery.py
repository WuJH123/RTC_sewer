from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from sewerrtc.v4.v42_outfall_bulk_recovery import recover_outfall_sidecars


def _write_inp(path: Path) -> None:
    path.write_text(
        """[JUNCTIONS]\nJ1 0 5\n[OUTFALLS]\nO1 0 FREE\n[CONDUITS]\nC1 J1 O1 10 0.01 0 0 0 0\n""",
        encoding="utf-8",
    )


def _write_validation(path: Path, *, status: str) -> None:
    path.write_text(
        json.dumps(
            {
                "detail_path": "validation.csv",
                "inp_path": "m.inp",
                "atol_m3s": 1e-5,
                "rtol": 1e-5,
                "status": status,
                "rows": [
                    {
                        "outfall_id": "O1",
                        "incoming_links": ["C1"],
                        "sample_count": 2,
                        "max_abs_error_m3s": 0.0,
                        "rmse_m3s": 0.0,
                        "pass_tolerance": status == "pass",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_bulk_recovery_requires_passing_validation(tmp_path: Path):
    inventory = tmp_path / "physical.csv"
    pd.DataFrame(
        [
            {
                "physical_identity_sha256": "p1",
                "detail_path": str(tmp_path / "detail.csv"),
                "available_outfall_flow": False,
                "available_outfall_reconstruction_candidate": True,
            }
        ]
    ).to_csv(inventory, index=False)
    validation = tmp_path / "validation.json"
    _write_validation(validation, status="fail")
    with pytest.raises(RuntimeError):
        recover_outfall_sidecars(
            physical_inventory=inventory,
            validation_json=validation,
            inp_path=tmp_path / "m.inp",
            sidecar_dir=tmp_path / "sidecars",
            output_manifest=tmp_path / "manifest.csv",
        )


def test_bulk_recovery_writes_sidecar_without_editing_detail(tmp_path: Path):
    inp = tmp_path / "m.inp"
    _write_inp(inp)
    detail = tmp_path / "detail.csv"
    original = pd.DataFrame({"elapsed_min": [0.0, 5.0], "flow:C1": [1.0, 2.0]})
    original.to_csv(detail, index=False)
    inventory = tmp_path / "physical.csv"
    pd.DataFrame(
        [
            {
                "physical_identity_sha256": "p1",
                "detail_path": str(detail),
                "available_outfall_flow": False,
                "available_outfall_reconstruction_candidate": True,
            }
        ]
    ).to_csv(inventory, index=False)
    validation = tmp_path / "validation.json"
    _write_validation(validation, status="pass")
    result = recover_outfall_sidecars(
        physical_inventory=inventory,
        validation_json=validation,
        inp_path=inp,
        sidecar_dir=tmp_path / "sidecars",
        output_manifest=tmp_path / "manifest.csv",
    )
    assert result.loc[0, "status"] == "recovered_validated"
    sidecar = pd.read_parquet(result.loc[0, "sidecar_path"])
    assert sidecar["outfall_flow:O1"].tolist() == [1.0, 2.0]
    pd.testing.assert_frame_equal(pd.read_csv(detail), original)
