"""Strictly adapt fresh PFV-only physical cases to the raw-readmission contract.

Unlike the first adapter, this module never marks scientific gates true merely
because a row exists in the fresh case manifest. It rechecks the persisted
completion evidence, frozen physical-network identity, no-hotstart contract and
Engineering36 executability of the actually recorded H3 readback before a case
can be handed to the shared Formal raw materializer.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.simulation.pyswmm_runner import physical_network_sha256
from sewerrtc.v4.formal_f2 import sha256_file
from sewerrtc.v4.v42_formal_runtime import (
    CONTROLLABLE_PREFIX_STEPS,
    HORIZON_STEPS,
    load_actuators,
    project_candidate_sequence,
)


def _completion_path(candidate_detail: Path, case_id: str) -> Path:
    # fresh_calibration_inputs/<event>/details/<detail>.csv
    event_dir = candidate_detail.resolve().parent.parent
    return event_dir / "completions" / str(case_id) / "completion.json"


def _branch_detail(completion: dict, role: str) -> Path:
    branches = completion.get("branches", {})
    value = branches.get(role, {}) if isinstance(branches, dict) else {}
    raw = value.get("detail_path") if isinstance(value, dict) else None
    if not raw:
        raise KeyError(f"completion missing branch detail: {role}")
    path = Path(str(raw)).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _read_readback(path: Path, actuator_ids: list[str]) -> pd.DataFrame:
    columns = ["elapsed_min", *[f"readback_setting:{aid}" for aid in actuator_ids]]
    header = pd.read_csv(path, nrows=0)
    missing = [column for column in columns if column not in header.columns]
    if missing:
        raise KeyError(f"{path}: missing readback columns {missing[:5]}")
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if numeric.empty or not np.isfinite(numeric.to_numpy(float)).all():
        raise RuntimeError(f"{path}: non-finite readback trajectory")
    return numeric


def _row_at(frame: pd.DataFrame, elapsed: float, actuator_ids: list[str]) -> np.ndarray:
    times = frame["elapsed_min"].to_numpy(float)
    idx = np.flatnonzero(np.isclose(times, float(elapsed), atol=1.0e-6, rtol=0.0))
    if len(idx) != 1:
        raise RuntimeError(f"expected one readback row at elapsed_min={elapsed}; got {len(idx)}")
    return frame.iloc[int(idx[0])][[f"readback_setting:{aid}" for aid in actuator_ids]].to_numpy(float)


def _engineering_audit(
    *,
    completion: dict,
    checkpoint: float,
    actuators: pd.DataFrame,
) -> dict:
    ids = actuators["actuator_id"].astype(str).tolist()
    candidate_path = _branch_detail(completion, "candidate")
    no_control_path = _branch_detail(completion, "no_control")
    candidate = _read_readback(candidate_path, ids)
    no_control = _read_readback(no_control_path, ids)

    prefix = no_control.loc[no_control["elapsed_min"] < float(checkpoint) - 1.0e-9]
    if prefix.empty:
        raise RuntimeError("fresh candidate lacks a pre-action No-control readback")
    current = prefix.iloc[-1][[f"readback_setting:{aid}" for aid in ids]].to_numpy(float)
    h3 = np.stack(
        [
            _row_at(candidate, float(checkpoint) + 10.0 * step, ids)
            for step in range(CONTROLLABLE_PREFIX_STEPS)
        ],
        axis=0,
    )
    sequence = np.repeat(current[None, :], HORIZON_STEPS, axis=0)
    sequence[:CONTROLLABLE_PREFIX_STEPS] = h3
    projected, engineering, k_count, executable = project_candidate_sequence(
        sequence.astype(np.float32), current.astype(np.float32), actuators
    )
    projection_match = bool(
        np.allclose(
            projected[:CONTROLLABLE_PREFIX_STEPS],
            h3,
            atol=1.0e-6,
            rtol=0.0,
        )
    )
    return {
        "engineering_pass": bool(engineering.passed),
        "executable": bool(executable),
        "projection_match": projection_match,
        "actual_k": int(k_count),
        "h3_action_sha256": __import__("hashlib").sha256(
            np.ascontiguousarray(h3, dtype=np.float64).tobytes(order="C")
        ).hexdigest(),
        "candidate_detail_sha256": sha256_file(candidate_path),
        "no_control_detail_sha256": sha256_file(no_control_path),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=root)
    ap.add_argument(
        "--case-manifest",
        type=Path,
        default=root
        / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/pfv_only_v2/FRESH_PFV_ONLY_CALIBRATION_CASE_MANIFEST.csv",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=root
        / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/pfv_only_v2",
    )
    args = ap.parse_args()

    frame = pd.read_csv(args.case_manifest)
    required = {
        "case_id",
        "event_id",
        "rainfall_sha256",
        "rainfall_group_key",
        "checkpoint_min",
        "candidate_detail_path",
        "history_detail_path",
        "physical_network_sha256",
        "hotstart_used",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"fresh case manifest missing columns: {missing}")
    if len(frame) != 36 or frame["case_id"].astype(str).nunique() != 36:
        raise RuntimeError("fresh raw pool requires exactly 36 unique candidate cases")
    if frame["rainfall_sha256"].astype(str).nunique() != 12:
        raise RuntimeError("fresh raw pool requires exactly 12 independent rainfall groups")

    expected_physical = physical_network_sha256(
        args.project_root / "data/wuhan_v8_storage_retrofit.inp"
    )
    actuators = load_actuators(args.project_root)
    audited_rows: list[dict] = []
    failures: list[dict] = []

    for position, row in enumerate(frame.to_dict("records")):
        case_id = str(row["case_id"])
        try:
            candidate_detail = Path(str(row["candidate_detail_path"])).resolve()
            history_detail = Path(str(row["history_detail_path"])).resolve()
            if not candidate_detail.is_file() or not history_detail.is_file():
                raise FileNotFoundError(f"missing candidate/history detail for {case_id}")
            completion_path = _completion_path(candidate_detail, case_id)
            if not completion_path.is_file():
                raise FileNotFoundError(completion_path)
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            if str(completion.get("case_id", "")) != case_id:
                raise RuntimeError("completion case_id mismatch")

            checkpoint = float(row["checkpoint_min"])
            physical_ok = bool(
                str(row["physical_network_sha256"]) == str(expected_physical)
                and str(completion.get("physical_network_sha256", ""))
                == str(expected_physical)
            )
            no_hotstart = bool(
                not bool(row.get("hotstart_used", True))
                and not bool(completion.get("hotstart_used", True))
            )
            same_state = completion.get("same_state_raw_verified") is True
            same_forcing = completion.get("same_forcing_raw_verified") is True
            readback = completion.get("actual_readback_verified") is True
            h120 = completion.get("h120_window_complete") is True
            kpi = completion.get("kpi_recompute_ok") is True
            engineering = _engineering_audit(
                completion=completion,
                checkpoint=checkpoint,
                actuators=actuators,
            )
            actuator_semantics_ok = bool(
                engineering["engineering_pass"]
                and engineering["executable"]
                and engineering["projection_match"]
            )
            checks = {
                "physical_sha_ok": physical_ok,
                "no_hotstart": no_hotstart,
                "same_state_raw_verified": bool(same_state),
                "same_forcing_raw_verified": bool(same_forcing),
                "actual_readback_verified": bool(readback),
                "h120_window_complete": bool(h120),
                "kpi_recompute_ok": bool(kpi),
                "actuator_semantics_ok": actuator_semantics_ok,
            }
            if not all(checks.values()):
                raise RuntimeError(
                    "fresh raw admission failed: "
                    + ",".join(key for key, value in checks.items() if not value)
                )

            source = dict(row)
            source.update(checks)
            source.update(engineering)
            source.update(
                {
                    "source_dataset": "pfv_only_fresh_calibration",
                    "formal_f2_role": "fresh_pfv_only_calibration",
                    "step2_accepted_from_manifest": True,
                    "raw_readmission_pending": False,
                    "training_admission_authorized": True,
                    "raw_independent_oracle_all_pass": True,
                    "completion_path": str(completion_path.resolve()),
                    "completion_sha256": sha256_file(completion_path),
                    "source_row_number": int(position),
                }
            )
            audited_rows.append(source)
        except Exception as exc:
            failures.append(
                {
                    "case_id": case_id,
                    "source_row_number": int(position),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    if failures or len(audited_rows) != len(frame):
        args.output_dir.mkdir(parents=True, exist_ok=True)
        audit = {
            "status": "fail",
            "formal_mainline_authorized": False,
            "input_case_manifest": str(args.case_manifest.resolve()),
            "input_case_manifest_sha256": sha256_file(args.case_manifest),
            "rows": int(len(frame)),
            "accepted_rows": int(len(audited_rows)),
            "failed_rows": int(len(failures)),
            "failure_examples": failures[:100],
            "raw_admission_authorized": False,
            "model_training_authorized": False,
            "admission_authority": "derived_from_completion_physics_and_engineering36_projection",
        }
        (args.output_dir / "FRESH_PFV_ONLY_RAW_POOL_AUDIT.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)
        return 3

    source = pd.DataFrame(audited_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_path = args.output_dir / "FRESH_PFV_ONLY_SOURCE_MANIFEST.csv"
    source.to_csv(source_path, index=False)
    metadata = source[
        [
            "case_id",
            "source_dataset",
            "checkpoint_min",
            "rainfall_group_key",
            "source_row_number",
        ]
    ].copy()
    metadata["source_id"] = "pfv_only_fresh_calibration"
    metadata["source_manifest"] = str(source_path.resolve())
    metadata["source_manifest_sha256"] = sha256_file(source_path)
    metadata_path = args.output_dir / "FRESH_PFV_ONLY_METADATA_POOL.parquet"
    metadata.to_parquet(metadata_path, index=False)
    audit = {
        "status": "pass",
        "formal_mainline_authorized": False,
        "input_case_manifest": str(args.case_manifest.resolve()),
        "input_case_manifest_sha256": sha256_file(args.case_manifest),
        "source_manifest": str(source_path.resolve()),
        "source_manifest_sha256": sha256_file(source_path),
        "metadata_pool": str(metadata_path.resolve()),
        "rows": int(len(source)),
        "rainfall_groups": int(source["rainfall_sha256"].astype(str).nunique()),
        "all_detail_paths_exist": True,
        "raw_admission_authorized": True,
        "model_training_authorized": False,
        "admission_authority": "derived_from_completion_physics_and_engineering36_projection",
        "engineering36_projection_verified": True,
        "self_asserted_scientific_gates": False,
    }
    (args.output_dir / "FRESH_PFV_ONLY_RAW_POOL_AUDIT.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
