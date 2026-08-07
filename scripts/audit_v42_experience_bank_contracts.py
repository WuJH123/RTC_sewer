"""Cheap, fail-closed authority audit for the V4.2 experience-bank inputs.

This script only reads manifests, split records, and the existing No-control
detail files.  It never starts SWMM and never uses stored KPI labels as an
authority.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.v4.v42_trajectory_builder import _load_engineering36_ids


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_parquet(path: Path, columns: list[str]) -> pd.DataFrame:
    import pyarrow.parquet as pq

    names = set(pq.ParquetFile(path).schema.names)
    missing = sorted(set(columns) - names)
    if missing:
        raise KeyError(f"{path} missing columns: {missing}")
    return pd.read_parquet(path, columns=columns)


def _groups_from_csv(path: Path) -> set[str]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path, usecols=lambda c: c in {"rainfall_sha256", "rainfall_group_key"})
    values: set[str] = set()
    for column in ("rainfall_sha256", "rainfall_group_key"):
        if column in frame:
            values.update(str(x).strip() for x in frame[column].dropna() if str(x).strip())
    return values


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _audit_no_control(root: Path, candidates: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    facility_ids = _load_engineering36_ids(root)
    setting_columns = [f"setting:{item}" for item in facility_ids]
    path_checkpoints: dict[str, set[float]] = {}
    for _, row in candidates[["source_detail_path_no_control", "checkpoint_min"]].dropna().iterrows():
        path = str(row["source_detail_path_no_control"]).strip()
        if path:
            path_checkpoints.setdefault(path, set()).add(float(row["checkpoint_min"]))
    paths = sorted(path_checkpoints)
    rows: list[dict[str, Any]] = []
    for path_text in paths:
        path = Path(path_text)
        result: dict[str, Any] = {"detail_path": path_text, "exists": path.exists()}
        if not path.exists():
            result["pass"] = False
            result["error"] = "missing_detail"
            rows.append(result)
            continue
        try:
            header = pd.read_csv(path, nrows=0)
            missing = [column for column in setting_columns if column not in header.columns]
            if missing:
                result.update({"pass": False, "error": "missing_setting_columns", "missing": missing})
                rows.append(result)
                continue
            frame = pd.read_csv(path, usecols=["elapsed_min", *setting_columns], low_memory=False)
            checks = []
            for checkpoint_min in sorted(path_checkpoints[path_text]):
                post = frame[frame["elapsed_min"].astype(float) >= checkpoint_min - 1.0e-7]
                values = post[setting_columns].to_numpy(dtype=float)
                finite = bool(values.size and np.isfinite(values).all())
                all_open = bool(finite and np.allclose(values, 1.0, atol=1.0e-7, rtol=0.0))
                checks.append(
                    {
                        "checkpoint_min": checkpoint_min,
                        "prefix_rows_excluded": int(len(frame) - len(post)),
                        "post_action_rows": int(len(post)),
                        "finite": finite,
                        "all_engineering36_settings_equal_1": all_open,
                        "min_setting": float(np.nanmin(values)) if values.size else None,
                        "max_setting": float(np.nanmax(values)) if values.size else None,
                        "pass": bool(len(post) > 0 and all_open),
                    }
                )
            result.update(
                {
                    "rows": int(len(frame)),
                    "setting_columns": int(len(setting_columns)),
                    "checks": checks,
                    "pass": bool(checks and all(bool(item["pass"]) for item in checks)),
                }
            )
        except Exception as exc:
            result.update({"pass": False, "error": repr(exc)})
        rows.append(result)
    report = {
        "contract": "PROJECT6_V42_NO_CONTROL_ALL_OPEN_V1",
        "status": "pass" if paths and all(bool(row.get("pass")) for row in rows) else "fail",
        "definition": "all_engineering36_settings_equal_1.0_after_checkpoint_min",
        "engineering36_count": len(facility_ids),
        "unique_no_control_detail_files": len(paths),
        "passed_detail_files": sum(bool(row.get("pass")) for row in rows),
        "rows": rows,
    }
    _write(output_dir / "NO_CONTROL_ALL_OPEN_AUTHORITY.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--state-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    base = args.project_root / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2"
    candidate_columns = [
        "state_key", "event_id", "rainfall_sha256", "checkpoint_min", "source_detail_path_no_control",
        "source_detail_path_candidate", "source_detail_path_dynamic_internal",
        "state_source", "history_input_contract", "reconstructor_contract",
        "reconstructed_history_contract", "authoritative_swmm_history_used_as_online_input",
        "realized_future_rainfall_used_online", "no_control_all_open_verified",
        "no_control_action_contract", "candidate_action_sha256", "action_candidate_readback",
    ]
    candidates = _read_parquet(args.candidate_manifest, candidate_columns)
    state_columns = ["state_key", "rainfall_sha256", "history_depth", "rainfall_forecast", "action_hold_previous_readback", "sensor_layout_sha256"]
    states = _read_parquet(args.state_manifest, state_columns)
    candidates["state_key"] = candidates["state_key"].astype(str)
    states["state_key"] = states["state_key"].astype(str)
    bank_groups = set(str(x).strip() for x in candidates["rainfall_sha256"].dropna() if str(x).strip())

    split_path = base / "diagnostics/pfv_tfv_pareto_baseline_corrected/STEP2_SPLIT_AND_CALIBRATION_AUTHORITY.json"
    split = _json(split_path)
    model_groups = split["model_reports"][0]["groups"]
    pfv_cal = _json(base / "calibration/PFV_ONLY_SAFETY_CALIBRATION.json").get("calibration_rainfall_groups", [])
    step1_cal = _json(base / "calibration/STEP1_UNCERTAINTY_OOD_CALIBRATION.json").get("calibration_rainfall_groups", [])
    reserved: dict[str, set[str]] = {
        "pfv_conformal_calibration12": set(map(str, pfv_cal)),
        "step1_uncertainty_calibration12": set(map(str, step1_cal)),
        "challenge": _groups_from_csv(base / "evaluation_inputs/challenge_case_manifest.csv"),
        "locked_validation": _groups_from_csv(base / "evaluation_inputs/locked_validation_case_manifest.csv"),
        "formal_blind": _groups_from_csv(base / "evaluation_inputs/formal_blind_case_manifest.csv"),
    }
    overlap = {name: sorted(bank_groups & values) for name, values in reserved.items()}
    signature_columns = [
        "state_source", "history_input_contract", "reconstructor_contract",
        "reconstructed_history_contract", "authoritative_swmm_history_used_as_online_input",
        "realized_future_rainfall_used_online",
    ]
    signature_values = {
        column: sorted(str(x) for x in candidates[column].dropna().unique())
        for column in signature_columns
    }
    flag_pass = {
        "state_source_gat_sparse_reconstruction": set(signature_values["state_source"]) == {"gat_sparse_reconstruction"},
        "history_input_contract_causal": set(signature_values["history_input_contract"]) == {"gat_compatible_causal_state"},
        "reconstructor_formal_temporal_v42": set(signature_values["reconstructor_contract"]) == {"formal_temporal_v42"},
        "reconstructed_history_contract_present": len(signature_values["reconstructed_history_contract"]) == 1,
        "no_authoritative_future_hydraulic_truth": set(signature_values["authoritative_swmm_history_used_as_online_input"]) <= {"False", "false", "0"},
        "no_realized_future_rainfall": set(signature_values["realized_future_rainfall_used_online"]) <= {"False", "false", "0"},
    }
    split_report = {
        "contract": "PROJECT6_V42_EXPERIENCE_BANK_SPLIT_AUDIT_V1",
        "status": "pass" if not any(overlap.values()) else "fail",
        "candidate_input_rows": int(len(candidates)),
        "candidate_input_states": int(candidates["state_key"].nunique()),
        "candidate_input_rainfall_groups": len(bank_groups),
        "model_split_counts": {key: len(value) for key, value in model_groups.items()},
        "model_split_overlaps": {
            "train__validation": len(set(model_groups["train_rainfall_groups"]) & set(model_groups["validation_rainfall_groups"])),
            "train__calibration": len(set(model_groups["train_rainfall_groups"]) & set(model_groups["calibration_rainfall_groups"])),
            "validation__calibration": len(set(model_groups["validation_rainfall_groups"]) & set(model_groups["calibration_rainfall_groups"])),
        },
        "reserved_group_counts": {key: len(value) for key, value in reserved.items()},
        "bank_reserved_overlaps": {key: values for key, values in overlap.items()},
        "independent_prelock_safety_audit_present": False,
        "note": "No independent pre-lock safety audit is admitted to the bank; it has not yet been generated.",
    }
    signature_report = {
        "contract": "PROJECT6_V42_CAUSAL_EXPERIENCE_SIGNATURE_V1",
        "status": "pass" if all(flag_pass.values()) else "fail",
        "representation": "gat_sparse_reconstruction",
        "state_history": "causal reconstructed history at decision time",
        "rainfall_input": "causal rainfall forecast available at decision time",
        "action_input": "current readback action at decision time",
        "future_hydraulic_truth_used": False,
        "realized_future_rainfall_used": False,
        "signature_columns": signature_values,
        "checks": flag_pass,
        "sensor_layout_sha256": sorted(str(x) for x in states["sensor_layout_sha256"].dropna().unique()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write(args.output_dir / "EXPERIENCE_BANK_SPLIT_AUDIT.json", split_report)
    _write(args.output_dir / "EXPERIENCE_SIGNATURE_CONTRACT.json", signature_report)
    no_control = _audit_no_control(args.project_root, candidates, args.output_dir)
    status = "pass" if split_report["status"] == "pass" and signature_report["status"] == "pass" and no_control["status"] == "pass" else "fail"
    summary = {
        "contract": "PROJECT6_V42_EXPERIENCE_BANK_AUTHORITY_AUDIT_V1",
        "status": status,
        "split_status": split_report["status"],
        "signature_status": signature_report["status"],
        "no_control_status": no_control["status"],
        "candidate_rows": int(len(candidates)),
        "states": int(candidates["state_key"].nunique()),
        "rainfall_groups": len(bank_groups),
    }
    _write(args.output_dir / "EXPERIENCE_BANK_AUTHORITY_AUDIT_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
