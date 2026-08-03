"""Restartable qualification-first orchestration for the V4.2 28-stage chain.

This runner accelerates software qualification by using a small, diverse subset
of already admitted development evidence.  It never writes Formal evidence and
never consumes untouched Calibration/Locked/Challenge/Formal-Blind rainfalls.
The expensive Formal production line remains unchanged and fail-closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION_CONTRACT = "PROJECT6_V42_QUALIFICATION_FIRST_PASS_V1"

STAGES = (
    "01_formal_source_ledger_prepare",
    "02_formal_step1_pool",
    "03_formal_step2_raw_readmission",
    "04_formal_evaluation_plan",
    "05_formal_r0_adapter",
    "06_step1_seed17",
    "07_step1_seed42",
    "08_step1_seed73",
    "09_causal_13frame_gat_history",
    "10_step2_seed17",
    "11_step2_seed42",
    "12_step2_seed73",
    "13_new_calibration_authoritative_swmm",
    "14_calibration_data_bridge",
    "15_step1_uncertainty_ood_calibration",
    "16_step2_pfv_peak_safety_calibration",
    "17_compile_step1_step2_evidence",
    "18_step3_authoritative_engineering_audit",
    "19_compile_step3_evidence",
    "20_true_state_offline",
    "21_exact_authoritative_swmm_closed_loop",
    "22_surrogate_closed_loop",
    "23_gat_integrated_closed_loop",
    "24_policy_lock",
    "25_challenge",
    "26_locked_validation",
    "27_qualification_blind_seven_strategies",
    "28_v42_qualification_audit",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(command: list[str], root: Path) -> None:
    print("\nRUN:", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(root), check=True)


def _pass_json(path: Path) -> bool:
    try:
        return path.exists() and _read_json(path).get("status") == "pass"
    except Exception:
        return False


def _status(root: Path, qualification: Path) -> dict[str, Any]:
    formal = root / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2"
    statuses = {stage: "NOT_STARTED" for stage in STAGES}
    formal_inputs = {
        "01_formal_source_ledger_prepare": formal / "prepare/FORMAL_F2_PREPARE_AUDIT.json",
        "02_formal_step1_pool": formal / "prepare/FORMAL_F2_STEP1_POOL_AUDIT.json",
        "03_formal_step2_raw_readmission": formal / "step2/FORMAL_F2_STEP2_RAW_ADMISSION_AUDIT.json",
        "04_formal_evaluation_plan": formal / "evaluation_plan/FORMAL_F2_EVALUATION_PLAN_AUDIT.json",
        "05_formal_r0_adapter": root / "outputs/project6_dual_reference_v4/final_v4/v42_paper/data_reuse/FORMAL_F2_R0_ADAPTER_AUDIT.json",
    }
    for stage, path in formal_inputs.items():
        statuses[stage] = "PASS_REUSABLE" if _pass_json(path) else "STALE_OR_INVALID"

    for seed, stage in ((17, "06_step1_seed17"), (42, "07_step1_seed42"), (73, "08_step1_seed73")):
        report = qualification / f"step1/seed_{seed}/qualification_step1_report.json"
        model = qualification / f"step1/seed_{seed}/best_model.pt"
        statuses[stage] = "PASS_REUSABLE" if _pass_json(report) and model.exists() else "NOT_STARTED"

    gat_audit = qualification / "step2/QUALIFICATION_GAT_HISTORY_AUDIT.json"
    statuses["09_causal_13frame_gat_history"] = "PASS_REUSABLE" if _pass_json(gat_audit) else "NOT_STARTED"
    for seed, stage in ((17, "10_step2_seed17"), (42, "11_step2_seed42"), (73, "12_step2_seed73")):
        report = qualification / f"step2/models/seed_{seed}/qualification_step2_report.json"
        model = qualification / f"step2/models/seed_{seed}/best_model.pt"
        statuses[stage] = "PASS_REUSABLE" if _pass_json(report) and model.exists() else "NOT_STARTED"

    passed = [stage for stage in STAGES if statuses[stage] == "PASS_REUSABLE"]
    next_stage = next((stage for stage in STAGES if statuses[stage] != "PASS_REUSABLE"), None)
    payload = {
        "contract_id": QUALIFICATION_CONTRACT,
        "qualification_only": True,
        "development_only": True,
        "formal_mainline_authorized": False,
        "stage_status": statuses,
        "passed_stage_count": len(passed),
        "next_stage": next_stage,
        "qualification_core_complete": all(statuses[stage] == "PASS_REUSABLE" for stage in STAGES[:12]),
        "formal_untouched_events_consumed": False,
        "formal_evidence_generated": False,
        "note": (
            "Stages 13-28 must be implemented/executed as micro authoritative qualification stages under this root. "
            "They must remain development-only and cannot satisfy Formal paper gates."
        ),
    }
    qualification.mkdir(parents=True, exist_ok=True)
    (qualification / "QUALIFICATION_28_STAGE_STATUS.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/v42_qualification_first_pass.json",
    )
    parser.add_argument("--stage", choices=("prepare", "step1", "step2", "core", "status"), default="status")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = _read_json(args.config)
    root = args.project_root
    py = sys.executable
    qualification = root / str(config["output_relative_root"])
    qualification.mkdir(parents=True, exist_ok=True)
    prepare_audit = qualification / "QUALIFICATION_PREPARE_AUDIT.json"
    step1_manifest = qualification / "QUALIFICATION_STEP1_WINDOW_MANIFEST.parquet"
    step2_raw = qualification / "QUALIFICATION_STEP2_RAW_MANIFEST.parquet"
    step2_gat = qualification / "step2/QUALIFICATION_STEP2_GAT_MANIFEST.parquet"

    def prepare() -> None:
        if not args.force and _pass_json(prepare_audit) and step1_manifest.exists() and step2_raw.exists():
            print("REUSE: qualification prepare", flush=True)
            return
        _run(
            [
                py,
                "-u",
                str(root / "scripts/build_v42_qualification_first_pass.py"),
                "--project-root",
                str(root),
                "--config",
                str(args.config),
                "--output-root",
                str(qualification),
            ],
            root,
        )

    def step1() -> None:
        prepare()
        training = config["training"]
        manifest_sha = _sha(step1_manifest)
        for seed in training["seeds"]:
            output = qualification / f"step1/seed_{seed}"
            report = output / "qualification_step1_report.json"
            if not args.force and _pass_json(report) and (output / "best_model.pt").exists():
                old = _read_json(report)
                if old.get("input_manifest_sha256") == manifest_sha:
                    print(f"REUSE: qualification Step1 seed {seed}", flush=True)
                    continue
            _run(
                [
                    py,
                    "-u",
                    str(root / "scripts/train_v42_step1_qualification.py"),
                    "--project-root",
                    str(root),
                    "--manifest",
                    str(step1_manifest),
                    "--output-dir",
                    str(output),
                    "--model-seed",
                    str(seed),
                    "--split-seed",
                    str(training["split_seed"]),
                    "--sensor-layout-seed",
                    str(training["sensor_layout_seed"]),
                    "--epochs",
                    str(training["step1_epochs"]),
                    "--batch-size",
                    str(training["step1_batch_size"]),
                    "--patience",
                    str(training["patience"]),
                    "--min-train-groups",
                    "65",
                ],
                root,
            )
            payload = _read_json(report)
            payload["input_manifest_sha256"] = manifest_sha
            report.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    def step2() -> None:
        step1()
        training = config["training"]
        primary = qualification / f"step1/seed_{training['primary_step1_seed']}"
        gat_audit = qualification / "step2/QUALIFICATION_GAT_HISTORY_AUDIT.json"
        if args.force or not (_pass_json(gat_audit) and step2_gat.exists()):
            _run(
                [
                    py,
                    "-u",
                    str(root / "scripts/materialize_v42_qualification_gat_history.py"),
                    "--project-root",
                    str(root),
                    "--input-manifest",
                    str(step2_raw),
                    "--step1-window-manifest",
                    str(step1_manifest),
                    "--step1-model-dir",
                    str(primary),
                    "--output-manifest",
                    str(step2_gat),
                    "--min-rainfall-groups",
                    "65",
                    "--sensor-layout-seed",
                    str(training["sensor_layout_seed"]),
                ],
                root,
            )
        else:
            print("REUSE: qualification causal GAT history", flush=True)

        gat_sha = _sha(step2_gat)
        for seed in training["seeds"]:
            output = qualification / f"step2/models/seed_{seed}"
            report = output / "qualification_step2_report.json"
            if not args.force and _pass_json(report) and (output / "best_model.pt").exists():
                old = _read_json(report)
                if old.get("input_manifest_sha256") == gat_sha:
                    print(f"REUSE: qualification Step2 seed {seed}", flush=True)
                    continue
            _run(
                [
                    py,
                    "-u",
                    str(root / "scripts/train_v42_step2_qualification.py"),
                    "--project-root",
                    str(root),
                    "--manifest",
                    str(step2_gat),
                    "--output-dir",
                    str(output),
                    "--seed",
                    str(seed),
                    "--epochs",
                    str(training["step2_epochs"]),
                    "--batch-size",
                    str(training["step2_batch_size"]),
                    "--patience",
                    str(training["patience"]),
                ],
                root,
            )
            payload = _read_json(report)
            payload["input_manifest_sha256"] = gat_sha
            report.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    if args.stage == "prepare":
        prepare()
    elif args.stage == "step1":
        step1()
    elif args.stage == "step2":
        step2()
    elif args.stage == "core":
        step2()

    status = _status(root, qualification)
    print(json.dumps(status, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
