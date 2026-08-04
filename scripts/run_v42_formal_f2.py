"""Restartable orchestration for Project6 V4.2 Formal F2.

Stages are independently restartable:
prepare -> Step1(3 seeds) -> causal GAT -> CONTROL_CORE target materialisation ->
Step2(3 seeds) -> current-generation Calibration -> evidence.

The default Formal Step2 target contract is CONTROL_CORE. FULL_HYDRAULIC is an
optional extension and must be requested explicitly after outfall targets exist.
This runner never fabricates later closed-loop/Policy-Lock/evaluation evidence.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.v4.formal_f2 import FORMAL_GENERATION_ID


def _run(cmd: list[str], root: Path) -> None:
    print("\nRUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(root), check=True)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _status(root: Path) -> dict[str, Any]:
    formal = root / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2"
    paths = {
        "prepare": formal / "prepare/FORMAL_F2_PREPARE_AUDIT.json",
        "step1_pool": formal / "prepare/FORMAL_F2_STEP1_POOL_AUDIT.json",
        "raw_step2": formal / "step2/FORMAL_F2_STEP2_RAW_ADMISSION_AUDIT.json",
        "gat_step2": formal / "step2/FORMAL_F2_STEP2_GAT_HISTORY_AUDIT.json",
        "target_step2": formal / "step2/FORMAL_F2_STEP2_CONTROL_CORE_MANIFEST_TARGET_AUDIT.json",
        "eval_plan": formal / "evaluation_plan/FORMAL_F2_EVALUATION_PLAN_AUDIT.json",
        "step1_calibration": formal / "calibration/STEP1_UNCERTAINTY_OOD_CALIBRATION.json",
        "step2_calibration": formal / "calibration/STEP2_SAFETY_CALIBRATION.json",
        "training_evidence": formal / "FORMAL_F2_TRAINING_EVIDENCE_COMPILE.json",
    }
    s1 = sorted((formal / "step1").glob("seed_*/formal_step1_report.json"))
    s2 = sorted((formal / "step2/models").glob("seed_*/formal_step2_report.json"))
    payload: dict[str, Any] = {
        "formal_generation_id": FORMAL_GENERATION_ID,
        "split_policy": "current_generation_rainfall_group_holdout",
        "default_step2_target_contract": "CONTROL_CORE",
        **{key: _json(path) if path.exists() else None for key, path in paths.items()},
        "step1_seed_reports": [str(x) for x in s1],
        "step2_seed_reports": [str(x) for x in s2],
        "formal_mainline_authorized": False,
    }
    structural_reasons: list[str] = []
    for key in ("prepare", "step1_pool", "raw_step2", "eval_plan"):
        item = payload.get(key)
        if not item or item.get("status") != "pass":
            structural_reasons.append(f"{key}_not_pass")
    if len(s1) < 3:
        structural_reasons.append("formal_step1_requires_three_model_seeds")
    if payload.get("gat_step2") is None or payload["gat_step2"].get("status") != "pass":
        structural_reasons.append("formal_step2_gat_history_not_pass")
    if payload.get("target_step2") is None or payload["target_step2"].get("status") != "pass":
        structural_reasons.append("formal_step2_control_core_targets_not_pass")
    if len(s2) < 3:
        structural_reasons.append("formal_step2_requires_three_model_seeds")
    step1_reports = [_json(x) for x in s1]
    if step1_reports:
        if len({tuple(x.get("train_rainfall_groups", [])) for x in step1_reports}) != 1:
            structural_reasons.append("step1_split_changes_with_model_seed")
        if any(int(x.get("train_rainfall_group_count", 0)) < 65 for x in step1_reports):
            structural_reasons.append("step1_train_rainfall_groups_below_65")
        if any(x.get("uses_future_hydraulic_truth") is not False for x in step1_reports):
            structural_reasons.append("step1_future_truth_contract_violation")
    step2_reports = [_json(x) for x in s2]
    if step2_reports:
        if len({tuple(x.get("train_rainfall_groups", [])) for x in step2_reports}) != 1:
            structural_reasons.append("step2_split_changes_with_model_seed")
        if any(int(x.get("train_rainfall_group_count", 0)) < 65 for x in step2_reports):
            structural_reasons.append("step2_train_rainfall_groups_below_65")
        if any(x.get("raw_independent_oracle_all_pass") is not True for x in step2_reports):
            structural_reasons.append("step2_raw_oracle_not_all_pass")
        if any(x.get("step2_target_contract") not in {"CONTROL_CORE", "FULL_HYDRAULIC"} for x in step2_reports):
            structural_reasons.append("step2_target_contract_missing_or_invalid")
        if any(x.get("storage_supervised") is not True or x.get("facility_flow_supervised") is not True for x in step2_reports):
            structural_reasons.append("step2_control_core_supervision_incomplete")
    payload["structural_training_chain_pass"] = not structural_reasons
    payload["structural_reasons"] = structural_reasons

    calibration_reasons: list[str] = []
    if not payload.get("step1_calibration") or payload["step1_calibration"].get("status") != "pass":
        calibration_reasons.append("step1_current_calibration_uncertainty_ood_not_pass")
    if not payload.get("step2_calibration") or payload["step2_calibration"].get("status") != "pass":
        calibration_reasons.append("step2_current_calibration_pfv_depth_safety_not_pass")
    payload["calibration_chain_pass"] = not calibration_reasons
    payload["calibration_reasons"] = calibration_reasons
    payload["training_evidence_compiled"] = bool(
        payload.get("training_evidence")
        and payload["training_evidence"].get("status") == "pass"
    )
    payload["next_required_stages"] = [
        "true_state_offline_validation",
        "authoritative exact SWMM closed loop",
        "surrogate closed loop",
        "GAT-integrated closed loop",
        "policy lock",
        "challenge current-generation holdout",
        "one-shot locked validation current-generation holdout",
        "final held-out >=24 rainfall groups with Proposed/EFD/Auto-RBC/All-close/No-control/Internal/Hold authoritative SWMM",
    ]
    formal.mkdir(parents=True, exist_ok=True)
    (formal / "FORMAL_F2_STATUS.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument(
        "--stage",
        choices=("prepare", "step1", "step2", "calibration", "evidence", "audit", "all"),
        default="prepare",
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 73])
    ap.add_argument("--primary-step1-seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--sensor-layout-seed", type=int, default=42)
    ap.add_argument(
        "--step2-target-contract",
        choices=("CONTROL_CORE", "FULL_HYDRAULIC"),
        default="CONTROL_CORE",
    )
    ap.add_argument("--step1-cache-dir", type=Path, default=None)
    ap.add_argument("--step1-num-workers", type=int, default=0)
    ap.add_argument("--step1-persistent-workers", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--step1-prefetch-factor", type=int, default=2)
    ap.add_argument("--step1-pin-memory", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--step1-runtime-status-dir", type=Path, default=None)
    ap.add_argument("--step2-runtime-status-dir", type=Path, default=None)
    args = ap.parse_args()
    root = args.project_root
    py = str(Path(sys.executable))
    formal = root / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2"
    prep, step1_root, step2_root = formal / "prepare", formal / "step1", formal / "step2"

    def prepare() -> None:
        _run(
            [
                py,
                "-u",
                str(root / "scripts/prepare_v42_formal_f2.py"),
                "--project-root",
                str(root),
                "--seed",
                str(args.split_seed),
            ],
            root,
        )
        _run(
            [
                py,
                "-u",
                str(root / "scripts/build_v42_formal_step1_pool_f2.py"),
                "--project-root",
                str(root),
                "--source-rows",
                str(prep / "FORMAL_F2_SOURCE_ROWS.parquet"),
                "--ledger",
                str(prep / "FORMAL_F2_EVENT_LEDGER.csv"),
                "--output-manifest",
                str(prep / "FORMAL_F2_STEP1_WINDOW_MANIFEST.parquet"),
                "--split-seed",
                str(args.split_seed),
                "--min-target-train-groups",
                "65",
            ],
            root,
        )
        _run(
            [
                py,
                "-u",
                str(root / "scripts/materialize_v42_formal_step2_f2.py"),
                "--project-root",
                str(root),
                "--metadata-pool",
                str(prep / "FORMAL_F2_STEP2_METADATA_POOL.parquet"),
                "--output-manifest",
                str(step2_root / "FORMAL_F2_STEP2_RAW_MANIFEST.parquet"),
                "--min-rainfall-groups",
                "69",
            ],
            root,
        )
        _run(
            [
                py,
                "-u",
                str(root / "scripts/build_v42_formal_eval_plan_f2.py"),
                "--project-root",
                str(root),
                "--ledger",
                str(prep / "FORMAL_F2_EVENT_LEDGER.csv"),
                "--output-dir",
                str(formal / "evaluation_plan"),
                "--seed",
                str(args.split_seed),
            ],
            root,
        )
        _run(
            [
                py,
                "-u",
                str(root / "scripts/write_v42_formal_f2_r0_adapter.py"),
                "--project-root",
                str(root),
                "--formal-root",
                str(formal),
                "--min-train-groups",
                "65",
            ],
            root,
        )

    def step1() -> None:
        for seed in args.seeds:
            runtime_status = None
            if args.step1_runtime_status_dir is not None:
                runtime_status = args.step1_runtime_status_dir / f"step1_seed_{seed}.json"
            step1_command = [
                py,
                "-u",
                str(root / "scripts/train_v42_step1_formal_f2.py"),
                "--project-root",
                str(root),
                "--manifest",
                str(prep / "FORMAL_F2_STEP1_WINDOW_MANIFEST.parquet"),
                "--output-dir",
                str(step1_root / f"seed_{seed}"),
                "--model-seed",
                str(seed),
                "--split-seed",
                str(args.split_seed),
                "--sensor-layout-seed",
                str(args.sensor_layout_seed),
                "--min-train-groups",
                "65",
                "--num-workers",
                str(args.step1_num_workers),
                "--prefetch-factor",
                str(args.step1_prefetch_factor),
                "--pin-memory" if args.step1_pin_memory else "--no-pin-memory",
                "--persistent-workers" if args.step1_persistent_workers else "--no-persistent-workers",
            ]
            if args.step1_cache_dir is not None:
                step1_command.extend(["--cache-dir", str(args.step1_cache_dir)])
            if runtime_status is not None:
                step1_command.extend(["--runtime-status-file", str(runtime_status)])
            _run(
                step1_command,
                root,
            )

    def step2() -> None:
        primary = step1_root / f"seed_{args.primary_step1_seed}"
        if not (primary / "best_model.pt").exists():
            raise FileNotFoundError(primary / "best_model.pt")
        gat_manifest = step2_root / "FORMAL_F2_STEP2_GAT_MANIFEST.parquet"
        history_source = step2_root / "FORMAL_F2_HISTORY_SOURCE_MANIFEST.parquet"
        _run(
            [
                py,
                "-u",
                str(root / "scripts/build_v42_formal_gat_history_source_f2.py"),
                "--project-root",
                str(root),
                "--raw-manifest",
                str(step2_root / "FORMAL_F2_STEP2_RAW_MANIFEST.parquet"),
                "--step1-window-manifest",
                str(prep / "FORMAL_F2_STEP1_WINDOW_MANIFEST.parquet"),
                "--output-manifest",
                str(history_source),
                "--min-rainfall-groups",
                "69",
            ],
            root,
        )
        _run(
            [
                py,
                "-u",
                str(root / "scripts/materialize_v42_formal_gat_history_f2.py"),
                "--project-root",
                str(root),
                "--input-manifest",
                str(step2_root / "FORMAL_F2_STEP2_RAW_MANIFEST.parquet"),
                "--step1-window-manifest",
                str(prep / "FORMAL_F2_STEP1_WINDOW_MANIFEST.parquet"),
                "--history-source-manifest",
                str(history_source),
                "--step1-model-dir",
                str(primary),
                "--output-manifest",
                str(gat_manifest),
                "--min-rainfall-groups",
                "69",
                "--sensor-layout-seed",
                str(args.sensor_layout_seed),
            ],
            root,
        )
        target_name = (
            "FORMAL_F2_STEP2_CONTROL_CORE_MANIFEST.parquet"
            if args.step2_target_contract == "CONTROL_CORE"
            else "FORMAL_F2_STEP2_FULL_HYDRAULIC_MANIFEST.parquet"
        )
        target_manifest = step2_root / target_name
        _run(
            [
                py,
                "-u",
                str(root / "scripts/materialize_v42_step2_target_contract.py"),
                "--project-root",
                str(root),
                "--input-manifest",
                str(gat_manifest),
                "--output-manifest",
                str(target_manifest),
                "--target-contract",
                args.step2_target_contract,
                "--min-rainfall-groups",
                "69",
            ],
            root,
        )
        for seed in args.seeds:
            runtime_status = None
            if args.step2_runtime_status_dir is not None:
                runtime_status = args.step2_runtime_status_dir / f"step2_seed_{seed}.json"
                runtime_status.parent.mkdir(parents=True, exist_ok=True)
            command = [
                py,
                "-u",
                str(root / "scripts/train_v42_step2_formal_f2.py"),
                "--project-root",
                str(root),
                "--manifest",
                str(target_manifest),
                "--output-dir",
                str(step2_root / "models" / f"seed_{seed}"),
                "--seed",
                str(seed),
                "--split-seed",
                str(args.split_seed),
                "--min-train-groups",
                "65",
                "--target-contract",
                args.step2_target_contract,
            ]
            if runtime_status is not None:
                command.extend(["--runtime-status-file", str(runtime_status)])
            _run(
                command,
                root,
            )

    def calibration() -> None:
        _run(
            [
                py,
                "-u",
                str(root / "scripts/calibrate_v42_formal_step1_f2.py"),
                "--project-root",
                str(root),
                "--model-dir",
                str(step1_root / f"seed_{args.primary_step1_seed}"),
                "--sensor-layout-seed",
                str(args.sensor_layout_seed),
            ],
            root,
        )
        _run(
            [
                py,
                "-u",
                str(root / "scripts/calibrate_v42_formal_step2_safety_f2.py"),
                "--project-root",
                str(root),
                "--models-root",
                str(step2_root / "models"),
                "--seeds",
                *[str(x) for x in args.seeds],
            ],
            root,
        )

    def evidence() -> None:
        _run(
            [
                py,
                "-u",
                str(root / "scripts/compile_v42_formal_training_evidence_f2.py"),
                "--project-root",
                str(root),
                "--formal-root",
                str(formal),
                "--seeds",
                *[str(x) for x in args.seeds],
                "--primary-seed",
                str(args.primary_step1_seed),
            ],
            root,
        )

    if args.stage == "prepare":
        prepare()
    elif args.stage == "step1":
        step1()
    elif args.stage == "step2":
        step2()
    elif args.stage == "calibration":
        calibration()
    elif args.stage == "evidence":
        evidence()
    elif args.stage == "all":
        prepare()
        step1()
        step2()
        calibration()
        evidence()
    payload = _status(root)
    print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    if args.stage == "audit" and not payload["structural_training_chain_pass"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
