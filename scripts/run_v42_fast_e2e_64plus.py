"""Canonical development-only V4.2 64+ rainfall end-to-end potential test.

This runner does NOT weaken the formal paper mainline. It exists to answer a
cheaper question first: does the actual sparse-state -> hydraulic surrogate ->
PFV-first control chain show enough drainage-control potential to justify formal
multi-seed training and authoritative blind experiments?

Execution
---------
1. select >=64 (default 96) diverse Step1 auxiliary rainfall fingerprints;
2. select >=64 strict four-reference Step2 groups, preferring Train1600-like
   evidence, with >=3 candidate cases at one state per rainfall group;
3. require checkpoint >=120 min so the integrated GAT history is truly causal;
4. train/reuse TemporalSparseGATReconstructorV42;
5. create Step2 history from 13 actual Step1 reconstructions at 5-min spacing;
6. replace realised future rainfall input with a causal persistence/decay forecast;
7. train MultiReferenceHydraulicSurrogate on the GAT-history manifest;
8. replay PFV-first candidate selection, scoring selected recorded candidates
   with authoritative SWMM outcomes;
9. compare Proposed vs No-control, Internal, Hold, EFD, Auto-RBC and All-close.

All artifacts are development-only and cannot authorize formal evidence.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.v4.v42_fast_e2e import (
    DEFAULT_CANDIDATES_PER_STATE,
    DEFAULT_TARGET_RAINFALL_GROUPS,
    FAST_E2E_CONTRACT_ID,
    MIN_RAINFALL_GROUPS,
    PREFERRED_SOURCE_TOKENS,
    build_fast_step1_aux_allowlist_64plus,
)
from sewerrtc.v4.v42_fast_e2e_warm import build_warm_fast_step2_dataset_64plus


def _run(cmd: list[str]) -> None:
    print("\nRUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return payload


def _assert_file(path: Path, what: str) -> None:
    if not path.exists() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"{what} missing or empty: {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--output-root", type=Path, default=None)
    ap.add_argument("--target-rainfall-groups", type=int, default=DEFAULT_TARGET_RAINFALL_GROUPS)
    ap.add_argument("--min-rainfall-groups", type=int, default=MIN_RAINFALL_GROUPS)
    ap.add_argument("--candidates-per-state", type=int, default=DEFAULT_CANDIDATES_PER_STATE)
    ap.add_argument("--preferred-source-token", action="append", default=None)
    ap.add_argument("--min-checkpoint-min", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sensor-ratio", type=float, default=0.10)
    ap.add_argument("--sensor-layout-seed", type=int, default=42)
    ap.add_argument("--step1-epochs", type=int, default=6)
    ap.add_argument("--step2-epochs", type=int, default=6)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument(
        "--reuse-step1-model-dir",
        type=Path,
        default=None,
        help="Optional compatible TemporalSparseGAT directory containing best_model.pt.",
    )
    args = ap.parse_args()

    if args.target_rainfall_groups < args.min_rainfall_groups:
        raise ValueError("target-rainfall-groups must be >= min-rainfall-groups")
    if args.min_rainfall_groups < 64:
        raise ValueError("this execution line intentionally requires at least 64 rainfall groups")
    if args.candidates_per_state < 2:
        raise ValueError("policy replay requires at least two candidates per state")

    root = args.project_root
    output_root = args.output_root or (root / "outputs/project6_dual_reference_v4/final_v4")
    r0 = output_root / "v42_paper/data_reuse"
    s1_manifest = output_root / "v42_paper/step1_gat/dataset/step1_window_manifest.parquet"
    fast = output_root / "v42_paper/fast_e2e_64plus"
    fast.mkdir(parents=True, exist_ok=True)

    for path, what in (
        (r0 / "reusable_pool_manifest.parquet", "strict R0 physical manifest"),
        (r0 / "reusable_case_manifest.parquet", "strict R0 case manifest"),
        (r0 / "split_group_manifest.parquet", "R0 rainfall split manifest"),
        (s1_manifest, "Step1 temporal window manifest"),
    ):
        _assert_file(path, what)

    preferred_tokens = tuple(args.preferred_source_token or PREFERRED_SOURCE_TOKENS)
    aux_allow = fast / "step1_aux_allowlist_64plus.json"
    aux = build_fast_step1_aux_allowlist_64plus(
        manifest_path=s1_manifest,
        output_path=aux_allow,
        target_groups=args.target_rainfall_groups,
        min_groups=args.min_rainfall_groups,
        seed=args.seed,
    )

    selection, step2_raw = build_warm_fast_step2_dataset_64plus(
        project_root=root,
        physical_manifest=r0 / "reusable_pool_manifest.parquet",
        case_manifest=r0 / "reusable_case_manifest.parquet",
        split_manifest=r0 / "split_group_manifest.parquet",
        working_dir=fast,
        target_groups=args.target_rainfall_groups,
        min_groups=args.min_rainfall_groups,
        candidates_per_state=args.candidates_per_state,
        preferred_source_tokens=preferred_tokens,
        seed=args.seed,
        min_checkpoint_min=args.min_checkpoint_min,
    )

    prep = {
        "contract_id": FAST_E2E_CONTRACT_ID,
        "stage": "prepare",
        "development_only": True,
        "formal_mainline_authorized": False,
        "data_policy": "enough_diverse_strict_development_data_not_exhaustive_historical_recovery",
        "target_rainfall_groups": int(args.target_rainfall_groups),
        "minimum_rainfall_groups": int(args.min_rainfall_groups),
        "minimum_checkpoint_min": float(args.min_checkpoint_min),
        "step1_selected_aux_groups": int(aux["selected_aux_groups"]),
        "step2_selected_groups": int(selection.selected_rainfall_groups),
        "step2_selected_states": int(selection.selected_states),
        "step2_selected_cases": int(selection.selected_cases),
        "step2_preferred_train1600_like_groups": int(selection.preferred_rainfall_groups),
        "step2_used_nonpreferred_fill": bool(selection.used_nonpreferred_fill),
        "step2_materialized_groups": int(step2_raw.rainfall_groups),
        "preferred_source_tokens": list(preferred_tokens),
        "next": "train_or_reuse_step1_then_causal_gat_history",
    }
    (fast / "FAST_E2E_PREPARE.json").write_text(
        json.dumps(prep, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(prep, indent=2, allow_nan=False), flush=True)
    if args.prepare_only:
        return 0

    py = str(Path(sys.executable))
    if args.reuse_step1_model_dir is None:
        step1_model_dir = fast / "step1_model"
        _run(
            [
                py,
                str(root / "scripts/train_v42_step1_streaming.py"),
                "--project-root", str(root),
                "--manifest", str(s1_manifest),
                "--model-seed", str(args.seed),
                "--sensor-layout-seed", str(args.sensor_layout_seed),
                "--split-seed", str(args.seed),
                "--aux-sampling-seed", str(args.seed),
                "--sensor-ratio", str(args.sensor_ratio),
                "--aux-pretrain",
                "--aux-allowlist", str(aux_allow),
                "--aux-epochs", "1",
                "--aux-max-windows-per-group", "8",
                "--aux-max-windows-per-run", "2",
                "--epochs", str(args.step1_epochs),
                "--patience", "3",
                "--priority-weight", "0",
                "--wet-priority-weight", "0",
                "--nll-weight", "0",
                "--output-dir", str(step1_model_dir),
            ]
        )
    else:
        step1_model_dir = args.reuse_step1_model_dir
    _assert_file(step1_model_dir / "best_model.pt", "Step1 model checkpoint")

    step1_report_path = step1_model_dir / "formal_training_report.json"
    if step1_report_path.exists():
        s1_report = _read_json(step1_report_path)
        aux_report = s1_report.get("auxiliary_pretraining") or {}
        selection_report = aux_report.get("selection") or {}
        seen_groups = int(selection_report.get("rainfall_groups", 0))
        if seen_groups and seen_groups < args.min_rainfall_groups:
            raise RuntimeError(
                f"Step1 auxiliary pretraining actually saw only {seen_groups} rainfall groups; "
                f"minimum is {args.min_rainfall_groups}"
            )

    raw_manifest = fast / "step2_fast_e2e_core_manifest.parquet"
    gat_manifest = fast / "step2_fast_e2e_gat_manifest.parquet"
    gat_audit = fast / "step2_fast_e2e_gat_history_audit.json"
    _run(
        [
            py,
            str(root / "scripts/materialize_v42_fast_gat_history.py"),
            "--project-root", str(root),
            "--input-manifest", str(raw_manifest),
            "--step1-model-dir", str(step1_model_dir),
            "--output-manifest", str(gat_manifest),
            "--audit-output", str(gat_audit),
            "--sensor-ratio", str(args.sensor_ratio),
            "--sensor-layout-seed", str(args.sensor_layout_seed),
            "--min-rainfall-groups", str(args.min_rainfall_groups),
        ]
    )
    gat_report = _read_json(gat_audit)
    if int(gat_report.get("output_rainfall_groups", 0)) < args.min_rainfall_groups:
        raise RuntimeError("GAT-integrated history did not retain the required rainfall diversity")
    if gat_report.get("realized_future_rainfall_used_online") is not False:
        raise RuntimeError("future realised rainfall leak detected in fast E2E manifest")
    if gat_report.get("current_frame_repetition_used") is not False:
        raise RuntimeError("current-frame repetition detected in fast E2E GAT history")
    if gat_report.get("authoritative_swmm_history_used_as_online_input") is not False:
        raise RuntimeError("SWMM truth history leaked into integrated Step2 input")

    step2_model_dir = fast / "step2_model"
    _run(
        [
            py,
            str(root / "scripts/train_v42_step2_fast.py"),
            "--project-root", str(root),
            "--manifest", str(gat_manifest),
            "--output-dir", str(step2_model_dir),
            "--epochs", str(args.step2_epochs),
            "--seed", str(args.seed),
        ]
    )
    _assert_file(step2_model_dir / "best_model.pt", "Step2 model checkpoint")

    replay_dir = fast / "policy_replay"
    _run(
        [
            py,
            str(root / "scripts/evaluate_v42_fast_policy_replay.py"),
            "--project-root", str(root),
            "--manifest", str(gat_manifest),
            "--model-dir", str(step2_model_dir),
            "--output-dir", str(replay_dir),
            "--seed", str(args.seed),
        ]
    )
    replay = _read_json(replay_dir / "fast_policy_replay_summary.json")
    if int(replay.get("state_groups_replayed", 0)) <= 0:
        raise RuntimeError("PFV-first replay did not contain any state with real candidate choice")

    baseline_dir = fast / "baseline_comparison"
    _run(
        [
            py,
            str(root / "scripts/evaluate_v42_fast_baselines.py"),
            "--project-root", str(root),
            "--manifest", str(gat_manifest),
            "--model-dir", str(step2_model_dir),
            "--policy-replay-dir", str(replay_dir),
            "--output-dir", str(baseline_dir),
            "--seed", str(args.seed),
        ]
    )
    baseline = _read_json(baseline_dir / "FAST_E2E_BASELINE_COMPARISON.json")
    if baseline.get("all_required_strategies_present") is not True:
        raise RuntimeError("fast E2E baseline table is incomplete")

    final = {
        "contract_id": FAST_E2E_CONTRACT_ID,
        "stage": "fast_e2e_verdict",
        "development_only": True,
        "formal_mainline_authorized": False,
        "rainfall_diversity_gate": {
            "minimum": int(args.min_rainfall_groups),
            "step1_aux_groups": int(aux["selected_aux_groups"]),
            "step2_selected_groups": int(selection.selected_rainfall_groups),
            "gat_integrated_groups": int(gat_report["output_rainfall_groups"]),
        },
        "state_source": gat_report.get("state_source"),
        "causal_reconstructed_history": gat_report.get("reconstructed_history_contract"),
        "current_frame_repetition_used": gat_report.get("current_frame_repetition_used"),
        "authoritative_swmm_history_used_as_online_input": gat_report.get(
            "authoritative_swmm_history_used_as_online_input"
        ),
        "realized_future_rainfall_used_online": gat_report.get("realized_future_rainfall_used_online"),
        "policy_replay_go_signal": bool(replay.get("go_signal")),
        "baseline_potential_go": bool(baseline.get("potential_go")),
        "potential_go": bool(replay.get("go_signal") and baseline.get("potential_go")),
        "baseline_comparison_json": str(baseline_dir / "FAST_E2E_BASELINE_COMPARISON.json"),
        "baseline_comparison_csv": str(baseline_dir / "FAST_E2E_BASELINE_COMPARISON.csv"),
        "next_if_go": "one_authoritative_SWMM_micro_closed_loop_for_Proposed_EFD_AutoRBC_AllClose_NoControl_Internal",
        "next_if_no_go": "diagnose_step1_step2_candidate_selection_using_saved_fast_e2e_artifacts_before_more_compute",
        "warning": (
            "Development screening only. EFD/Auto-RBC/All-close rows may be surrogate-screened "
            "when no close recorded SWMM candidate exists; formal paper claims still require "
            "authoritative rolling SWMM for every strategy."
        ),
    }
    (fast / "FAST_E2E_VERDICT.json").write_text(
        json.dumps(final, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(final, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
