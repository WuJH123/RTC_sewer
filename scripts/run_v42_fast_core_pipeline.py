"""Orchestrate the bounded existing-data fast scientific potential line.

The pipeline intentionally avoids exhaustive Phase-0 recovery:
1. build a core pool from structured V4/Train1600 manifests;
2. materialise only selected existing SWMM cases;
3. run the existing causal GAT -> Step2 -> PFV-first -> baseline chain.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("\nRUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "potential"), default="potential")
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reuse-step1-model-dir", type=Path, default=None)
    args = ap.parse_args()

    root = args.project_root
    py = str(Path(sys.executable))
    v4_search_root = root / "outputs/project6_dual_reference_v4"
    final_v4 = v4_search_root / "final_v4"
    fast = final_v4 / "v42_paper/fast_e2e_64plus"
    core = fast / "core_pool"
    raw = fast / "step2_fast_e2e_core_manifest.parquet"

    if args.mode == "smoke":
        target_groups = 8
        materialize_min = 8
        runner_min = 8
        step1_epochs = 1
        step2_epochs = 1
        min_train = 0
    else:
        target_groups = 88
        # With the downstream deterministic 80/20 group split, 81 total gives
        # at least 65 train groups.  Prefer 88 but fail before training below 81.
        materialize_min = 81
        runner_min = 64
        step1_epochs = 3
        step2_epochs = 4
        min_train = 65

    run(
        [
            py,
            "-u",
            str(root / "scripts/build_v42_fast_core_pool.py"),
            "--project-root",
            str(root),
            "--v4-root",
            str(v4_search_root),
            "--min-checkpoint-min",
            "120",
            "--candidates-per-state",
            "3",
            "--seed",
            str(args.seed),
        ]
    )

    run(
        [
            py,
            "-u",
            str(root / "scripts/materialize_v42_fast_core_train1600.py"),
            "--project-root",
            str(root),
            "--output-root",
            str(v4_search_root),
            "--core-pool-dir",
            str(core),
            "--output-manifest",
            str(raw),
            "--target-groups",
            str(target_groups),
            "--min-groups",
            str(materialize_min),
            "--candidates-per-state",
            "3",
            "--seed",
            str(args.seed),
        ]
    )

    cmd = [
        py,
        "-u",
        str(root / "scripts/run_v42_fast_e2e_64plus.py"),
        "--project-root",
        str(root),
        "--target-rainfall-groups",
        str(target_groups),
        "--min-rainfall-groups",
        str(runner_min),
        "--min-step2-train-groups",
        str(min_train),
        "--candidates-per-state",
        "3",
        "--min-checkpoint-min",
        "120",
        "--step1-epochs",
        str(step1_epochs),
        "--step2-epochs",
        str(step2_epochs),
        "--seed",
        str(args.seed),
        "--prebuilt-step2-manifest",
        str(raw),
    ]
    if args.mode == "smoke":
        cmd.append("--smoke")
    if args.reuse_step1_model_dir is not None:
        cmd.extend(["--reuse-step1-model-dir", str(args.reuse_step1_model_dir)])
    run(cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
