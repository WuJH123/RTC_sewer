"""Orchestrate the development-only V4.2 fast feasibility screen.

The fast path intentionally does not write formal evidence.json files.  It
prepares a small diverse Step-1 auxiliary allow-list, materialises a small
four-reference Step-2 control-core dataset, trains the lightweight formal-model
architecture, and performs SWMM-backed offline PFV-first policy replay.

A positive result authorizes only the *next* micro experiment: one authoritative
rolling SWMM event with frozen pilot models.  It never authorizes the paper
mainline or Formal Blind.
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

from sewerrtc.v4.v42_fast_feasibility import (
    FAST_CONTRACT_ID,
    build_fast_step1_aux_allowlist,
    build_fast_step2_core_dataset,
)


def _run(cmd: list[str]) -> None:
    print("RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--output-root", type=Path, default=None)
    ap.add_argument("--max-aux-groups", type=int, default=64)
    ap.add_argument("--max-step2-cases", type=int, default=96)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--step2-epochs", type=int, default=6)
    args = ap.parse_args()

    root = args.project_root
    output_root = args.output_root or (root / "outputs/project6_dual_reference_v4/final_v4")
    r0 = output_root / "v42_paper/data_reuse"
    s1_manifest = output_root / "v42_paper/step1_gat/dataset/step1_window_manifest.parquet"
    fast = output_root / "v42_paper/fast_feasibility"
    fast.mkdir(parents=True, exist_ok=True)

    aux_allow = fast / "step1_fast_aux_allowlist.json"
    aux_payload = build_fast_step1_aux_allowlist(
        manifest_path=s1_manifest,
        output_path=aux_allow,
        max_groups=args.max_aux_groups,
        seed=args.seed,
    )
    step2_manifest = fast / "step2_fast_core_manifest.parquet"
    step2_audit = fast / "step2_fast_core_audit.json"
    ds = build_fast_step2_core_dataset(
        project_root=root,
        physical_manifest=r0 / "reusable_pool_manifest.parquet",
        case_manifest=r0 / "reusable_case_manifest.parquet",
        split_manifest=r0 / "split_group_manifest.parquet",
        output_manifest=step2_manifest,
        audit_output=step2_audit,
        max_cases=args.max_step2_cases,
        seed=args.seed,
    )
    prep = {
        "contract_id": FAST_CONTRACT_ID,
        "development_only": True,
        "formal_mainline_authorized": False,
        "step1_aux_allowlist": str(aux_allow),
        "step1_aux_groups": aux_payload["selected_aux_groups"],
        "step2_manifest": str(step2_manifest),
        "step2_cases": ds.accepted_cases,
        "step2_rainfall_groups": ds.rainfall_groups,
    }
    (fast / "fast_prepare_summary.json").write_text(json.dumps(prep, indent=2), encoding="utf-8")
    print(json.dumps(prep, indent=2), flush=True)
    if args.prepare_only:
        return 0

    py = str(Path(sys.executable))
    model_dir = fast / "step2_model"
    replay_dir = fast / "policy_replay"
    _run([
        py,
        str(root / "scripts/train_v42_step2_fast.py"),
        "--project-root", str(root),
        "--manifest", str(step2_manifest),
        "--output-dir", str(model_dir),
        "--epochs", str(args.step2_epochs),
        "--seed", str(args.seed),
    ])
    _run([
        py,
        str(root / "scripts/evaluate_v42_fast_policy_replay.py"),
        "--project-root", str(root),
        "--manifest", str(step2_manifest),
        "--model-dir", str(model_dir),
        "--output-dir", str(replay_dir),
        "--seed", str(args.seed),
    ])
    replay = json.loads((replay_dir / "fast_policy_replay_summary.json").read_text(encoding="utf-8"))
    final = {
        "contract_id": FAST_CONTRACT_ID,
        "development_only": True,
        "formal_mainline_authorized": False,
        "step2_policy_replay_go_signal": bool(replay.get("go_signal")),
        "integrated_step1_to_step2_closed_loop_proven": False,
        "next_required": (
            "one_event_authoritative_SWMM_GAT_integrated_rolling_closed_loop"
            if replay.get("go_signal")
            else "diagnose_step2_or_candidate_selection_before_new_SWMM"
        ),
        "warning": "This is a fast feasibility screen, not formal paper evidence.",
    }
    (fast / "FAST_FEASIBILITY_VERDICT.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
