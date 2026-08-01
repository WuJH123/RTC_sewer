"""Fail-closed audit for the development-only V4.2 fast E2E potential test."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.v4.v42_fast_e2e import FAST_E2E_CONTRACT_ID


REQUIRED_STRATEGIES = {
    "proposed_gat_surrogate_pfvfirst",
    "efd",
    "auto_rbc",
    "all_close",
    "no_control",
    "internal_rule",
    "hold_previous",
}


def _read(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--output-root", type=Path, default=None)
    ap.add_argument("--min-step1-groups", type=int, default=64)
    ap.add_argument("--min-step2-train-groups", type=int, default=65)
    ap.add_argument("--min-integrated-groups", type=int, default=64)
    args = ap.parse_args()

    output_root = args.output_root or (
        args.project_root / "outputs/project6_dual_reference_v4/final_v4"
    )
    fast = output_root / "v42_paper/fast_e2e_64plus"
    prep = _read(fast / "FAST_E2E_PREPARE.json")
    gat = _read(fast / "step2_fast_e2e_gat_history_audit.json")
    step2 = _read(fast / "step2_model/fast_step2_report.json")
    replay = _read(fast / "policy_replay/fast_policy_replay_summary.json")
    baseline = _read(fast / "baseline_comparison/FAST_E2E_BASELINE_COMPARISON.json")
    verdict = _read(fast / "FAST_E2E_VERDICT.json")

    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    checks["development_only"] = all(
        p.get("development_only") is True for p in (prep, gat, step2, replay, baseline, verdict)
    )
    checks["formal_not_authorized"] = all(
        p.get("formal_mainline_authorized") is not True
        for p in (prep, gat, step2, baseline, verdict)
    )

    step1_groups = int(prep.get("step1_selected_aux_groups", 0))
    selected_groups = int(prep.get("step2_selected_groups", 0))
    integrated_groups = int(gat.get("output_rainfall_groups", 0))
    train_groups = [str(x) for x in step2.get("train_rainfall_groups", [])]
    validation_groups = [str(x) for x in step2.get("validation_rainfall_groups", [])]
    checks["step1_rainfall_groups_ge_min"] = step1_groups >= int(args.min_step1_groups)
    checks["step2_selected_groups_ge_64"] = selected_groups >= 64
    checks["step2_train_groups_gt_64"] = len(set(train_groups)) >= int(args.min_step2_train_groups)
    checks["integrated_gat_groups_ge_min"] = integrated_groups >= int(args.min_integrated_groups)
    checks["step2_train_validation_disjoint"] = not (set(train_groups) & set(validation_groups))

    checks["gat_state_source"] = gat.get("state_source") == "gat_sparse_reconstruction"
    checks["no_current_frame_repetition"] = gat.get("current_frame_repetition_used") is False
    checks["no_swmm_truth_history_online"] = (
        gat.get("authoritative_swmm_history_used_as_online_input") is False
    )
    checks["no_realized_future_rain_online"] = gat.get("realized_future_rainfall_used_online") is False
    checks["causal_rain_authority"] = str(gat.get("rainfall_input_authority", "")).startswith("causal_")

    checks["step2_trajectory_first"] = step2.get("trajectory_first") is True
    checks["step2_outfall_claim_disabled"] = step2.get("outfall_supervised") is False
    checks["policy_replayed_states_exist"] = int(replay.get("state_groups_replayed", 0)) > 0
    false_safe = replay.get("false_safe_rate")
    checks["false_safe_rate_reported"] = false_safe is not None

    strategies = set((baseline.get("strategy_summary") or {}).keys())
    checks["all_required_baselines_present"] = REQUIRED_STRATEGIES <= strategies
    checks["baseline_contract_flag"] = baseline.get("all_required_strategies_present") is True
    proposed = (baseline.get("strategy_summary") or {}).get(
        "proposed_gat_surrogate_pfvfirst", {}
    )
    checks["proposed_is_swmm_backed"] = float(proposed.get("swmm_backed_fraction", 0.0)) == 1.0

    details.update(
        {
            "step1_selected_groups": step1_groups,
            "step2_selected_groups": selected_groups,
            "step2_train_groups": len(set(train_groups)),
            "step2_validation_groups": len(set(validation_groups)),
            "integrated_gat_groups": integrated_groups,
            "replayed_states": int(replay.get("state_groups_replayed", 0)),
            "false_safe_rate": false_safe,
            "fallback_rate": replay.get("fallback_rate"),
            "baseline_strategies": sorted(strategies),
            "baseline_potential_go": baseline.get("potential_go"),
            "final_potential_go": verdict.get("potential_go"),
        }
    )

    failed = [name for name, passed in checks.items() if not passed]
    report = {
        "contract_id": FAST_E2E_CONTRACT_ID,
        "stage": "fast_e2e_final_audit",
        "development_only": True,
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
        "details": details,
        "interpretation": (
            "PASS means the bounded development execution line is internally coherent and trained "
            "Step2 on >64 independent rainfall groups. It does not mean formal paper evidence PASS."
        ),
    }
    (fast / "FAST_E2E_AUDIT.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    lines = [
        "# V4.2 Fast E2E Final Audit",
        "",
        f"Overall: {'PASS' if not failed else 'FAIL'}",
        "",
        "## Counts",
        "",
    ]
    for key, value in details.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Checks", ""])
    for key, value in checks.items():
        lines.append(f"- {'PASS' if value else 'FAIL'} — {key}")
    if failed:
        lines.extend(["", "## Failed checks", ""])
        lines.extend(f"- {x}" for x in failed)
    (fast / "FAST_E2E_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
