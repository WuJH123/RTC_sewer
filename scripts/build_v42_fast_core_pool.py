"""Build a bounded fast-E2E pool from existing reusable manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(r"E:\RTC_sewer\Project6")
DATA = ROOT / "outputs/project6_dual_reference_v4/final_v4/v42_paper/data_reuse"
OUT = ROOT / "outputs/project6_dual_reference_v4/final_v4/v42_paper/fast_e2e_64plus/core_pool"
ROLES = ("candidate", "no_control", "dynamic_internal", "hold_previous")

def yes(v: object) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes"}

def json_ids(v: object) -> list[str]:
    try:
        return [str(x) for x in json.loads(str(v))]
    except Exception:
        return []

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--output-dir", type=Path, default=OUT)
    args = ap.parse_args()
    cases = pd.read_parquet(args.data_dir / "reusable_case_manifest.parquet")
    physical = pd.read_parquet(args.data_dir / "reusable_pool_manifest.parquet").set_index("physical_identity_sha256")
    rows, tiers, reserved = [], {"target_strict": 0, "source_strict": 0, "fast_core_compatible": 0}, 0
    for _, case in cases.iterrows():
        if str(case.get("source_role", "")) == "reserved_evaluation":
            reserved += 1
            continue
        ids = json_ids(case.get("branch_physical_ids", "[]"))
        role_rows = {role: physical.loc[[i for i in ids if i in physical.index]] for role in ROLES}
        if any(frame.empty for frame in role_rows.values()):
            continue
        forcing = any(yes(case.get(c, False)) for c in ("same_forcing_pass_y", "same_forcing_pass", "same_forcing_pass_x"))
        core = all(yes(case.get(c, False)) for c in ("four_reference_complete", "same_state_numeric_pass", "four_reference_finite_pass", "core_trajectory_targets")) and forcing
        if not core:
            continue
        tier = "target_strict" if yes(case.get("eligible_counterfactual_flood")) else "source_strict" if yes(case.get("eligible_source_domain_counterfactual_aux")) else "fast_core_compatible"
        tiers[tier] += 1
        candidates = role_rows["candidate"].drop_duplicates("action_readback_sha256")
        state = hashlib.sha256("|".join(str(case.get(c, "")) for c in ("rainfall_sha256", "checkpoint_min", "network_sha256", "case_id", "event_id")).encode()).hexdigest()
        for _, cand in candidates.iterrows():
            rows.append({"rainfall_group_key": str(case.get("rainfall_sha256", "")), "counterfactual_state_key": state, "checkpoint_min": case.get("checkpoint_min"), "candidate_trajectory_key": str(cand.name), "no_control_trajectory_key": str(role_rows["no_control"].index[0]), "dynamic_internal_trajectory_key": str(role_rows["dynamic_internal"].index[0]), "hold_previous_trajectory_key": str(role_rows["hold_previous"].index[0]), "candidate_action_signature": str(cand.get("action_readback_sha256", "")), "fast_e2e_admission_tier": tier, "domain_id": str(case.get("domain_id", "")), "source_role": str(case.get("source_role", "")), "development_only": True})
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        manifest = pd.DataFrame(columns=["rainfall_group_key", "counterfactual_state_key", "checkpoint_min", "candidate_action_signature"])
    manifest["checkpoint_min"] = pd.to_numeric(manifest["checkpoint_min"], errors="coerce")
    late = manifest[manifest.checkpoint_min.ge(120)]
    counts = late.groupby(["rainfall_group_key", "counterfactual_state_key"]).candidate_action_signature.nunique() if not late.empty else pd.Series(dtype=int)
    usable = counts[counts.ge(3)]
    groups = sorted(usable.index.get_level_values(0).unique()) if len(usable) else []
    split = pd.DataFrame({"rainfall_group_key": groups, "split": ["validation" if i % 5 == 0 else "train" for i in range(len(groups))]})
    audit = {"development_only": True, "total_cases": len(manifest), "unique_rainfall_groups": int(manifest.rainfall_group_key.nunique()), "checkpoint_ge120_groups": int(late.rainfall_group_key.nunique()), "states": int(late.counterfactual_state_key.nunique()), "states_with_ge3_candidates": len(usable), "usable_groups": len(groups), "candidate_count_distribution": counts.value_counts().sort_index().to_dict(), "source_tier_counts": tiers, "reserved_excluded": reserved, "train_rainfall_groups": int((split.split == "train").sum()), "validation_rainfall_groups": int((split.split == "validation").sum())}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(args.output_dir / "FAST_CORE_CASE_MANIFEST.parquet", index=False)
    split.to_csv(args.output_dir / "FAST_CORE_RAINFALL_GROUPS.csv", index=False)
    (args.output_dir / "FAST_CORE_POOL_AUDIT.json").write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
