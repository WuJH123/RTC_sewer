"""Generate Step1 R0 input audit report (Section 2).

Reads the R0 manifests and produces:
- step1_r0_input_audit.json
- step1_r0_input_audit.md
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

R0 = ROOT / "outputs/project6_dual_reference_v4/final_v4/v42_paper/data_reuse"
S1 = ROOT / "outputs/project6_dual_reference_v4/final_v4/v42_paper/step1_gat"


def main() -> int:
    # Load R0 manifests
    pool = pd.read_parquet(R0 / "reusable_pool_manifest.parquet")
    cases = pd.read_parquet(R0 / "reusable_case_manifest.parquet")
    split = pd.read_parquet(R0 / "split_group_manifest.parquet")

    # Load audit
    audit = json.loads((R0 / "data_reuse_audit.json").read_text())

    # Compute statistics
    logical_rows = audit["logical_detail_records"]
    unique_physical = len(cases["physical_identity_sha256"].unique()) if "physical_identity_sha256" in cases.columns else audit["unique_physical_runs"]

    # Step1 dynamics eligible
    eligible_cols = [c for c in pool.columns if "eligible" in c.lower() or "dynamics" in c.lower()]
    if "eligible_dynamics_pretrain" in pool.columns:
        eligible_mask = pool["eligible_dynamics_pretrain"].fillna(False).astype(bool)
    else:
        eligible_mask = pd.Series([True] * len(pool))
    eligible_runs = pool[eligible_mask]
    n_eligible = len(eligible_runs)

    # Unique detail files
    n_detail_files = eligible_runs["detail_path"].nunique() if "detail_path" in eligible_runs.columns else 0

    # Unique rainfall groups
    if "rainfall_group_key" in eligible_runs.columns:
        n_rainfall_groups = eligible_runs["rainfall_group_key"].nunique()
    elif "event_id" in eligible_runs.columns:
        n_rainfall_groups = eligible_runs["event_id"].nunique()
    else:
        n_rainfall_groups = 0

    # Reserved evaluation
    if "source_role" in pool.columns:
        n_reserved = (pool["source_role"].astype(str) == "reserved_evaluation").sum()
    else:
        n_reserved = 0

    # Domain distribution
    if "domain_id" in eligible_runs.columns:
        domain_dist = eligible_runs["domain_id"].value_counts().head(20).to_dict()
    else:
        domain_dist = {}

    # Source experiment distribution
    if "source_experiment" in eligible_runs.columns:
        source_dist = eligible_runs["source_experiment"].value_counts().head(20).to_dict()
    else:
        source_dist = {}

    report = {
        "contract": "PROJECT6_V42_STEP1_R0_INPUT_AUDIT",
        "logical_detail_rows": int(logical_rows),
        "unique_physical_identities": int(unique_physical),
        "step1_dynamics_eligible_runs": int(n_eligible),
        "unique_detail_files": int(n_detail_files),
        "unique_rainfall_groups": int(n_rainfall_groups),
        "reserved_evaluation_count": int(n_reserved),
        "domain_distribution_top20": {str(k): int(v) for k, v in domain_dist.items()},
        "source_experiment_distribution_top20": {str(k): int(v) for k, v in source_dist.items()},
        "r0_audit_fields": {
            "full_finite_check": audit["full_finite_check"],
            "missing_targets_are_imputed": audit["missing_targets_are_imputed"],
            "strict_semantics_wrapper": audit["strict_semantics_wrapper"],
            "discovery_cache_current": audit["discovery_cache_current"],
        },
    }

    S1.mkdir(parents=True, exist_ok=True)
    (S1 / "step1_r0_input_audit.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    # Markdown report
    md = f"""# Step1 R0 Input Audit

## Summary

| Metric | Value |
|--------|-------|
| Logical detail rows | {logical_rows:,} |
| Unique physical identities | {unique_physical:,} |
| Step1 dynamics eligible runs | {n_eligible:,} |
| Unique detail files | {n_detail_files:,} |
| Unique rainfall groups | {n_rainfall_groups:,} |
| Reserved evaluation count | {n_reserved:,} |

## R0 Audit Fields

- full_finite_check: {audit['full_finite_check']}
- missing_targets_are_imputed: {audit['missing_targets_are_imputed']}
- strict_semantics_wrapper: {audit['strict_semantics_wrapper']}
- discovery_cache_current: {audit['discovery_cache_current']}

## Domain Distribution (top 20)

| Domain | Count |
|--------|-------|
"""
    for k, v in domain_dist.items():
        md += f"| {k} | {v:,} |\n"

    md += f"""
## Source Experiment Distribution (top 20)

| Experiment | Count |
|------------|-------|
"""
    for k, v in source_dist.items():
        md += f"| {k} | {v:,} |\n"

    (S1 / "step1_r0_input_audit.md").write_text(md, encoding="utf-8")
    print(f"Wrote {S1 / 'step1_r0_input_audit.json'}")
    print(f"Wrote {S1 / 'step1_r0_input_audit.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
