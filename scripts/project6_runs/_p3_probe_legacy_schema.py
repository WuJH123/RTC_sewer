# Read-only probe: exact schemas + identity anchors for the P3 legacy audit.
from pathlib import Path

import pandas as pd

ROOT = Path(r"E:\RTC_sewer\Project6")
V4 = ROOT / "outputs" / "project6_dual_reference_v4"
FINAL = V4 / "final_v4"

SOURCES = {
    "oracle_pareto_20ev": V4 / "oracle_pareto_20ev" / "oracle_case_results.csv",
    "oracle_pareto_smoke1_fix": V4
    / "oracle_pareto_smoke1_fix"
    / "oracle_case_results.csv",
    "constraint_ablation": V4
    / "oracle_bottleneck_diagnosis"
    / "constraint_ablation_results.csv",
    "gate0_proof": V4
    / "oracle_bottleneck_diagnosis"
    / "gate0_feasible_candidate_proof.csv",
    "v4_dataset_manifest": V4
    / "action_effect_dataset_v4"
    / "v4_dataset_manifest.csv",
}

for name, path in SOURCES.items():
    if not path.exists():
        print(f"[{name}] MISSING: {path}")
        continue
    df = pd.read_csv(path, nrows=5)
    total = sum(1 for _ in open(path, encoding="utf-8")) - 1
    print(f"[{name}] rows={total} cols={len(df.columns)}")
    print("  columns:", list(df.columns))

# Identity anchors on the current frozen chain
inv = FINAL / "inventory" / "event_inventory.csv"
if inv.exists():
    df = pd.read_csv(inv, nrows=3)
    print("[event_inventory] columns:", list(df.columns))
else:
    print("[event_inventory] MISSING:", inv)

samples = FINAL / "pilot" / "dataset" / "pilot_sample_manifest.csv"
df = pd.read_csv(samples, nrows=2)
ident_cols = [
    c
    for c in df.columns
    if "sha" in c.lower() or c in ("event_id", "checkpoint_id", "split")
]
print("[pilot_sample_manifest] identity cols:", ident_cols)

plan = FINAL / "pilot" / "planning" / "pilot_candidate_plan.csv"
df = pd.read_csv(plan, nrows=2)
ident_cols = [c for c in df.columns if "sha" in c.lower()]
print("[pilot_candidate_plan] sha cols:", ident_cols)
print(
    "[pilot_candidate_plan] sample network/rainfall sha:",
    df.iloc[0].get("network_sha256", "?"),
    df.iloc[0].get("rainfall_sha256", "?"),
)

# Legacy value distributions needed for dimension checks
res = pd.read_csv(SOURCES["oracle_pareto_20ev"])
print("[20ev] constraint_mode:", sorted(res["constraint_mode"].astype(str).unique()))
print("[20ev] inp_sha256 nunique:", res["inp_sha256"].astype(str).nunique())
print(
    "[20ev] rainfall_sha256 nunique:",
    res["rainfall_sha256"].astype(str).nunique() if "rainfall_sha256" in res else "n/a",
)
print("[20ev] has checkpoint col:", [c for c in res.columns if "checkpoint" in c])

v4m = pd.read_csv(SOURCES["v4_dataset_manifest"], nrows=200)
sha_cols = [c for c in v4m.columns if "sha" in c.lower()]
print("[v4ds] sha cols:", sha_cols)
for c in ("hotstart_used_for_label", "same_state_method", "v4_label_contract", "k_value"):
    if c in v4m.columns:
        print(f"[v4ds] {c}:", sorted(v4m[c].astype(str).unique())[:5])
