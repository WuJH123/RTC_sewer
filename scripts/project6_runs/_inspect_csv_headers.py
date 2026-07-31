from pathlib import Path

root = Path(r"E:\RTC_sewer\Project6\outputs\project6_dual_reference_v4\final_v4")
files = [
    "peak_boundary/peak_boundary_anchor_library.csv",
    "opportunities/standard_checkpoint_catalog.csv",
    "pilot/planning/pilot_candidate_plan.csv",
    "pilot/planning/pilot_checkpoint_catalog.csv",
    "peak_boundary/sample_manifest.csv",
    "peak_boundary/branch_manifest.csv",
    "peak_boundary/dataset/sample_manifest.csv",
]
for rel in files:
    path = root / rel
    if not path.exists():
        print(f"MISSING {rel}")
        continue
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip()
    print(f"=== {rel}")
    print(header)
    print()
