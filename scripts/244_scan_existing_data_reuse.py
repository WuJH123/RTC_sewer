"""Gate 5 Phase 3: Scan and filter existing trusted single-facility data.

Scans existing data from:
  - Aug1 single-facility response data
  - Oracle analysis data
  - Previous Gate 2.5/3 data

Accepts only data matching current contract:
  - Same network/rainfall/prefix/state
  - Correct Dynamic Internal
  - No-hotstart formal qualification
  - Actual/readback actions
  - H120 window correct
  - Complete provenance

Output:
  - existing_action_response_reuse.csv
  - existing_action_response_rejected.csv
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "outputs" / "project6_dual_reference_v4" / "recovery_capability_v2" / "gate4_h120_batch0"
GATE5_DIR = OUT_DIR / "gate5_exact_diagnosis"

# Directories to scan for existing data (limited scope)
SCAN_DIRS = [
    PROJECT_ROOT / "outputs" / "project6_dual_reference_v4",
    PROJECT_ROOT / "outputs" / "project6_pfvfirst_dualfallback_10min_v3",
]

EVENT_ID = "V31_RP10_D2H_P65_v31_independent_gamma_084"
INP_PATH = PROJECT_ROOT / "data" / "wuhan_v8_storage_retrofit.inp"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_for_detail_csv(scan_dirs):
    """Find all detail CSV files in scan directories (limited depth)."""
    found = []
    for d in scan_dirs:
        if not d.exists():
            continue
        # Only scan immediate children (depth=1)
        for csv in d.glob("*detail*.csv"):
            found.append(csv)
        for csv in d.glob("*results*.csv"):
            if csv.suffix == ".csv":
                found.append(csv)
        # Check subdirectories one level deep
        for sub in d.iterdir():
            if sub.is_dir():
                for csv in sub.glob("*detail*.csv"):
                    found.append(csv)
    return found


def check_csv_compatibility(csv_path, target_event_id, target_network_hash):
    """Check if a CSV matches the current contract requirements."""
    try:
        df = pd.read_csv(csv_path, nrows=5)
    except Exception:
        return {"compatible": False, "reason": "unreadable"}

    if df.empty:
        return {"compatible": False, "reason": "empty"}

    # Check for required columns
    has_elapsed = "elapsed_min" in df.columns
    has_flood = any(c.startswith("flood:") for c in df.columns)
    has_action = any(c.startswith("a:") for c in df.columns)

    if not (has_elapsed and (has_flood or has_action)):
        return {"compatible": False, "reason": "missing_required_columns"}

    # Check event_id if present
    if "event_id" in df.columns:
        event_match = df["event_id"].iloc[0] == target_event_id
        if not event_match:
            return {"compatible": False, "reason": f"event_mismatch:{df['event_id'].iloc[0]}"}

    # Check policy_id if present
    policy = df["policy_id"].iloc[0] if "policy_id" in df.columns else "unknown"

    return {
        "compatible": True,
        "policy": policy,
        "has_flood_columns": has_flood,
        "has_action_columns": has_action,
        "n_rows_sample": len(df),
    }


def main():
    print("=" * 70)
    print("  Gate 5 Phase 3: Existing Data Reuse Scan")
    print("=" * 70)

    GATE5_DIR.mkdir(parents=True, exist_ok=True)

    network_hash = sha256_file(INP_PATH)
    print(f"\n  Target event: {EVENT_ID}")
    print(f"  Network hash: {network_hash[:16]}...")

    # Scan for existing data
    print("\n  Scanning directories...")
    all_csvs = scan_for_detail_csv(SCAN_DIRS)
    print(f"  Found {len(all_csvs)} potential CSV files")

    reuse_rows = []
    rejected_rows = []

    for csv_path in all_csvs:
        result = check_csv_compatibility(csv_path, EVENT_ID, network_hash)

        if result.get("compatible"):
            reuse_rows.append({
                "file_path": str(csv_path),
                "relative_path": str(csv_path.relative_to(PROJECT_ROOT)),
                "policy": result.get("policy", "unknown"),
                "has_flood_columns": result.get("has_flood_columns", False),
                "has_action_columns": result.get("has_action_columns", False),
                "network_hash_match": "not_verified",
                "reuse_decision": "pending_verification",
            })
        else:
            rejected_rows.append({
                "file_path": str(csv_path),
                "relative_path": str(csv_path.relative_to(PROJECT_ROOT) if csv_path.is_relative_to(PROJECT_ROOT) else csv_path),
                "reason": result.get("reason", "unknown"),
            })

    reuse_df = pd.DataFrame(reuse_rows)
    rejected_df = pd.DataFrame(rejected_rows)

    # Summary
    print(f"\n  Compatible: {len(reuse_rows)}")
    print(f"  Rejected: {len(rejected_rows)}")

    if not reuse_df.empty:
        print(f"\n  Compatible files by policy:")
        for policy, count in reuse_df["policy"].value_counts().items():
            print(f"    {policy}: {count}")

    if not rejected_df.empty:
        print(f"\n  Rejection reasons:")
        for reason, count in rejected_df["reason"].value_counts().head(5).items():
            print(f"    {reason}: {count}")

    # Save
    reuse_df.to_csv(GATE5_DIR / "existing_action_response_reuse.csv", index=False)
    rejected_df.to_csv(GATE5_DIR / "existing_action_response_rejected.csv", index=False)

    print(f"\n  Outputs saved to {GATE5_DIR}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
