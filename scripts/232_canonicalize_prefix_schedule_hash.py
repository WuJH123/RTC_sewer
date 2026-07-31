"""Gate 3 Planning Preflight #1: Canonical Prefix Schedule Hash Audit.

Recomputes prefix schedule SHA256 using a canonical serialization that is
independent of branch name, file path, CSV column order, metadata, requested
action, and post-checkpoint data.

Uses:
  - canonical 90-link facility order (sorted alphabetically)
  - timestamp ascending (elapsed_min)
  - actual/readback setting values (setting: columns)
  - Truth Contract frozen precision (float6 -> string with 6 decimals)
  - rows strictly before checkpoint (elapsed_min < checkpoint_elapsed_min)

Outputs:
  - canonical_prefix_hash_audit.json
  - canonical_prefix_matrices.csv (per-branch canonical matrices for inspection)
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

V3_DIR = PROJECT_ROOT / "outputs" / "project6_dual_reference_v4" / "recovery_validation" / "gate2p5_real_v3"
SCOPE_CONTRACT = PROJECT_ROOT / "docs" / "contracts" / "PROJECT6_V4_CONTROL_SCOPE_CONTRACT_V2.json"
OUT_DIR = V3_DIR / "gate3_planning"

# Checkpoint definitions
CHECKPOINTS = {
    "A_pre_peak": 225.0,
    "B_recession": 305.0,
}

BRANCHES = ["dynamic_internal", "no_control", "hold_snapshot", "hold_previous"]

# Map branch name -> detail CSV file stem
BRANCH_FILE_MAP = {
    "dynamic_internal": "v3_cp{cp}__dynamic_internal_detail",
    "no_control": "v3_cp{cp}__no_control_detail",
    "hold_snapshot": "v3_cp{cp}__hold_snapshot_detail",
    "hold_previous": "v3_cp{cp}__hold_previous_detail",
}

# Precision for Truth Contract frozen serialization
PRECISION = 6


def load_canonical_link_order() -> list[str]:
    """Load the canonical 90-link prefix order from Scope Contract V2."""
    sc = json.loads(SCOPE_CONTRACT.read_text(encoding="utf-8"))
    links = sc["shared_prefix_contract"]["prefix_links"]
    return sorted(links)  # Canonical: alphabetical


def get_setting_column(link_id: str, eng36_ids: set[str]) -> str:
    """Return the CSV column name for a given link's setting.

    Eng36 links use 'a:' columns (actuator action = readback setting).
    Non-Eng36 links use 'setting:' columns.
    """
    if link_id in eng36_ids:
        return f"a:{link_id}"
    return f"setting:{link_id}"


def build_canonical_matrix(
    detail_df: pd.DataFrame,
    canonical_links: list[str],
    checkpoint_min: float,
    eng36_ids: set[str],
) -> pd.DataFrame:
    """Build canonical prefix matrix for one branch.

    Rows: elapsed_min < checkpoint (strictly before), sorted ascending.
    Columns: canonical link order (sorted alphabetically).
    Values: setting frozen to PRECISION decimals.
    """
    # Filter prefix rows (strictly before checkpoint)
    prefix_mask = detail_df["elapsed_min"] < checkpoint_min - 1e-6
    prefix_df = detail_df[prefix_mask].copy()
    prefix_df = prefix_df.sort_values("elapsed_min").reset_index(drop=True)

    # Build matrix
    records = []
    for _, row in prefix_df.iterrows():
        rec = {"elapsed_min": round(float(row["elapsed_min"]), PRECISION)}
        for lid in canonical_links:
            col = get_setting_column(lid, eng36_ids)
            if col in prefix_df.columns:
                val = float(pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0])
                if np.isnan(val):
                    val = 1.0  # Default: fully open
                rec[lid] = round(val, PRECISION)
            else:
                rec[lid] = round(1.0, PRECISION)  # Missing -> default
        records.append(rec)

    return pd.DataFrame(records)


def matrix_to_canonical_string(matrix: pd.DataFrame, canonical_links: list[str]) -> str:
    """Serialize canonical matrix to a deterministic string for hashing.

    Format: one row per line, values separated by '|', rounded to PRECISION.
    No branch name, no file path, no metadata.
    """
    lines = []
    for _, row in matrix.iterrows():
        parts = [f"{float(row['elapsed_min']):.{PRECISION}f}"]
        for lid in canonical_links:
            parts.append(f"{float(row[lid]):.{PRECISION}f}")
        lines.append("|".join(parts))
    return "\n".join(lines)


def compute_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load canonical link order
    canonical_links = load_canonical_link_order()
    print(f"Canonical prefix links: {len(canonical_links)}")

    # Load Eng36 IDs from Scope Contract
    sc = json.loads(SCOPE_CONTRACT.read_text(encoding="utf-8"))
    eng36_ids = set(sc["engineering36_ids"])
    print(f"Eng36 IDs: {len(eng36_ids)}")

    results = {}
    all_matrices = {}

    for cp_label, cp_min in CHECKPOINTS.items():
        print(f"\n{'='*60}")
        print(f"  Checkpoint {cp_label} at t={cp_min}min")
        print(f"{'='*60}")

        cp_results = {}
        cp_matrices = {}

        for branch in BRANCHES:
            fname = BRANCH_FILE_MAP[branch].format(cp=cp_label)
            csv_path = V3_DIR / "work" / f"{fname}.csv"
            if not csv_path.exists():
                # Try alternate naming
                csv_path = V3_DIR / f"{fname}.csv"
            if not csv_path.exists():
                print(f"  [{branch}] CSV not found: {csv_path}")
                cp_results[branch] = {"error": f"CSV not found: {csv_path}"}
                continue

            df = pd.read_csv(csv_path)
            print(f"  [{branch}] rows={len(df)}, elapsed_max={df.elapsed_min.max():.0f}")

            # Build canonical matrix
            matrix = build_canonical_matrix(df, canonical_links, cp_min, eng36_ids)
            print(f"    Canonical matrix: {matrix.shape}")

            # Compute SHA
            text = matrix_to_canonical_string(matrix, canonical_links)
            sha = compute_sha256(text)
            print(f"    Canonical SHA256: {sha[:32]}...")

            cp_results[branch] = {
                "canonical_sha256": sha,
                "canonical_matrix_shape": list(matrix.shape),
                "prefix_rows": int(matrix.shape[0]),
                "prefix_columns": int(matrix.shape[1]),  # elapsed_min + 90 links
            }
            cp_matrices[branch] = matrix

        # Compare across branches
        shas = [r.get("canonical_sha256", "") for r in cp_results.values() if "canonical_sha256" in r]
        unique_shas = set(shas)
        canonical_match = len(unique_shas) <= 1

        # Compute pairwise max abs diff and mismatched cells
        max_abs_diff = 0.0
        mismatched_cells = 0
        total_cells = 0
        branch_list = [b for b in BRANCHES if b in cp_matrices]

        if len(branch_list) >= 2:
            ref_matrix = cp_matrices[branch_list[0]]
            for other_branch in branch_list[1:]:
                other_matrix = cp_matrices[other_branch]
                # Align on elapsed_min
                merged = ref_matrix.merge(
                    other_matrix, on="elapsed_min", suffixes=("_ref", "_other"), how="inner"
                )
                for lid in canonical_links:
                    ref_col = f"{lid}_ref"
                    other_col = f"{lid}_other"
                    if ref_col in merged.columns and other_col in merged.columns:
                        diff = (merged[ref_col] - merged[other_col]).abs()
                        max_abs_diff = max(max_abs_diff, float(diff.max()))
                        mismatched_cells += int((diff > 10 ** (-PRECISION)).sum())
                        total_cells += len(merged)

        results[cp_label] = {
            "checkpoint_elapsed_min": cp_min,
            "branches": cp_results,
            "canonical_sha256_all_match": canonical_match,
            "unique_canonical_sha_count": len(unique_shas),
            "max_abs_setting_difference": float(max_abs_diff),
            "mismatched_cell_count": int(mismatched_cells),
            "total_cells_compared": int(total_cells),
            "canonical_matrix_shape": [int(ref_matrix.shape[0]), len(canonical_links)] if branch_list else [],
            "old_hash_failure": None,
        }

        if not canonical_match:
            # Determine if it's a serialization defect or real numerical difference
            if max_abs_diff <= 10 ** (-PRECISION + 1):  # Within 10x precision
                results[cp_label]["old_hash_failure"] = "serialization_defect"
                print(f"  >> Old hash failure: SERIALIZATION DEFECT (max_diff={max_abs_diff:.2e})")
            else:
                results[cp_label]["old_hash_failure"] = "numerical_divergence"
                print(f"  >> Old hash failure: NUMERICAL DIVERGENCE (max_diff={max_abs_diff:.2e})")
        else:
            print(f"  >> Canonical SHA match: TRUE (all 4 branches identical)")

        all_matrices.update(cp_matrices)

    # Write audit JSON
    audit = {
        "audit_name": "canonical_prefix_schedule_hash",
        "audit_version": "1.0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope_contract": str(SCOPE_CONTRACT),
        "canonical_link_count": len(canonical_links),
        "canonical_link_order": canonical_links,
        "precision": PRECISION,
        "excluded_from_hash": [
            "branch_name", "file_path", "csv_column_order",
            "metadata", "requested_action", "post_checkpoint_data",
        ],
        "checkpoints": results,
        "gate3_status": "PASS" if all(
            r.get("canonical_sha256_all_match", False) for r in results.values()
        ) else ("PARTIAL" if any(
            r.get("old_hash_failure") == "serialization_defect" for r in results.values()
        ) else "BLOCKED"),
        "wall_time_sec": round(time.time() - t0, 1),
    }

    audit_path = OUT_DIR / "canonical_prefix_hash_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote: {audit_path}")

    # Write per-branch canonical matrices (for inspection)
    for branch, matrix in all_matrices.items():
        for cp_label in CHECKPOINTS:
            if cp_label in branch:
                out_csv = OUT_DIR / f"canonical_prefix_{cp_label}_{branch}.csv"
                matrix.to_csv(out_csv, index=False)

    print(f"\nGate 3 Canonical Prefix Hash Status: {audit['gate3_status']}")
    print(f"Done in {audit['wall_time_sec']:.1f}s")


if __name__ == "__main__":
    main()
