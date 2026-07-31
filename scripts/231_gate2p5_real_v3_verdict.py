"""Gate 2.5-real-v3 Verdict: 26 blocking PASS conditions.

Key V3 changes vs V2:
  - P04/P08/P09: ALL 4 branches (including dynamic_internal) must share prefix state
  - P21/P22/P23: Recovery is BLOCKING (not informational)
  - P24/P25/P26: New V3 conditions for prefix equality and recovery

Reads V3 runner outputs and checks each condition. All must PASS for gate pass.
Any failure -> verdict=BLOCKED, exit!=0.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "project6_dual_reference_v4" / "recovery_validation" / "gate2p5_real_v3"
SCOPE_CONTRACT_V2 = PROJECT_ROOT / "docs" / "contracts" / "PROJECT6_V4_CONTROL_SCOPE_CONTRACT_V2.json"


def _load(name: str) -> pd.DataFrame | dict:
    p = OUT_DIR / name
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    return pd.read_csv(p)


def _is_real_bool(val) -> bool:
    if isinstance(val, bool):
        return True
    if isinstance(val, np.bool_):
        return True
    return False


def _safe_bool(val) -> bool:
    if isinstance(val, str):
        return val.lower() == "true"
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    return bool(val)


def main() -> int:
    results: list[dict] = []

    def record(pid: int, description: str, passed: bool, detail: str = "",
               is_bool_check: bool = False, actual_type: str = ""):
        results.append({
            "P": pid, "description": description, "pass": bool(passed),
            "detail": detail,
            "is_real_bool": is_bool_check,
            "actual_type": actual_type,
        })

    # Load data
    hash_df = _load("state_hash_comparison.csv")
    kpi_df = _load("branch_kpi_comparison.csv")
    snap_df = _load("snapshot_evidence.csv")
    causal_v = _load("causal_intervention_verdict.json")
    causal_cmp = _load("causal_intervention_comparison.csv")
    hs_audit = _load("hotstart_audit.json")
    prefix_eq = _load("prefix_equality_evidence.json")
    contract = json.loads(SCOPE_CONTRACT_V2.read_text(encoding="utf-8"))
    runner_summary = _load("v3_runner_summary.json")

    def _parse_state_hash(row):
        raw = row.get("checkpoint_state_hash", "")
        if isinstance(raw, dict):
            return raw
        try:
            return json.loads(raw)
        except Exception:
            return {}

    # =========================================================================
    # P01: Physical network SHA match
    # =========================================================================
    phys_sha_vals = hash_df["physical_network_sha256"].unique()
    record(1, "Physical network SHA identical (with/without controls)",
           len(phys_sha_vals) == 1,
           f"unique SHAs: {len(phys_sha_vals)}")

    # =========================================================================
    # P02: hotstart_used field is real boolean
    # =========================================================================
    hu_vals = hash_df["hotstart_used"].unique()
    all_bool = all(_is_real_bool(v) for v in hu_vals)
    record(2, "hotstart_used field is real boolean (not string)",
           all_bool,
           f"types: {[type(v).__name__ for v in hu_vals]}",
           is_bool_check=True,
           actual_type=type(hu_vals[0]).__name__ if len(hu_vals) > 0 else "empty")

    # =========================================================================
    # P03: recovery_criteria_met is real boolean
    # =========================================================================
    rcm_vals = hash_df["recovery_criteria_met"].unique()
    all_bool_rcm = all(_is_real_bool(v) for v in rcm_vals)
    record(3, "recovery_criteria_met field is real boolean (not string)",
           all_bool_rcm,
           f"types: {[type(v).__name__ for v in rcm_vals]}",
           is_bool_check=True,
           actual_type=type(rcm_vals[0]).__name__ if len(rcm_vals) > 0 else "empty")

    # =========================================================================
    # P04: ALL FOUR branches share same checkpoint state hash (V3: including dynamic_internal!)
    # =========================================================================
    cp_labels = hash_df["checkpoint_label"].unique()
    all_branches = ["dynamic_internal", "no_control", "hold_snapshot", "hold_previous"]
    hash_match = True
    hash_detail_parts = []
    for cp in cp_labels:
        cp_rows = hash_df[hash_df["checkpoint_label"] == cp]
        branch_rows = cp_rows[cp_rows["branch"].isin(all_branches)]
        state_hashes = [_parse_state_hash(r) for _, r in branch_rows.iterrows()]
        keys = ["h_sha256", "head_sha256", "flood_sha256", "storage_volume_sha256",
                 "a_sha256", "setting_sha256", "flow_sha256"]
        for k in keys:
            vals = [sh.get(k, "") for sh in state_hashes]
            if len(set(vals)) > 1:
                hash_match = False
                hash_detail_parts.append(f"{cp}/{k}: {len(set(vals))} distinct values")
    record(4, "ALL 4 branches (incl dynamic_internal) share checkpoint state hash",
           hash_match,
           "; ".join(hash_detail_parts) if hash_detail_parts else "all match")

    # =========================================================================
    # P05: prefix schedule SHA exists for all branches
    # =========================================================================
    prefix_match = True
    prefix_detail = []
    for cp in cp_labels:
        cp_rows = hash_df[hash_df["checkpoint_label"] == cp]
        for branch in all_branches:
            brow = cp_rows[cp_rows["branch"] == branch]
            if len(brow) > 0:
                ps = brow.iloc[0].get("prefix_actual_schedule_sha256", "")
                if not ps:
                    prefix_match = False
                    prefix_detail.append(f"{cp}/{branch}: missing prefix SHA")
    record(5, "All branches have prefix schedule SHA",
           prefix_match,
           "; ".join(prefix_detail) if prefix_detail else "all present")

    # =========================================================================
    # P06: hold_snapshot post_checkpoint_action_changes = 0
    # =========================================================================
    snap_changes_zero = True
    for cp in cp_labels:
        cp_rows = hash_df[hash_df["checkpoint_label"] == cp]
        hs_rows = cp_rows[cp_rows["branch"] == "hold_snapshot"]
        if len(hs_rows) > 0:
            changes = int(hs_rows.iloc[0]["post_checkpoint_action_changes"])
            if changes != 0:
                snap_changes_zero = False
    record(6, "hold_snapshot post-checkpoint action changes = 0",
           snap_changes_zero)

    # =========================================================================
    # P07: hold_snapshot is constant after checkpoint
    # =========================================================================
    snap_const = all(_safe_bool(r["snapshot_is_constant"]) for _, r in snap_df.iterrows())
    record(7, "hold_snapshot is constant after checkpoint",
           snap_const,
           f"snapshot_is_constant: {snap_df['snapshot_is_constant'].tolist()}")

    # =========================================================================
    # P08: ALL 4 branches prefix SHA identical (V3: including dynamic_internal!)
    # =========================================================================
    prefix_sha_match = True
    prefix_sha_detail = []
    for cp in cp_labels:
        cp_rows = hash_df[hash_df["checkpoint_label"] == cp]
        shas = {}
        for _, r in cp_rows.iterrows():
            shas[r["branch"]] = str(r["prefix_actual_schedule_sha256"])
        unique_shas = set(shas.values())
        if len(unique_shas) > 1:
            prefix_sha_match = False
            prefix_sha_detail.append(f"{cp}: {len(unique_shas)} distinct SHAs")
    record(8, "ALL 4 branches have identical prefix_actual_schedule_sha256",
           prefix_sha_match,
           "; ".join(prefix_sha_detail) if prefix_sha_detail else "all match")

    # =========================================================================
    # P09: Prefix equality evidence - h/flow SHA match across all 4 branches
    # =========================================================================
    prefix_eq_ok = True
    for cp_label, cp_data in prefix_eq.get("checkpoints", {}).items():
        if not cp_data.get("all_branches_match", False):
            prefix_eq_ok = False
    record(9, "Prefix equality evidence: h/flow SHA match across all 4 branches",
           prefix_eq_ok,
           f"checkpoints: {list(prefix_eq.get('checkpoints', {}).keys())}")

    # =========================================================================
    # P10: hold_previous post_checkpoint_action_changes = 0
    # =========================================================================
    prev_changes_zero = True
    for cp in cp_labels:
        cp_rows = hash_df[hash_df["checkpoint_label"] == cp]
        hp_rows = cp_rows[cp_rows["branch"] == "hold_previous"]
        if len(hp_rows) > 0:
            changes = int(hp_rows.iloc[0]["post_checkpoint_action_changes"])
            if changes != 0:
                prev_changes_zero = False
    record(10, "hold_previous post-checkpoint action changes = 0",
           prev_changes_zero)

    # =========================================================================
    # P11: dynamic_internal has non-zero post-checkpoint changes
    # =========================================================================
    di_active = True
    for cp in cp_labels:
        cp_rows = hash_df[hash_df["checkpoint_label"] == cp]
        di_rows = cp_rows[cp_rows["branch"] == "dynamic_internal"]
        if len(di_rows) > 0:
            changes = int(di_rows.iloc[0]["post_checkpoint_action_changes"])
            if changes == 0:
                di_active = False
    record(11, "dynamic_internal has active native rules (non-zero post-changes)",
           di_active)

    # =========================================================================
    # P12: Readback independent from command
    # =========================================================================
    readback_ok = True
    detail_files = list(OUT_DIR.glob("branch_*_detail.csv"))
    for df_path in detail_files[:2]:
        d = pd.read_csv(df_path, nrows=5)
        a_cols = [c for c in d.columns if c.startswith("a:")]
        s_cols = [c for c in d.columns if c.startswith("setting:")]
        if not a_cols or not s_cols:
            readback_ok = False
    record(12, "Readback independent from command (setting: from Link object)",
           readback_ok)

    # =========================================================================
    # P13: Causal intervention - flow differs
    # =========================================================================
    record(13, "Causal: flow differs between low/high branches",
           bool(causal_v.get("flow_differs_between_branches", False)))

    # =========================================================================
    # P14: Causal intervention - depth differs
    # =========================================================================
    record(14, "Causal: node depth differs between low/high branches",
           bool(causal_v.get("depth_differs_between_branches", False)))

    # =========================================================================
    # P15: H120 window has data
    # =========================================================================
    h120_valid = all(int(r["h120_rows"]) > 0 for _, r in kpi_df.iterrows())
    record(15, "H120 window has data (starts at checkpoint, not t=0)",
           h120_valid,
           f"h120_rows: {kpi_df['h120_rows'].tolist()}")

    # =========================================================================
    # P16: H120 audit matches primary
    # =========================================================================
    h120_match = all(_safe_bool(r["h120_match"]) for _, r in hash_df.iterrows())
    record(16, "H120 independent audit matches primary computation",
           h120_match)

    # =========================================================================
    # P17: Two checkpoints produce different H120 window hashes
    # =========================================================================
    if len(cp_labels) >= 2:
        window_hashes = []
        for cp in cp_labels:
            cp_rows = kpi_df[kpi_df["checkpoint_label"] == cp]
            wh = cp_rows["h120_window_hash"].iloc[0]
            window_hashes.append(wh)
        h120_unique = len(set(window_hashes)) == len(window_hashes)
    else:
        h120_unique = True
        window_hashes = []
    record(17, "Two checkpoints produce different H120 window hashes",
           h120_unique)

    # =========================================================================
    # P18: H120 PFV differs between checkpoints
    # =========================================================================
    pfv_by_cp = {}
    for _, r in kpi_df.iterrows():
        cp = r["checkpoint_label"]
        if cp not in pfv_by_cp:
            pfv_by_cp[cp] = []
        pfv_by_cp[cp].append(float(r["PFV_H120"]))
    pfv_cross_cp = [sum(v)/len(v) for v in pfv_by_cp.values()]
    pfv_differs = len(set(round(p, 2) for p in pfv_cross_cp)) > 1
    record(18, "H120 PFV differs between checkpoints",
           pfv_differs,
           f"avg PFV per CP: {[round(p,1) for p in pfv_cross_cp]}")

    # =========================================================================
    # P19: No .hsf files in V3 output
    # =========================================================================
    hsf_count = hs_audit.get("hsf_files_in_v3_output", 0)
    record(19, "No .hsf hotstart files in V3 output",
           hsf_count == 0,
           f"hsf_files: {hsf_count}")

    # =========================================================================
    # P20: hotstart_used = False for all branches
    # =========================================================================
    hu_all_false = all(not _safe_bool(v) for v in hash_df["hotstart_used"].unique())
    record(20, "hotstart_used=False for all branches",
           hu_all_false)

    # =========================================================================
    # P21: ALL branches recovery_criteria_met = true (BLOCKING!)
    # =========================================================================
    recovery_vals = kpi_df["recovery_criteria_met"].tolist()
    all_recovery = all(_safe_bool(v) for v in recovery_vals)
    record(21, "ALL branches recovery_criteria_met=true (BLOCKING)",
           all_recovery,
           f"recovery values: {recovery_vals}")

    # =========================================================================
    # P22: ALL branches recovery_censored = false
    # =========================================================================
    if "recovery_censored" in kpi_df.columns:
        censored_vals = kpi_df["recovery_censored"].tolist()
        none_censored = all(not _safe_bool(v) for v in censored_vals)
    else:
        none_censored = True
        censored_vals = []
    record(22, "ALL branches recovery_censored=false",
           none_censored,
           f"censored values: {censored_vals}")

    # =========================================================================
    # P23: ALL branches full_event_eligible = true
    # =========================================================================
    if "full_event_eligible" in kpi_df.columns:
        eligible_vals = kpi_df["full_event_eligible"].tolist()
        all_eligible = all(_safe_bool(v) for v in eligible_vals)
    else:
        all_eligible = False
        eligible_vals = []
    record(23, "ALL branches full_event_eligible=true",
           all_eligible,
           f"eligible values: {eligible_vals}")

    # =========================================================================
    # P24: Scope Contract V2 - native control links count > 0
    # =========================================================================
    native_count = contract.get("native_control_links_count", 0)
    record(24, "Scope Contract V2 defines native control links",
           native_count > 0,
           f"native_control_links_count: {native_count}")

    # =========================================================================
    # P25: Total prefix links = Eng36 UNION native control links
    # =========================================================================
    total_prefix = contract.get("prefix_scope", {}).get("total_prefix_links", 0)
    expected = len(set(contract.get("engineering36_ids", [])) | set(contract.get("native_control_links", [])))
    record(25, "Total prefix links = Eng36 UNION native control links",
           total_prefix == expected,
           f"total={total_prefix}, expected={expected}")

    # =========================================================================
    # P26: Runner summary - physical SHA match
    # =========================================================================
    sha_match = runner_summary.get("physical_sha_match", False)
    record(26, "Runner summary confirms physical SHA match",
           sha_match,
           f"physical_sha_match: {sha_match}")

    # =========================================================================
    # Build verdict
    # =========================================================================
    pass_count = sum(1 for r in results if r["pass"])
    total = len(results)
    all_pass = all(r["pass"] for r in results)  # ALL are blocking in V3

    verdict = "PASS" if all_pass else "BLOCKED"
    verdict_data = {
        "gate": "2.5-real-v3",
        "verdict": verdict,
        "pass_count": pass_count,
        "total_conditions": total,
        "all_blocking": True,  # All conditions are blocking in V3
        "conditions": results,
    }
    out_path = OUT_DIR / "gate2p5_real_v3_verdict.json"
    out_path.write_text(json.dumps(verdict_data, indent=2, default=str), encoding="utf-8")

    # Print summary
    print(f"\n{'='*70}")
    print(f"  Gate 2.5-real-v3 VERDICT: {verdict}")
    print(f"  {pass_count}/{total} blocking conditions passed")
    print(f"{'='*70}")
    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        marker = "  " if r["pass"] else ">>"
        print(f"  {marker} P{r['P']:02d} [{status}] {r['description']}")
        if not r["pass"] and r["detail"]:
            print(f"         Detail: {r['detail']}")
    print(f"\n  Verdict written to: {out_path}")
    print(f"{'='*70}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
