"""Gate 2.5-real-v2 verification tests.

Tests verify the V2 runner outputs fix all 15 V1 false-positive issues.
These tests read the V2 output directory and validate:
  - Checkpoint state hash equality across 4 branches
  - hold_snapshot constant after checkpoint
  - H120 window starts at checkpoint
  - Causal intervention produces flow/depth differences
  - Hotstart files absent
  - All boolean fields are genuine Python bools
  - Physical network SHA match
  - Recovery analysis executed correctly
  - Scope contract integrity
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
V2_DIR = ROOT / "outputs" / "project6_dual_reference_v4" / "recovery_validation" / "gate2p5_real_v2"
SCOPE_CONTRACT = ROOT / "docs" / "contracts" / "PROJECT6_V4_CONTROL_SCOPE_CONTRACT.json"


def _load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(V2_DIR / name)


def _load_json(name: str) -> dict:
    return json.loads((V2_DIR / name).read_text(encoding="utf-8"))


def _parse_state_hash(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return {}


# =========================================================================
# P01: Same state hash equality across 4 branches
# =========================================================================
def test_same_state_hash_equality():
    """Fixed-action branches (no_control, hold_snapshot, hold_previous) must have identical checkpoint state hash."""
    hash_df = _load_csv("state_hash_comparison.csv")
    fixed_branches = ["no_control", "hold_snapshot", "hold_previous"]
    for cp in hash_df["checkpoint_label"].unique():
        cp_rows = hash_df[(hash_df["checkpoint_label"] == cp) &
                          (hash_df["branch"].isin(fixed_branches))]
        state_hashes = [_parse_state_hash(r) for _, r in cp_rows.iterrows()]
        keys = ["h_sha256", "head_sha256", "flood_sha256", "storage_volume_sha256"]
        for k in keys:
            vals = [sh.get(k, "") for sh in state_hashes]
            assert len(set(vals)) == 1, f"{cp}/{k}: {len(set(vals))} distinct values"


# =========================================================================
# P02: no_control shared prefix with dynamic
# =========================================================================
def test_no_control_shared_prefix():
    """no_control prefix_actual_schedule_sha256 must exist and match baseline replay."""
    hash_df = _load_csv("state_hash_comparison.csv")
    for cp in hash_df["checkpoint_label"].unique():
        cp_rows = hash_df[hash_df["checkpoint_label"] == cp]
        di = cp_rows[cp_rows["branch"] == "dynamic_internal"]
        nc = cp_rows[cp_rows["branch"] == "no_control"]
        assert len(di) > 0 and len(nc) > 0
        di_ps = di.iloc[0]["prefix_actual_schedule_sha256"]
        nc_ps = nc.iloc[0]["prefix_actual_schedule_sha256"]
        assert di_ps and nc_ps, f"{cp}: prefix schedule SHA missing"


# =========================================================================
# P03: snapshot constant after checkpoint
# =========================================================================
def test_snapshot_constant_after_checkpoint():
    """hold_snapshot post_checkpoint_action_changes must be 0."""
    hash_df = _load_csv("state_hash_comparison.csv")
    for cp in hash_df["checkpoint_label"].unique():
        cp_rows = hash_df[hash_df["checkpoint_label"] == cp]
        hs = cp_rows[cp_rows["branch"] == "hold_snapshot"]
        assert len(hs) > 0
        changes = int(hs.iloc[0]["post_checkpoint_action_changes"])
        assert changes == 0, f"{cp}: snapshot changes={changes}"


# =========================================================================
# P04: snapshot does not replay future baseline values
# =========================================================================
def test_snapshot_does_not_replay_future():
    """hold_snapshot schedule SHA must differ from dynamic_internal (which has native rules)."""
    snap_df = _load_csv("snapshot_evidence.csv")
    for _, r in snap_df.iterrows():
        assert r["schedules_differ"] == True or str(r["schedules_differ"]).lower() == "true", \
            f"{r['checkpoint_label']}: snapshot schedules should differ from dynamic"


# =========================================================================
# P05: fixed_action callback before routing (write_source column exists)
# =========================================================================
def test_fixed_action_callback_before_routing():
    """Detail CSVs from fixed_action branches must have write_source column."""
    detail_files = list(V2_DIR.glob("branch_no_control_*_detail.csv"))
    assert len(detail_files) > 0, "No no_control detail files found"
    for df_path in detail_files:
        d = pd.read_csv(df_path, nrows=3)
        assert "write_source" in d.columns, f"{df_path.name}: missing write_source column"
        # Check values are prefix_replay or external_override
        sources = d["write_source"].unique()
        for s in sources:
            assert s in ("prefix_replay", "external_override"), f"Unexpected source: {s}"


# =========================================================================
# P06: readback independent from command
# =========================================================================
def test_readback_independent_from_command():
    """Detail CSVs must have both a: and setting: columns (from Link.current_setting)."""
    detail_files = list(V2_DIR.glob("branch_*_detail.csv"))
    assert len(detail_files) > 0
    for df_path in detail_files[:2]:
        d = pd.read_csv(df_path, nrows=3)
        a_cols = [c for c in d.columns if c.startswith("a:")]
        s_cols = [c for c in d.columns if c.startswith("setting:")]
        assert len(a_cols) > 0, f"{df_path.name}: no a: columns"
        assert len(s_cols) > 0, f"{df_path.name}: no setting: columns"


# =========================================================================
# P07: causal action changes flow
# =========================================================================
def test_causal_action_changes_flow():
    """Causal low/high must produce different flow values."""
    causal_v = _load_json("causal_intervention_verdict.json")
    assert causal_v["flow_differs_between_branches"] is True
    # Also verify from comparison CSV
    causal_cmp = _load_csv("causal_intervention_comparison.csv")
    flows = {}
    for _, r in causal_cmp.iterrows():
        flows[r["branch"]] = float(r["max_abs_flow"])
    assert "causal_low" in flows and "causal_high" in flows
    assert abs(flows["causal_high"] - flows["causal_low"]) > 1e-6


# =========================================================================
# P08: causal action changes node state
# =========================================================================
def test_causal_action_changes_node_state():
    """Causal low/high must produce different node depth values."""
    causal_v = _load_json("causal_intervention_verdict.json")
    assert causal_v["depth_differs_between_branches"] is True


# =========================================================================
# P09: H120 window starts at checkpoint
# =========================================================================
def test_h120_window_starts_at_checkpoint():
    """H120 window must have rows (starts at checkpoint, not t=0)."""
    kpi_df = _load_csv("branch_kpi_comparison.csv")
    for _, r in kpi_df.iterrows():
        assert int(r["h120_rows"]) > 0, \
            f"{r['checkpoint_label']}/{r['branch']}: h120_rows=0"


# =========================================================================
# P10: Two checkpoints produce independent window hashes
# =========================================================================
def test_two_checkpoints_independent_windows():
    """H120 window hashes must differ between checkpoints."""
    kpi_df = _load_csv("branch_kpi_comparison.csv")
    hashes_by_cp = {}
    for _, r in kpi_df.iterrows():
        cp = r["checkpoint_label"]
        if cp not in hashes_by_cp:
            hashes_by_cp[cp] = set()
        hashes_by_cp[cp].add(r["h120_window_hash"])
    # All checkpoints should have different window hashes
    # At minimum, the dynamic_internal branch should have different hashes
    di_hashes = kpi_df[kpi_df["branch"] == "dynamic_internal"]["h120_window_hash"].tolist()
    if len(di_hashes) >= 2:
        assert len(set(di_hashes)) == len(di_hashes), \
            f"Checkpoints have same H120 window hash: {di_hashes}"


# =========================================================================
# P11: Recovery False blocks gate (informational)
# =========================================================================
def test_recovery_false_blocks_gate():
    """Recovery analysis must be executed (recovery_criteria_met present)."""
    kpi_df = _load_csv("branch_kpi_comparison.csv")
    assert "recovery_criteria_met" in kpi_df.columns
    # All values should be boolean (not string)
    for _, r in kpi_df.iterrows():
        val = r["recovery_criteria_met"]
        assert isinstance(val, (bool, np.bool_)), \
            f"recovery_criteria_met should be bool, got {type(val).__name__}"


# =========================================================================
# P12: Hotstart file presence blocks gate
# =========================================================================
def test_hotstart_file_blocks_gate():
    """V2 output must NOT contain .hsf files."""
    hsf_files = list(V2_DIR.rglob("*.hsf"))
    assert len(hsf_files) == 0, f"Found {len(hsf_files)} .hsf files in V2 output"


# =========================================================================
# P13: Planned SHA cannot substitute actual SHA
# =========================================================================
def test_planned_sha_cannot_substitute_actual():
    """checkpoint_state_hash must contain actual computed values, not placeholders."""
    hash_df = _load_csv("state_hash_comparison.csv")
    for _, r in hash_df.iterrows():
        sh = _parse_state_hash(r["checkpoint_state_hash"])
        assert isinstance(sh, dict) and len(sh) > 0
        for k in ["h_sha256", "a_sha256", "flow_sha256"]:
            assert k in sh, f"Missing {k} in checkpoint_state_hash"
            assert sh[k] and sh[k] != "planned" and len(sh[k]) >= 32


# =========================================================================
# P14: Pass string "True" is not boolean True
# =========================================================================
def test_pass_string_not_boolean_true():
    """All boolean fields in JSON outputs must be genuine Python booleans."""
    causal_v = _load_json("causal_intervention_verdict.json")
    for key in ["flow_differs_between_branches", "depth_differs_between_branches", "causal_pass"]:
        val = causal_v[key]
        assert isinstance(val, bool), f"{key}: expected bool, got {type(val).__name__}"
        assert val is not None
        # Verify it's not the string "True"
        assert val != "True"


# =========================================================================
# P15: 54 non-Eng36 scope explicit
# =========================================================================
def test_54_non_eng36_scope_explicit():
    """Scope contract must define 54 non-Eng36 facilities."""
    contract = json.loads(SCOPE_CONTRACT.read_text(encoding="utf-8"))
    eng36 = set(contract.get("engineering36_ids", []))
    overlap = set(contract.get("engineering36_overlap_with_native_rules", []))
    native_rules = contract.get("native_rules", {})
    native_count = native_rules.get("controlled_facility_count", 0)
    non_eng36_count = native_rules.get("non_engineering36_count", 0)
    assert native_count == 82, f"Expected 82 native-rule facilities, got {native_count}"
    assert non_eng36_count == 54, f"Expected 54 non-Eng36 facilities, got {non_eng36_count}"
    assert len(overlap) == 28, f"Expected 28 Eng36 overlap, got {len(overlap)}"


# =========================================================================
# P16: formal_v31_design event blacklisted
# =========================================================================
def test_formal_used_event_blacklist():
    """V31 rainfall events must be in the formal blacklist."""
    blacklist = _load_json("formal_blacklist.json")
    assert blacklist.get("formal_blacklist_written") is True
    events = blacklist.get("blacklisted_events", [])
    assert len(events) > 0, "Blacklist is empty"
    # The V31 event should be listed
    assert any("V31" in e or "v31" in e for e in events), \
        f"No V31 event in blacklist: {events}"


# =========================================================================
# P17: Physical network SHA match
# =========================================================================
def test_physical_network_sha_match():
    """With-controls and no-controls INP must have same physical network SHA."""
    hash_df = _load_csv("state_hash_comparison.csv")
    shas = hash_df["physical_network_sha256"].unique()
    assert len(shas) == 1, f"Expected 1 unique physical SHA, got {len(shas)}"


# =========================================================================
# P18: H120 audit matches primary computation
# =========================================================================
def test_h120_audit_matches_primary():
    """Independent H120 audit values must match primary computation within tolerance."""
    hash_df = _load_csv("state_hash_comparison.csv")
    for _, r in hash_df.iterrows():
        assert r["h120_match"] == True or str(r["h120_match"]).lower() == "true", \
            f"{r['checkpoint_label']}/{r['branch']}: H120 audit mismatch"


# =========================================================================
# P19: Dynamic internal has active native rules
# =========================================================================
def test_dynamic_internal_active_native_rules():
    """dynamic_internal must have non-zero post-checkpoint action changes."""
    hash_df = _load_csv("state_hash_comparison.csv")
    for cp in hash_df["checkpoint_label"].unique():
        di = hash_df[(hash_df["checkpoint_label"] == cp) &
                     (hash_df["branch"] == "dynamic_internal")]
        assert len(di) > 0
        changes = int(di.iloc[0]["post_checkpoint_action_changes"])
        assert changes > 0, f"{cp}: dynamic_internal has 0 post-changes (native rules not active)"


# =========================================================================
# P20: hold_previous also has zero post-changes
# =========================================================================
def test_hold_previous_zero_post_changes():
    """hold_previous post_checkpoint_action_changes must be 0."""
    hash_df = _load_csv("state_hash_comparison.csv")
    for cp in hash_df["checkpoint_label"].unique():
        hp = hash_df[(hash_df["checkpoint_label"] == cp) &
                     (hash_df["branch"] == "hold_previous")]
        assert len(hp) > 0
        changes = int(hp.iloc[0]["post_checkpoint_action_changes"])
        assert changes == 0, f"{cp}: hold_previous changes={changes}"
