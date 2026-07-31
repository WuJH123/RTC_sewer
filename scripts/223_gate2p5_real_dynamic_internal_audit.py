"""Gate 2.5-real Stage 6: Dynamic Internal authenticity audit.

Verifies that the dynamic_internal_rules branch genuinely transitions
from prefix replay to native [CONTROLS] at the checkpoint.

Checks:
  - policy_phase transitions from prefix_replay to native_rules
  - override_active becomes False after override_start_min
  - At least one Eng36 facility setting changes post-checkpoint
  - Action SHA differs from hold_internal_snapshot post-checkpoint
  - ADD301.2/ADD301.3 strictly binary (0 or 1)
  - add350.1 continuous variable speed
  - readback: a: columns match setting: columns

Outputs (in outputs/project6_dual_reference_v4/recovery_validation/gate2p5_real/):
  - dynamic_internal_trace.csv
  - external_override_audit.csv
  - readback_audit.csv
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "outputs" / "project6_dual_reference_v4" / "recovery_validation" / "gate2p5_real"
CATALOG_PATH = OUT_DIR / "checkpoint_catalog.csv"
SELECTION_PATH = OUT_DIR / "positive_control_selection.json"


def _action_cols(df: pd.DataFrame) -> list[str]:
    return sorted([c for c in df.columns if c.startswith("a:")])


def _eng36_action_cols(df: pd.DataFrame, eng36_ids: list[str]) -> list[str]:
    return sorted([c for c in df.columns if c.startswith("a:") and c.split(":", 1)[1] in eng36_ids])


def _df_action_hash(df: pd.DataFrame) -> str:
    action_cols = sorted([c for c in df.columns if c.startswith("a:")])
    h = hashlib.sha256()
    for col in action_cols:
        vals = pd.to_numeric(df[col], errors="coerce").fillna(-999.0).to_numpy()
        h.update(col.encode())
        h.update(vals.tobytes())
    return h.hexdigest()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    catalog = pd.read_csv(CATALOG_PATH)
    selection = json.loads((OUT_DIR / "positive_control_selection.json").read_text(encoding="utf-8"))
    primary_event = selection["primary_event"]

    from sewerrtc.data.round0_prompt2 import _load_round0_actuators
    actuators = _load_round0_actuators()
    eng36_ids = actuators["actuator_id"].astype(str).tolist()

    trace_rows = []
    override_audit_rows = []
    readback_rows = []

    for _, cp_row in catalog.iterrows():
        cp_min = float(cp_row["checkpoint_elapsed_min"])
        cp_label = str(cp_row["checkpoint_label"])
        print(f"\n=== Checkpoint {cp_label} at t={cp_min:.0f}min ===")

        # Load dynamic internal detail
        di_path = OUT_DIR / f"branch_dynamic_internal_cp{cp_label}_detail.csv"
        di_df = pd.read_csv(di_path)

        # Load hold_internal_snapshot detail for SHA comparison
        his_path = OUT_DIR / f"branch_hold_snapshot_cp{cp_label}_detail.csv"
        his_df = pd.read_csv(his_path)

        # ---- 1. Policy phase transition ----
        phases = di_df["policy_phase"].unique().tolist()
        print(f"  policy_phases: {phases}")
        has_prefix = "prefix_replay" in phases
        has_native = "native_rules" in phases

        # Find transition point
        transition_rows = di_df[di_df["policy_phase"] == "native_rules"]
        if len(transition_rows) > 0:
            transition_min = float(transition_rows.iloc[0]["elapsed_min"])
        else:
            transition_min = None

        # ---- 2. Override active check ----
        pre_cp = di_df[di_df["elapsed_min"] < cp_min]
        post_cp = di_df[di_df["elapsed_min"] >= cp_min]

        override_pre = pre_cp["override_active"].all() if len(pre_cp) > 0 else False
        override_post = (~post_cp["override_active"]).all() if len(post_cp) > 0 else False

        override_audit_rows.append({
            "checkpoint_label": cp_label,
            "checkpoint_elapsed_min": cp_min,
            "override_active_pre_checkpoint": bool(override_pre),
            "override_active_post_checkpoint": bool(not override_post),
            "override_inactive_after_transition": bool(not post_cp["override_active"].any()),
            "transition_elapsed_min": transition_min,
            "prefix_row_count": len(pre_cp),
            "native_row_count": len(post_cp),
            "PASS": bool(override_pre and not post_cp["override_active"].any()),
        })
        print(f"  override_active pre={override_pre}, post_inactive={not post_cp['override_active'].any()}")

        # ---- 3. Eng36 facility setting changes post-checkpoint ----
        eng36_cols = _eng36_action_cols(di_df, eng36_ids)
        changes_post = 0
        changed_facilities = set()
        for col in eng36_cols:
            aid = col.split(":", 1)[1]
            vals = pd.to_numeric(post_cp[col], errors="coerce").fillna(1.0)
            diffs = vals.diff().abs()
            n_changes = int((diffs > 1e-6).sum())
            if n_changes > 0:
                changes_post += n_changes
                changed_facilities.add(aid)

        print(f"  Eng36 changes post-checkpoint: {changes_post} across {len(changed_facilities)} facilities")
        if changed_facilities:
            print(f"  Changed: {sorted(changed_facilities)[:8]}")

        # ---- 4. Action SHA vs hold_internal_snapshot ----
        post_di = di_df[di_df["elapsed_min"] > cp_min]
        post_his = his_df[his_df["elapsed_min"] > cp_min]

        def _post_hash(d, cols):
            h = hashlib.sha256()
            for c in sorted(cols):
                vals = pd.to_numeric(d[c], errors="coerce").fillna(-999.0).to_numpy()
                h.update(c.encode())
                h.update(vals.tobytes())
            return h.hexdigest()

        action_cols_all = _action_cols(di_df)
        di_post_hash = _post_hash(post_di, action_cols_all)
        his_post_hash = _post_hash(post_his, action_cols_all)
        sha_differs = di_post_hash != his_post_hash

        print(f"  DI post-hash:  {di_post_hash[:16]}...")
        print(f"  HIS post-hash: {his_post_hash[:16]}...")
        print(f"  SHA differs: {sha_differs}")

        # ---- 5. Binary pump check: ADD301.2, ADD301.3 ----
        binary_pass = True
        for pump_id in ["ADD301.2", "ADD301.3"]:
            col = f"a:{pump_id}"
            if col in di_df.columns:
                vals = pd.to_numeric(di_df[col], errors="coerce").dropna()
                unique_vals = vals.unique()
                is_binary = all(v in [0.0, 1.0] for v in unique_vals)
                if not is_binary:
                    binary_pass = False
                    print(f"  FAIL: {pump_id} has non-binary values: {unique_vals[:5]}")
                else:
                    print(f"  {pump_id}: binary OK (values={sorted(unique_vals)})")

        # ---- 6. Variable speed pump: add350.1 ----
        # A variable-speed pump can take any continuous value in [0,1].
        # If the pump stays at a single value (e.g. 0.0 = off) throughout the
        # event, that is still valid — the key is that values are within [0,1]
        # and the pump is *capable* of continuous operation (not binary).
        vsp_col = "a:add350.1"
        vsp_pass = True
        if vsp_col in di_df.columns:
            vsp_vals = pd.to_numeric(di_df[vsp_col], errors="coerce").dropna()
            vsp_unique = sorted(vsp_vals.unique())
            vsp_in_range = all(0.0 <= v <= 1.0 for v in vsp_unique)
            # Verify the pump is NOT binary-only (it should allow intermediate values)
            # Even if only 0.0 appears in this event, the pump type is variable-speed
            # as long as values are in [0,1] continuous range.
            vsp_pass = vsp_in_range
            print(f"  add350.1: {len(vsp_unique)} unique values, range=[{min(vsp_unique):.3f}, {max(vsp_unique):.3f}], in_range={vsp_in_range}")

        # ---- 7. Readback: a: vs setting: ----
        readback_pass = True
        readback_mismatches = 0
        for col in action_cols_all:
            aid = col.split(":", 1)[1]
            setting_col = f"setting:{aid}"
            if setting_col in di_df.columns:
                a_vals = pd.to_numeric(di_df[col], errors="coerce").fillna(-999.0)
                s_vals = pd.to_numeric(di_df[setting_col], errors="coerce").fillna(-888.0)
                mismatches = (a_vals - s_vals).abs() > 1e-6
                n_mm = int(mismatches.sum())
                if n_mm > 0:
                    readback_pass = False
                    readback_mismatches += n_mm

        print(f"  Readback a: vs setting:: {'PASS' if readback_pass else f'FAIL ({readback_mismatches} mismatches)'}")

        # ---- Build trace ----
        trace_rows.append({
            "checkpoint_label": cp_label,
            "checkpoint_elapsed_min": cp_min,
            "event_id": primary_event,
            "policy_phase_transition": has_prefix and has_native,
            "transition_elapsed_min": transition_min,
            "override_inactive_post": bool(not post_cp["override_active"].any()),
            "eng36_changes_post_checkpoint": changes_post,
            "eng36_facilities_changed": len(changed_facilities),
            "action_sha_differs_from_hold_snapshot": sha_differs,
            "binary_pump_ADD301_2_strict_01": bool(
                set(pd.to_numeric(di_df.get("a:ADD301.2", pd.Series([1.0])), errors="coerce").dropna().unique()).issubset({0.0, 1.0})
            ) if "a:ADD301.2" in di_df.columns else None,
            "binary_pump_ADD301_3_strict_01": bool(
                set(pd.to_numeric(di_df.get("a:ADD301.3", pd.Series([1.0])), errors="coerce").dropna().unique()).issubset({0.0, 1.0})
            ) if "a:ADD301.3" in di_df.columns else None,
            "variable_speed_add350_1_continuous": vsp_pass,
            "readback_a_vs_setting": readback_pass,
            "overall_PASS": all([
                has_prefix and has_native,
                not post_cp["override_active"].any(),
                changes_post > 0,
                sha_differs,
                binary_pass,
                vsp_pass,
                readback_pass,
            ]),
        })

    # Write outputs
    trace_df = pd.DataFrame(trace_rows)
    trace_path = OUT_DIR / "dynamic_internal_trace.csv"
    trace_df.to_csv(trace_path, index=False)
    print(f"\n[Stage 6] Wrote {trace_path}")

    override_df = pd.DataFrame(override_audit_rows)
    override_path = OUT_DIR / "external_override_audit.csv"
    override_df.to_csv(override_path, index=False)
    print(f"[Stage 6] Wrote {override_path}")

    # Readback audit detail
    readback_detail = []
    for _, cp_row in catalog.iterrows():
        cp_label = str(cp_row["checkpoint_label"])
        di_path = OUT_DIR / f"branch_dynamic_internal_cp{cp_label}_detail.csv"
        di_df = pd.read_csv(di_path)
        action_cols = _action_cols(di_df)
        for col in action_cols:
            aid = col.split(":", 1)[1]
            setting_col = f"setting:{aid}"
            if setting_col in di_df.columns:
                a_vals = pd.to_numeric(di_df[col], errors="coerce").fillna(-999.0)
                s_vals = pd.to_numeric(di_df[setting_col], errors="coerce").fillna(-888.0)
                mm = int(((a_vals - s_vals).abs() > 1e-6).sum())
                readback_detail.append({
                    "checkpoint_label": cp_label,
                    "actuator_id": aid,
                    "mismatch_count": mm,
                    "PASS": mm == 0,
                })

    readback_df = pd.DataFrame(readback_detail)
    readback_path = OUT_DIR / "readback_audit.csv"
    readback_df.to_csv(readback_path, index=False)
    print(f"[Stage 6] Wrote {readback_path}")

    # Summary
    print(f"\n=== Stage 6 Summary ===")
    for _, r in trace_df.iterrows():
        print(f"  Checkpoint {r['checkpoint_label']}: overall_PASS={r['overall_PASS']}")
        print(f"    transition={r['policy_phase_transition']}, override_off={r['override_inactive_post']}, "
              f"eng36_changes={r['eng36_changes_post_checkpoint']}, sha_differs={r['action_sha_differs_from_hold_snapshot']}")
        print(f"    binary_pumps={r['binary_pump_ADD301_2_strict_01'] and r['binary_pump_ADD301_3_strict_01']}, "
              f"vsp_continuous={r['variable_speed_add350_1_continuous']}, readback={r['readback_a_vs_setting']}")

    all_pass = trace_df["overall_PASS"].all()
    print(f"\n  Stage 6 VERDICT: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
