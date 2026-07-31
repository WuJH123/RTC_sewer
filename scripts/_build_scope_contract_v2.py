"""Parse [CONTROLS] section and create Scope Contract V2."""
from __future__ import annotations
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

INP_PATH = PROJECT_ROOT / "data" / "wuhan_v8_storage_retrofit.inp"
V1_CONTRACT = PROJECT_ROOT / "docs" / "contracts" / "PROJECT6_V4_CONTROL_SCOPE_CONTRACT.json"
OUT_PATH = PROJECT_ROOT / "docs" / "contracts" / "PROJECT6_V4_CONTROL_SCOPE_CONTRACT_V2.json"


def parse_control_links(inp_path: Path) -> list[str]:
    """Parse [CONTROLS] RULE section to find all controlled link IDs."""
    lines = inp_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    in_ctrl = False
    ctrl_links = set()
    for line in lines:
        s = line.strip()
        if s.upper() == "[CONTROLS]":
            in_ctrl = True
            continue
        if s.startswith("[") and in_ctrl:
            break
        if in_ctrl and s and not s.startswith(";"):
            parts = s.split()
            # THEN ORIFICE/LINK/PUMP/WEIR <id> SETTING = <val>
            if len(parts) >= 3 and parts[0].upper() in ("THEN", "AND", "PRIORITY"):
                if parts[1].upper() in ("LINK", "PUMP", "ORIFICE", "WEIR"):
                    ctrl_links.add(parts[2])
            # Also: THEN LINK <id> = <val> (older format)
            if len(parts) >= 4 and parts[0].upper() in ("THEN", "AND"):
                if parts[1].upper() in ("LINK", "PUMP", "ORIFICE", "WEIR"):
                    ctrl_links.add(parts[2])
    return sorted(ctrl_links)


def main() -> int:
    # Parse native control links
    native_ctrl_links = parse_control_links(INP_PATH)
    print(f"Parsed {len(native_ctrl_links)} native control links from [CONTROLS]")

    # Load V1 contract
    v1 = json.loads(V1_CONTRACT.read_text(encoding="utf-8"))

    # Build V2 contract
    v2 = {
        "contract_name": "PROJECT6_V4_CONTROL_SCOPE_CONTRACT_V2",
        "contract_version": "2.0",
        "created_for": "Gate 2.5-real-v3",
        "supersedes": "PROJECT6_V4_CONTROL_SCOPE_CONTRACT v1.0",
        "supersede_reason": "V2 prefix schedule only covered 36 Eng36 facilities. "
                           "179 native control links were missing, causing 6.1m hydraulic divergence.",

        "network": v1.get("network", {}),

        "native_rules": v1.get("native_rules", {}),

        "engineering36_ids": v1.get("engineering36_ids", []),
        "engineering36_overlap_with_native_rules": v1.get("engineering36_overlap_with_native_rules", []),

        # NEW in V2: complete list of native control links
        "native_control_links": native_ctrl_links,
        "native_control_links_count": len(native_ctrl_links),

        # V2 prefix scope: ALL links that must be replayed during prefix
        "prefix_scope": {
            "description": "All links that must be included in the authoritative prefix schedule. "
                          "During prefix replay, ALL of these links must have their baseline "
                          "current_setting written to target_setting by the external override.",
            "eng36_actuator_facilities": v1.get("engineering36_ids", []),
            "eng36_count": len(v1.get("engineering36_ids", [])),
            "native_rule_facilities_82": v1.get("engineering36_overlap_with_native_rules", []) +
                [f for f in native_ctrl_links if f not in v1.get("engineering36_overlap_with_native_rules", [])],
            "native_control_links_179": native_ctrl_links,
            "total_prefix_links": len(set(v1.get("engineering36_ids", [])) | set(native_ctrl_links)),
            "note": "prefix_links = Eng36 actuators (36) UNION native control links (179) = 215 unique links. "
                    "Some Eng36 facilities may overlap with native control links.",
        },

        # Branch definitions
        "branches": {
            "dynamic_internal": {
                "description": "Prefix replay of ALL 215 links up to checkpoint, then native [CONTROLS] govern.",
                "inp_variant": "with_controls",
                "prefix_action": "External override writes ALL 215 link settings from baseline",
                "post_checkpoint": "Stop ALL external overrides. Native [CONTROLS] govern all facilities.",
                "controlled_facilities": "All 215 prefix links during prefix; native rules after checkpoint",
                "external_override_count_after_checkpoint": 0,
            },
            "no_control": {
                "description": "Prefix replay of ALL 215 links, then all settings = 1.0 (fully open).",
                "inp_variant": "no_controls (strip_controls=True)",
                "prefix_action": "External override writes ALL 215 link settings from baseline",
                "post_checkpoint": "All 215 links set to 1.0 (fully open)",
                "controlled_facilities": "All 215 prefix links during prefix; all 1.0 after checkpoint",
                "note": "With [CONTROLS] stripped, native rules don't exist. External override must cover "
                        "ALL 215 links to match baseline during prefix.",
            },
            "hold_all82_internal_snapshot": {
                "description": "Prefix replay of ALL 215 links, then freeze at checkpoint current_setting.",
                "inp_variant": "no_controls (strip_controls=True)",
                "prefix_action": "External override writes ALL 215 link settings from baseline",
                "post_checkpoint": "All 215 links frozen at checkpoint current_setting (constant)",
                "freeze_scope": "All 215 prefix links",
                "preferred_for_gate": True,
                "note": "Gate 2.5 diagnostic uses this variant to verify Dynamic Internal difference "
                        "relative to full native action freeze.",
            },
            "hold_previous": {
                "description": "Prefix replay of ALL 215 links, then freeze at checkpoint-5min setting.",
                "inp_variant": "no_controls (strip_controls=True)",
                "prefix_action": "External override writes ALL 215 link settings from baseline",
                "post_checkpoint": "All 215 links frozen at checkpoint-5min current_setting (constant)",
                "freeze_scope": "All 215 prefix links",
            },
            "causal_low": {
                "description": "Causal intervention: single actuator at setting=0.",
                "inp_variant": "no_controls",
                "prefix_action": "External override writes ALL 215 link settings from baseline",
                "post_checkpoint": "Selected actuator=0.0, all others at baseline values",
            },
            "causal_high": {
                "description": "Causal intervention: single actuator at setting=1.",
                "inp_variant": "no_controls",
                "prefix_action": "External override writes ALL 215 link settings from baseline",
                "post_checkpoint": "Selected actuator=1.0, all others at baseline values",
            },
        },

        "shared_prefix_contract": {
            "description": "All branches must share identical hydraulic state at the last prefix timestep.",
            "prefix_links": sorted(set(v1.get("engineering36_ids", [])) | set(native_ctrl_links)),
            "prefix_link_count": len(set(v1.get("engineering36_ids", [])) | set(native_ctrl_links)),
            "verification": {
                "method": "Compare all node depths, heads, flooding, storage volumes, link flows, "
                          "and link settings at the last elapsed_min < checkpoint row.",
                "tolerance": 1e-6,
                "must_include": ["dynamic_internal", "no_control", "hold_snapshot", "hold_previous"],
                "pass_condition": "max_abs_diff <= tolerance for ALL quantities across ALL 4 branches",
            },
        },

        "recovery_contract": {
            "minimum_tail_min": 180,
            "max_tail_min": 720,
            "stable_recovery_duration_min": 60,
            "recovery_is_blocking": True,
            "censoring_rule": "If max_tail reached without recovery: recovery_censored=true, "
                             "full_event_eligible=false, Gate BLOCKED",
        },

        "scope_conflict_resolution": v1.get("scope_conflict_resolution", "RESOLVED"),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(v2, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {OUT_PATH}")
    print(f"  Native control links: {len(native_ctrl_links)}")
    prefix_total = len(set(v1.get("engineering36_ids", [])) | set(native_ctrl_links))
    print(f"  Total prefix links: {prefix_total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
