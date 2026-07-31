from __future__ import annotations

import json
from pathlib import Path


TEMPORAL_FRAME_OFFSETS_MIN = [0, -10, -20, -30, -40, -50, -60]


NODE_STATE_FIELDS = [
    "reconstructed_depth",
    "observed_depth",
    "depth_source",
    "depth_quality",
    "filling_degree",
    "hydraulic_head",
    "depth_headroom",
    "rim_margin",
    "surcharge_margin",
    "flooding_rate",
    "node_type",
    "is_priority",
    "is_sentinel_candidate",
    "is_storage",
    "observation_mask",
    "uncertainty",
    "ood_score",
]


FACILITY_STATE_FIELDS = [
    "facility_id",
    "facility_type",
    "anchor_setting",
    "native_target_setting",
    "requested_setting",
    "projected_setting",
    "target_setting",
    "actual_current_setting",
    "previous_actual_setting",
    "setting_rate",
    "upstream_head",
    "downstream_head",
    "head_difference",
    "local_flow",
    "capacity_ratio",
    "flow_direction",
    "flow_trend",
    "residual_override_active",
    "override_ttl",
    "released_to_native",
    "data_quality",
    "ood",
]


PUMP_STATE_FIELDS = [
    "facility_id",
    "pump_control_mode",
    "pump_on",
    "speed_setting_target",
    "speed_setting_actual",
    "speed_setting_previous",
    "speed_change_rate",
    "pump_curve_id",
    "pump_curve_type",
    "upstream_head",
    "downstream_head",
    "head_difference",
    "flow",
    "inferred_load_capacity_ratio",
    "native_rule_target",
    "residual_override_target",
    "actual_executed_speed",
    "binary_target",
    "binary_actual",
    "previous_binary_state",
    "starts_stops",
    "on_duration",
    "off_duration",
    "minimum_on_remaining",
    "minimum_off_remaining",
    "dwell_remaining",
    "switch_requested",
    "switch_allowed",
    "blocking_reason",
    "engineering_limits_status",
]


STORAGE_STATE_FIELDS = [
    "current_volume",
    "full_volume",
    "filling_ratio",
    "remaining_capacity",
    "inlet_flow",
    "outlet_flow",
    "net_flow",
    "depth",
    "headroom",
    "terminal_risk_proxy",
    "data_source",
    "uncertainty",
]


def build_state_feature_contract(config_sha256: str | None, network_sha256: str | None) -> dict:
    return {
        "contract_name": "project6_v3_augmented_state_contract",
        "control_interval_min": 10,
        "history_window_min": 60,
        "temporal_frame_offsets_min": TEMPORAL_FRAME_OFFSETS_MIN,
        "strict_causality": {
            "decision_after_observation_required": True,
            "future_observations_forbidden": True,
            "future_interpolation_forbidden": True,
            "truth_future_state_for_gap_fill_forbidden": True,
            "forecast_valid_time_is_not_observation_time": True,
        },
        "gat_selection": {
            "selected_primary_gat": "sr0p15",
            "selection_decision_status": "user_confirmed",
            "selection_lock_status": "pending_manual_execution",
            "gat_robustness_status": "pending",
            "compatible_strict_required_for_formal": True,
        },
        "sentinel": {
            "candidate_nodes": ["MH0200770", "HS1355904"],
            "sentinel_contract_status": "human_resolution_required",
            "safety_pass_flag_allowed": False,
        },
        "node_state_fields": NODE_STATE_FIELDS,
        "facility_state_fields": FACILITY_STATE_FIELDS,
        "pump_state_fields": PUMP_STATE_FIELDS,
        "priority_summary_fields": [
            "priority_depth_max",
            "priority_depth_mean",
            "priority_filling_degree_max",
            "priority_flooding_rate_sum",
            "priority_active_node_count",
            "priority_trend",
            "priority_uncertainty",
        ],
        "storage_state_fields": STORAGE_STATE_FIELDS,
        "flow_feature_policy": {
            "missing_flow_is_not_zero": True,
            "future_truth_flow_forbidden": True,
            "availability_mask_required": True,
        },
        "provenance": {
            "config_sha256": config_sha256,
            "network_sha256": network_sha256,
        },
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
