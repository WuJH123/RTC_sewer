from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ELIGIBLE = "eligible"
INELIGIBLE = "ineligible"
INCOMPLETE_METADATA = "incomplete_metadata"
REQUIRES_NEW_TRAJECTORY = "requires_new_trajectory"


@dataclass(frozen=True)
class HoldoutEligibility:
    status: str
    exclusion_reason: str
    match_type: str = ""
    matching_contaminated_event: str = ""


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def classify_holdout_candidate(candidate: dict[str, Any], contaminated: dict[str, set[str]]) -> HoldoutEligibility:
    event_id = _norm(candidate.get("event_id") or candidate.get("canonical_event_id"))
    family = _norm(candidate.get("storm_family_id"))
    rain_hash = _norm(candidate.get("rainfall_series_sha256") or candidate.get("rainfall_file_sha256"))
    trajectory_hash = _norm(candidate.get("trajectory_sha256"))

    if not event_id:
        return HoldoutEligibility(INCOMPLETE_METADATA, "missing_event_id")
    if event_id in contaminated.get("event_ids", set()):
        return HoldoutEligibility(INELIGIBLE, "exact_event_overlap", "event_id", event_id)
    if rain_hash and rain_hash in contaminated.get("rainfall_hashes", set()):
        return HoldoutEligibility(INELIGIBLE, "rainfall_hash_overlap", "rainfall_hash", rain_hash)
    if trajectory_hash and trajectory_hash in contaminated.get("trajectory_hashes", set()):
        return HoldoutEligibility(INELIGIBLE, "trajectory_hash_overlap", "trajectory_hash", trajectory_hash)
    if family and family in contaminated.get("storm_families", set()):
        return HoldoutEligibility(INELIGIBLE, "storm_family_overlap", "storm_family", family)
    near_duplicate = _norm(candidate.get("rainfall_near_duplicate_match_type"))
    if near_duplicate in {"intensity_scale", "time_shift", "renamed_same_series", "truncated_or_padded_series"}:
        matched = _norm(candidate.get("matching_contaminated_rainfall") or candidate.get("matching_contaminated_event"))
        return HoldoutEligibility(INELIGIBLE, f"rainfall_near_duplicate:{near_duplicate}", near_duplicate, matched)

    has_truth = str(candidate.get("full_node_truth_available", "")).lower() == "true"
    has_sr0p15_sensor = str(candidate.get("sr0p15_sensor_available", "")).lower() == "true"
    has_timestamps = str(candidate.get("timestamps_available", "")).lower() == "true"
    has_history = str(candidate.get("has_60min_history", "")).lower() == "true"
    if has_truth and has_sr0p15_sensor and has_timestamps and has_history:
        return HoldoutEligibility(ELIGIBLE, "")

    if _norm(candidate.get("rainfall_path")) or rain_hash:
        return HoldoutEligibility(REQUIRES_NEW_TRAJECTORY, "independent_rainfall_without_complete_truth")
    return HoldoutEligibility(INCOMPLETE_METADATA, "missing_rainfall_or_truth_metadata")
