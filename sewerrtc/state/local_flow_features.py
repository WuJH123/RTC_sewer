from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FlowSource(str, Enum):
    OBSERVED = "observed_flow"
    SWMM_TRUTH_OFFLINE = "swmm_truth_flow_offline"
    RECONSTRUCTED = "reconstructed_flow"
    UNAVAILABLE = "unavailable_flow"


@dataclass(frozen=True)
class FlowFeature:
    facility_id: str
    link_id: str | None
    source: FlowSource
    value: float | None
    availability_mask: bool
    uncertainty: float | None
    uses_future_truth: bool = False


def unavailable_flow(facility_id: str, link_id: str | None = None) -> FlowFeature:
    return FlowFeature(
        facility_id=facility_id,
        link_id=link_id,
        source=FlowSource.UNAVAILABLE,
        value=None,
        availability_mask=False,
        uncertainty=None,
        uses_future_truth=False,
    )


def validate_flow_feature(feature: FlowFeature) -> None:
    if feature.uses_future_truth:
        raise ValueError(f"future true flow is forbidden for online feature: {feature.facility_id}")
    if feature.source == FlowSource.UNAVAILABLE and feature.value == 0:
        raise ValueError("unavailable flow must not be encoded as true zero flow")
    if feature.availability_mask is False and feature.value is not None:
        raise ValueError("flow value supplied while availability_mask is false")
