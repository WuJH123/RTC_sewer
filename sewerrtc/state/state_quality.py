from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StateQualitySummary:
    stale_frame_count: int
    missing_observation_count: int
    future_data_violation_count: int
    ood_score: float | None
    quality_flag: str


def summarize_state_quality(
    stale_frame_count: int,
    missing_observation_count: int,
    future_data_violation_count: int,
    ood_score: float | None,
    ood_threshold: float | None,
) -> StateQualitySummary:
    if future_data_violation_count:
        flag = "invalid_future_data"
    elif stale_frame_count or missing_observation_count:
        flag = "degraded"
    elif ood_threshold is not None and ood_score is not None and ood_score > ood_threshold:
        flag = "ood"
    else:
        flag = "usable"
    return StateQualitySummary(
        stale_frame_count=stale_frame_count,
        missing_observation_count=missing_observation_count,
        future_data_violation_count=future_data_violation_count,
        ood_score=ood_score,
        quality_flag=flag,
    )
