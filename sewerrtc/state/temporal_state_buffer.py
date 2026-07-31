from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from .state_contract import TEMPORAL_FRAME_OFFSETS_MIN


@dataclass(frozen=True)
class TemporalFrameSpec:
    frame_index: int
    offset_min: int
    measurement_time: datetime
    source_time: datetime | None
    state_estimation_time: datetime
    decision_time: datetime
    data_age_min: float | None
    valid_before_decision: bool
    interpolation_method: str
    quality_flag: str


def build_temporal_frame_schedule(
    decision_time: datetime,
    available_source_times: list[datetime],
    max_age_min: float,
    interpolation_method: str = "causal_last_observation_carried_forward",
) -> list[TemporalFrameSpec]:
    frames: list[TemporalFrameSpec] = []
    sorted_times = sorted(t for t in available_source_times if t <= decision_time)
    for idx, offset in enumerate(TEMPORAL_FRAME_OFFSETS_MIN):
        measurement_time = decision_time + timedelta(minutes=offset)
        candidates = [t for t in sorted_times if t <= measurement_time]
        source_time = candidates[-1] if candidates else None
        data_age = (measurement_time - source_time).total_seconds() / 60.0 if source_time else None
        valid = source_time is not None and source_time <= decision_time and source_time <= measurement_time
        stale = data_age is None or data_age > max_age_min
        frames.append(
            TemporalFrameSpec(
                frame_index=idx,
                offset_min=offset,
                measurement_time=measurement_time,
                source_time=source_time,
                state_estimation_time=decision_time,
                decision_time=decision_time,
                data_age_min=data_age,
                valid_before_decision=valid,
                interpolation_method=interpolation_method,
                quality_flag="stale_or_missing" if stale else "usable",
            )
        )
    return frames


def assert_no_future_observation(frames: list[TemporalFrameSpec]) -> None:
    for frame in frames:
        if frame.source_time is not None and frame.source_time > frame.decision_time:
            raise ValueError(f"future observation entered state frame {frame.frame_index}")
        if frame.source_time is not None and frame.source_time > frame.measurement_time:
            raise ValueError(f"future interpolation entered state frame {frame.frame_index}")
