from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


@dataclass(frozen=True)
class EventContext:
    event_id: str
    return_period: int | None
    duration_min: int | None
    pattern: str


@dataclass(frozen=True)
class BoundaryDecision:
    allowed: bool
    reason: str = ""
    safe_prob_extra: float = 0.0
    peak_prob_extra: float = 0.0
    cautious_event: bool = False


def parse_event_context(event_id: str) -> EventContext:
    text = str(event_id or "")
    rp = None
    dur = None
    m = re.search(r"(?:^|_)T(\d+)(?:_|$)", text, flags=re.IGNORECASE)
    if m:
        rp = int(m.group(1))
    m = re.search(r"(?:^|_)D(\d+)(?:_|$)", text, flags=re.IGNORECASE)
    if m:
        dur = int(m.group(1))
    pattern = ""
    if "_D" in text:
        parts = text.split("_")
        if len(parts) >= 3:
            pattern = "_".join(parts[2:]).lower()
    return EventContext(text, rp, dur, pattern)


def should_block_low_risk_takeover(
    nominal_pfv_reference: float,
    threshold: float,
    use_native_shield: bool,
) -> tuple[bool, str]:
    if not use_native_shield:
        return False, ""
    try:
        pfv = float(nominal_pfv_reference)
    except Exception:
        pfv = 0.0
    if pfv < float(threshold):
        return True, "low_internal_pfv"
    return False, ""


def should_cancel_held_action_in_low_risk(
    low_risk_blocked: bool,
    current_priority_risk_class: str,
) -> tuple[bool, str]:
    """Cancel a previously accepted intervention once risk has receded.

    A multi-step hold action is useful for actuator smoothness, but it must not
    bypass the risk-stratified safety shield. If the event is classified as
    low-risk by the native baseline, or the current priority-zone state has
    dropped back to low risk, the controller should return to native/nominal
    control rather than continue a stale enhancement action.
    """
    if bool(low_risk_blocked):
        return True, "low_internal_pfv_cancel_held_action"
    if str(current_priority_risk_class or "").lower() == "low_risk_state":
        return True, "current_priority_low_risk_cancel_held_action"
    return False, ""


def is_release_or_boost(candidate_label: str) -> bool:
    text = str(candidate_label).lower()
    return "release" in text or "boost" in text or "empty" in text


def is_cautious_event(
    event_id: str,
    strict_guard_return_period_max: int = 15,
    strict_guard_patterns: Iterable[str] = ("chicago_late", "block", "double_peak"),
) -> bool:
    ctx = parse_event_context(event_id)
    if ctx.return_period is not None and ctx.return_period <= int(strict_guard_return_period_max):
        return True
    patterns = {str(p).strip().lower() for p in strict_guard_patterns if str(p).strip()}
    return bool(ctx.pattern and ctx.pattern in patterns)


def candidate_boundary_decision(
    candidate_label: str,
    event_id: str,
    phase: str,
    nominal_pfv_reference: float,
    priority_depth_max: float,
    rainfall_mm_h: float,
    release_recession_pfv_min: float = 500.0,
    release_recession_priority_depth_min: float = 1.0,
    strict_guard_return_period_max: int = 15,
    strict_guard_patterns: Iterable[str] = ("chicago_late", "block", "double_peak"),
    strict_guard_prob_extra: float = 0.10,
) -> BoundaryDecision:
    cautious = is_cautious_event(event_id, strict_guard_return_period_max, strict_guard_patterns)
    label = str(candidate_label or "")
    ph = str(phase or "").lower()
    is_release = is_release_or_boost(label)
    pfv = float(nominal_pfv_reference or 0.0)
    depth = float(priority_depth_max or 0.0)

    if is_release and ph == "recession":
        if pfv < float(release_recession_pfv_min):
            return BoundaryDecision(False, "release_recession_low_internal_pfv", cautious_event=cautious)
        if cautious and depth < float(release_recession_priority_depth_min):
            return BoundaryDecision(False, "release_recession_cautious_event_low_priority_depth", cautious_event=True)

    if cautious:
        extra = max(0.0, float(strict_guard_prob_extra))
        return BoundaryDecision(True, "", safe_prob_extra=extra, peak_prob_extra=extra, cautious_event=True)
    return BoundaryDecision(True, "", cautious_event=False)
