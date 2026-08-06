"""Deterministic high-information candidate expansion for V4.2 RTC.

The previous authoritative Pareto audit showed that the existing average of
roughly six candidates per state cannot establish a useful oracle ceiling for
36 Engineering36 actuators.  This module expands the *search population* while
keeping the executable contract unchanged:

* H12 prediction horizon;
* only H3 is modified;
* ADD301.2 and ADD301.3 remain binary;
* the remaining managed assets stay within [0, 1];
* K is recorded for search/engineering diagnostics, not used as a hydraulic
  safety certificate;
* exact sequence deduplication is applied after construction.

It does not evaluate safety and does not call SWMM.  PFV admission remains the
responsibility of the authoritative selector/oracle.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping, Sequence

import numpy as np


BINARY_ACTUATOR_IDS = frozenset({"ADD301.2", "ADD301.3"})


@dataclass(frozen=True)
class CandidateExpansionConfig:
    magnitudes: tuple[float, ...] = (0.05, 0.10, 0.20)
    temporal_profiles: tuple[str, ...] = (
        "constant_h3",
        "early_pulse",
        "ramp_h3",
        "release_h3",
    )
    max_pair_assets: int = 12
    max_quad_assets: int = 12
    max_candidates: int = 384
    include_pairs: bool = True
    include_quads: bool = True

    def __post_init__(self) -> None:
        if not self.magnitudes or any(
            not np.isfinite(value) or value <= 0.0 for value in self.magnitudes
        ):
            raise ValueError("candidate magnitudes must be finite and positive")
        if self.max_pair_assets < 2 or self.max_quad_assets < 4:
            raise ValueError("pair/quad asset limits are too small")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be positive")


def _profile(name: str, delta: float, prefix: int) -> np.ndarray:
    if prefix < 1:
        raise ValueError("controllable prefix must be positive")
    if name == "constant_h3":
        return np.full(prefix, delta, dtype=np.float32)
    if name == "early_pulse":
        values = np.zeros(prefix, dtype=np.float32)
        values[0] = delta
        return values
    if name == "ramp_h3":
        return np.linspace(0.5 * delta, delta, prefix, dtype=np.float32)
    if name == "release_h3":
        values = np.full(prefix, delta, dtype=np.float32)
        values[-1] = 0.0
        return values
    raise ValueError(f"unsupported candidate temporal profile: {name}")


def _role_direction(role: str) -> float:
    text = str(role or "").lower()
    if "pump" in text or "outlet" in text:
        return 1.0
    if "inlet" in text or "orifice" in text or "weir" in text:
        return -1.0
    return -1.0


def _normalise_ranked_ids(
    actuator_ids: Sequence[str], ranked_actuator_ids: Iterable[str] | None
) -> list[str]:
    known = {str(value) for value in actuator_ids}
    result: list[str] = []
    for value in ranked_actuator_ids or ():
        text = str(value)
        if text in known and text not in result:
            result.append(text)
    for value in actuator_ids:
        text = str(value)
        if text not in result:
            result.append(text)
    return result


def _sequence_key(sequence: np.ndarray) -> tuple[tuple[int, ...], bytes]:
    rounded = np.round(np.asarray(sequence, dtype=np.float32), 6)
    return tuple(rounded.shape), rounded.tobytes()


def generate_targeted_candidate_sequences(
    *,
    current_action: Sequence[float] | np.ndarray,
    actuator_ids: Sequence[str],
    horizon_steps: int = 12,
    controllable_prefix_steps: int = 3,
    actuator_roles: Mapping[str, str] | None = None,
    ranked_actuator_ids: Iterable[str] | None = None,
    successful_action_templates: Sequence[Mapping[str, object]] | None = None,
    config: CandidateExpansionConfig | None = None,
) -> list[dict[str, object]]:
    """Build a diverse, deterministic H3 candidate population.

    ``successful_action_templates`` may contain dictionaries with
    ``actuator_ids``, ``deltas`` and optional ``profile`` fields.  These are
    local neighbours of previously successful authoritative actions and never
    bypass bounds or binary semantics.
    """
    cfg = config or CandidateExpansionConfig()
    ids = [str(value) for value in actuator_ids]
    base = np.asarray(current_action, dtype=np.float32).reshape(-1)
    if len(ids) != base.size:
        raise ValueError("current_action and actuator_ids must have equal length")
    if not np.all(np.isfinite(base)):
        raise ValueError("current_action must be finite")
    if np.any(base < -1.0e-7) or np.any(base > 1.0 + 1.0e-7):
        raise ValueError("current_action must be within [0, 1]")
    horizon = max(1, int(horizon_steps))
    prefix = min(max(1, int(controllable_prefix_steps)), horizon)
    roles = {str(key): str(value) for key, value in (actuator_roles or {}).items()}
    ranked = _normalise_ranked_ids(ids, ranked_actuator_ids)
    index = {aid: idx for idx, aid in enumerate(ids)}
    reference = np.repeat(base[None, :], horizon, axis=0).astype(np.float32)
    candidates: list[dict[str, object]] = []
    seen: set[tuple[tuple[int, ...], bytes]] = set()

    def append(
        *,
        label: str,
        changes: Mapping[str, np.ndarray | float],
        family: str,
        rationale: str,
    ) -> None:
        sequence = reference.copy()
        changed: list[str] = []
        for aid, values in changes.items():
            if aid not in index:
                continue
            idx = index[aid]
            if aid in BINARY_ACTUATOR_IDS:
                target = 0.0 if float(base[idx]) >= 0.5 else 1.0
                sequence[:prefix, idx] = target
            else:
                profile = np.asarray(values, dtype=np.float32).reshape(-1)
                if profile.size == 1:
                    profile = np.full(prefix, float(profile[0]), dtype=np.float32)
                if profile.size != prefix:
                    raise ValueError("candidate profile length must equal H3 prefix")
                sequence[:prefix, idx] = np.clip(
                    base[idx] + profile, 0.0, 1.0
                )
            if np.any(np.abs(sequence[:prefix, idx] - base[idx]) > 1.0e-7):
                changed.append(aid)
        if not changed and family != "hold":
            return
        key = _sequence_key(sequence)
        if key in seen:
            return
        seen.add(key)
        candidates.append(
            {
                "label": label,
                "sequence": sequence,
                "absolute_sequence": sequence,
                "reference_sequence": reference.copy(),
                "target_actuators": ",".join(changed),
                "physical_rationale": rationale,
                "candidate_family": family,
                "changed_facilities": len(changed),
                "executable_prefix_steps": prefix,
                "tail_is_current_readback": bool(
                    np.allclose(sequence[prefix:], reference[prefix:])
                ),
            }
        )

    append(
        label="hold_native",
        changes={},
        family="hold",
        rationale="Preserve current Engineering36 readback.",
    )

    # Global single-asset coverage at several amplitudes and H3 profiles.
    for aid in ranked:
        if aid in BINARY_ACTUATOR_IDS:
            append(
                label=f"expand_binary_toggle|actuator={aid}",
                changes={aid: 1.0},
                family="binary_single",
                rationale="Explicit binary pump transition over executable H3.",
            )
            continue
        for magnitude in cfg.magnitudes:
            for sign_name, sign in (("decrease", -1.0), ("increase", 1.0)):
                for profile_name in cfg.temporal_profiles:
                    append(
                        label=(
                            f"expand_single|actuator={aid}|direction={sign_name}"
                            f"|magnitude={magnitude:g}|profile={profile_name}"
                        ),
                        changes={aid: _profile(profile_name, sign * magnitude, prefix)},
                        family="continuous_single",
                        rationale=(
                            "Global single-facility neighbourhood candidate; "
                            "authoritative PFV admission is applied later."
                        ),
                    )

    # Pair candidates use deterministic cross-role and sign-diverse patterns.
    pair_ids = ranked[: min(len(ranked), cfg.max_pair_assets)]
    if cfg.include_pairs:
        for left, right in combinations(pair_ids, 2):
            left_sign = _role_direction(roles.get(left, ""))
            right_sign = _role_direction(roles.get(right, ""))
            for pattern_name, multipliers in (
                ("role", (left_sign, right_sign)),
                ("opposed", (left_sign, -right_sign)),
            ):
                magnitude = cfg.magnitudes[min(1, len(cfg.magnitudes) - 1)]
                append(
                    label=(
                        f"expand_pair|a={left}|b={right}|pattern={pattern_name}"
                        f"|magnitude={magnitude:g}"
                    ),
                    changes={
                        left: _profile("constant_h3", multipliers[0] * magnitude, prefix),
                        right: _profile("constant_h3", multipliers[1] * magnitude, prefix),
                    },
                    family="coordinated_pair",
                    rationale="Two-facility coordinated action around high-priority assets.",
                )

    # Quad windows prevent combinatorial explosion while covering coordinated
    # multi-facility directions that the old ~6 candidates/state missed.
    quad_ids = ranked[: min(len(ranked), cfg.max_quad_assets)]
    if cfg.include_quads and len(quad_ids) >= 4:
        magnitude = cfg.magnitudes[0]
        for start in range(0, len(quad_ids) - 3, 2):
            selected = quad_ids[start : start + 4]
            changes = {
                aid: _profile(
                    "release_h3",
                    _role_direction(roles.get(aid, "")) * magnitude,
                    prefix,
                )
                for aid in selected
            }
            append(
                label=f"expand_quad|start={start}|magnitude={magnitude:g}",
                changes=changes,
                family="coordinated_quad",
                rationale="Bounded four-facility retain/release coordination.",
            )

    # Neighbourhoods around authoritative positive controls.
    for template_idx, template in enumerate(successful_action_templates or ()):
        template_ids = [str(value) for value in template.get("actuator_ids", ())]
        deltas = [float(value) for value in template.get("deltas", ())]
        if len(template_ids) != len(deltas) or not template_ids:
            continue
        profile_name = str(template.get("profile", "constant_h3"))
        for scale in (0.75, 1.0, 1.25):
            append(
                label=f"expand_positive_neighbour|template={template_idx}|scale={scale:g}",
                changes={
                    aid: _profile(profile_name, delta * scale, prefix)
                    for aid, delta in zip(template_ids, deltas)
                },
                family="positive_control_neighbour",
                rationale="Local perturbation around an authoritative successful action.",
            )

    # Preserve family diversity under a bounded candidate budget.
    if len(candidates) <= cfg.max_candidates:
        return candidates
    family_order = (
        "hold",
        "binary_single",
        "positive_control_neighbour",
        "continuous_single",
        "coordinated_pair",
        "coordinated_quad",
    )
    by_family = {
        family: [item for item in candidates if item["candidate_family"] == family]
        for family in family_order
    }
    selected: list[dict[str, object]] = []
    cursor = {family: 0 for family in family_order}
    while len(selected) < cfg.max_candidates:
        progressed = False
        for family in family_order:
            position = cursor[family]
            items = by_family[family]
            if position >= len(items):
                continue
            selected.append(items[position])
            cursor[family] += 1
            progressed = True
            if len(selected) >= cfg.max_candidates:
                break
        if not progressed:
            break
    return selected
