from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CandidateSpec:
    label: str
    template: str
    scope: str
    delta: float
    hold_steps: int
    action: np.ndarray


def _as_bool(values) -> np.ndarray:
    return pd.Series(values).fillna(False).astype(bool).to_numpy()


def _as_float_array(values, n: int, default: float = np.nan) -> np.ndarray:
    try:
        arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    except Exception:
        arr = np.full(n, default, dtype=float)
    if len(arr) < n:
        arr = np.resize(arr, n)
    return arr[:n]


def _slug(text: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z_]+", "_", str(text).strip().lower()).strip("_")
    return slug or "unknown"


def _delta_tier(delta: float) -> str:
    d = abs(float(delta))
    if d <= 1e-9:
        return "identity"
    if d <= 0.080001:
        return "small"
    if d <= 0.160001:
        return "medium"
    return "large"


def _label(template: str, scope: str, delta: float, hold_steps: int) -> str:
    return f"{template}|scope={scope}|d={float(delta):.3f}|hold={int(hold_steps)}"


def parse_candidate_label(label: str) -> dict[str, object]:
    parts = str(label).split("|")
    template = parts[0].strip() or "unknown"
    out: dict[str, object] = {
        "template": template,
        "scope": "all",
        "delta": np.nan,
        "hold_steps": 1,
        "delta_tier": "small",
        "label": str(label),
    }
    for part in parts[1:]:
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k == "scope":
            out["scope"] = v or "all"
        elif k == "d":
            try:
                out["delta"] = float(v)
            except Exception:
                out["delta"] = np.nan
        elif k == "hold":
            try:
                out["hold_steps"] = max(1, int(float(v)))
            except Exception:
                out["hold_steps"] = 1
    out["delta_tier"] = _delta_tier(float(out["delta"]) if np.isfinite(float(out["delta"])) else 0.0)
    return out


def candidate_metadata_features(label: str) -> dict[str, float]:
    meta = parse_candidate_label(label)
    template = str(meta["template"])
    scope = str(meta["scope"])
    delta = float(meta["delta"]) if np.isfinite(float(meta["delta"])) else 0.0
    hold_steps = int(meta["hold_steps"])
    tier = str(meta["delta_tier"])
    return {
        f"feat_template_{_slug(template)}": 1.0,
        f"feat_candidate_scope_{_slug(scope)}": 1.0,
        f"feat_delta_tier_{_slug(tier)}": 1.0,
        "feat_residual_delta": float(delta),
        "feat_residual_delta_abs": float(abs(delta)),
        "feat_hold_steps": float(hold_steps),
    }


def _delta_values(max_delta: float, explicit: tuple[float, ...] | None = None) -> list[float]:
    if explicit:
        vals = [abs(float(v)) for v in explicit if abs(float(v)) > 1e-9]
    else:
        m = max(0.02, abs(float(max_delta)))
        seeds = [0.5 * m, m]
        if m > 0.080001:
            seeds.extend([0.08, 0.16])
        vals = seeds
    vals = sorted({round(min(abs(v), abs(float(max_delta))), 6) for v in vals if v > 1e-9})
    return [v for v in vals if v > 1e-9]


def _normalise_name_set(values) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        raw = values.split(",")
    else:
        raw = values
    return {str(v).strip() for v in raw if str(v).strip()}


def _normalise_scope_map(values) -> dict[str, set[str]]:
    if not values:
        return {}
    out: dict[str, set[str]] = {}
    for key, scopes in dict(values).items():
        name = str(key).strip()
        if not name:
            continue
        out[name] = _normalise_name_set(scopes)
    return out


def _template_scope_allowed(template: str, scope: str, allowed_scopes_by_template: dict[str, set[str]]) -> bool:
    if not allowed_scopes_by_template:
        return True
    allowed = allowed_scopes_by_template.get(template)
    if allowed is None:
        allowed = allowed_scopes_by_template.get("*")
    if not allowed:
        return True
    return scope in allowed


def _hot_local_mask(a: pd.DataFrame, state: np.ndarray | None, n: int, quantile: float = 0.80) -> np.ndarray:
    if state is None or len(state) == 0:
        return np.zeros(n, dtype=bool)
    s = np.asarray(state, dtype=float).reshape(-1)
    from_idx = pd.to_numeric(a.get("from_index", pd.Series(-1, index=a.index)), errors="coerce").fillna(-1).astype(int).to_numpy()
    to_idx = pd.to_numeric(a.get("to_index", pd.Series(-1, index=a.index)), errors="coerce").fillna(-1).astype(int).to_numpy()
    vals = []
    for i in range(n):
        local = []
        if 0 <= from_idx[i] < len(s):
            local.append(s[from_idx[i]])
        if 0 <= to_idx[i] < len(s):
            local.append(s[to_idx[i]])
        vals.append(float(np.nanmax(local)) if local else np.nan)
    vals_arr = np.asarray(vals, dtype=float)
    finite = vals_arr[np.isfinite(vals_arr)]
    if finite.size == 0:
        return np.zeros(n, dtype=bool)
    thr = float(np.nanquantile(finite, quantile))
    return np.isfinite(vals_arr) & (vals_arr >= thr)


def _scope_masks(
    base_mask: np.ndarray,
    a: pd.DataFrame,
    state: np.ndarray | None,
    n: int,
    priority_upstream_nodes: set[str] | None = None,
    priority_downstream_nodes: set[str] | None = None,
) -> list[tuple[str, np.ndarray]]:
    masks: list[tuple[str, np.ndarray]] = [("all", base_mask)]
    hot = _hot_local_mask(a, state, n)
    if np.any(base_mask & hot):
        masks.append(("hot_local", base_mask & hot))
    has_rule = _as_bool(a.get("has_internal_rule", pd.Series(False, index=a.index)))
    if np.any(base_mask & has_rule):
        masks.append(("native_rule", base_mask & has_rule))
    from_node = a.get("from_node", pd.Series("", index=a.index)).fillna("").astype(str).to_numpy()
    to_node = a.get("to_node", pd.Series("", index=a.index)).fillna("").astype(str).to_numpy()
    if priority_upstream_nodes:
        up = np.asarray(
            [(fn in priority_upstream_nodes) or (tn in priority_upstream_nodes) for fn, tn in zip(from_node, to_node)],
            dtype=bool,
        )
        if np.any(base_mask & up):
            masks.append(("priority_upstream", base_mask & up))
    if priority_downstream_nodes:
        down = np.asarray(
            [(fn in priority_downstream_nodes) or (tn in priority_downstream_nodes) for fn, tn in zip(from_node, to_node)],
            dtype=bool,
        )
        if np.any(base_mask & down):
            masks.append(("priority_downstream", base_mask & down))
    if priority_upstream_nodes or priority_downstream_nodes:
        corridor_nodes = set(priority_upstream_nodes or set()) | set(priority_downstream_nodes or set())
        corridor = np.asarray(
            [(fn in corridor_nodes) or (tn in corridor_nodes) for fn, tn in zip(from_node, to_node)],
            dtype=bool,
        )
        if np.any(base_mask & corridor):
            masks.append(("priority_corridor", base_mask & corridor))
    return [(name, mask) for name, mask in masks if np.any(mask)]


def _phase_templates(phase: str) -> list[tuple[str, str, float]]:
    """Return (template, mask_name, signed_multiplier)."""
    if phase == "recession":
        return [
            ("storage_outlet_release", "storage_outlet", +1.0),
            ("storage_inlet_open", "storage_inlet", +1.0),
            ("storage_all_release", "storage_nonpump", +1.0),
            ("pump_boost", "pump", +1.0),
            ("release_plus_pump_boost", "storage_or_pump", +1.0),
        ]
    if phase == "pre_peak":
        return [
            ("pump_pre_emptying", "pump", +0.5),
            ("storage_outlet_release", "storage_outlet", +0.5),
            ("storage_inlet_open", "storage_inlet", +0.5),
            ("storage_inlet_restrict", "storage_inlet", -1.0),
            ("pump_throttle", "pump", -0.5),
            ("storage_retain_pump_throttle", "storage_or_pump", -0.5),
        ]
    return [
        ("storage_inlet_restrict", "storage_inlet", -1.0),
        ("storage_outlet_retain", "storage_outlet", -1.0),
        ("storage_all_retain", "storage_nonpump", -1.0),
        ("pump_throttle", "pump", -0.5),
        ("storage_retain_pump_throttle", "storage_or_pump", -0.75),
        ("inlet_restrict_pump_throttle", "inlet_or_pump", -0.75),
    ]


def generate_candidate_specs(
    nominal: np.ndarray,
    actuators: pd.DataFrame,
    phase: str,
    max_delta: float = 0.08,
    include_nominal: bool = False,
    state: np.ndarray | None = None,
    delta_values: tuple[float, ...] | None = None,
    hold_steps: tuple[int, ...] = (1, 2, 3),
    max_candidates: int = 64,
    priority_upstream_nodes: set[str] | None = None,
    priority_downstream_nodes: set[str] | None = None,
    allowed_templates: set[str] | list[str] | tuple[str, ...] | str | None = None,
    blocked_templates: set[str] | list[str] | tuple[str, ...] | str | None = None,
    allowed_scopes_by_template: dict[str, set[str] | list[str] | tuple[str, ...] | str] | None = None,
) -> list[CandidateSpec]:
    nominal = np.asarray(nominal, dtype=np.float32).reshape(-1)
    n = len(nominal)
    if n == 0:
        return [CandidateSpec("nominal", "nominal", "identity", 0.0, 1, nominal)] if include_nominal else []
    a = actuators.reset_index(drop=True).iloc[:n].copy()
    typ = a.get("link_type", pd.Series("", index=a.index)).fillna("").astype(str).to_numpy()
    role = a.get("storage_control_type", pd.Series("", index=a.index)).fillna("").astype(str).to_numpy()
    near_storage = _as_bool(a.get("near_storage", pd.Series(False, index=a.index)))
    masks = {
        "pump": typ == "pump",
        "storage_inlet": role == "storage_inlet",
        "storage_outlet": role == "storage_outlet",
        "storage_nonpump": near_storage & (typ != "pump"),
        "storage_or_pump": near_storage | (typ == "pump"),
        "inlet_or_pump": (role == "storage_inlet") | (typ == "pump"),
    }
    deltas = _delta_values(max_delta, delta_values)
    holds = tuple(max(1, int(h)) for h in hold_steps if int(h) > 0) or (1,)
    allowed_template_set = _normalise_name_set(allowed_templates)
    blocked_template_set = _normalise_name_set(blocked_templates)
    allowed_scope_map = _normalise_scope_map(allowed_scopes_by_template)
    out: list[CandidateSpec] = []
    if include_nominal:
        out.append(CandidateSpec("nominal", "nominal", "identity", 0.0, 1, nominal.copy()))
    seen: set[tuple[str, tuple[float, ...]]] = set()
    for template, mask_name, sign in _phase_templates(str(phase)):
        if allowed_template_set and template not in allowed_template_set:
            continue
        if template in blocked_template_set:
            continue
        base_mask = masks.get(mask_name, np.zeros(n, dtype=bool))
        if not np.any(base_mask):
            continue
        for scope, scope_mask in _scope_masks(
            base_mask,
            a,
            state,
            n,
            priority_upstream_nodes=priority_upstream_nodes,
            priority_downstream_nodes=priority_downstream_nodes,
        ):
            if not _template_scope_allowed(template, scope, allowed_scope_map):
                continue
            for d_abs in deltas:
                signed_delta = float(sign) * float(d_abs)
                if abs(signed_delta) <= 1e-9:
                    continue
                u = nominal.copy()
                u[scope_mask] = np.clip(u[scope_mask] + signed_delta, 0.0, 1.0)
                if np.nanmax(np.abs(u - nominal)) <= 1e-6:
                    continue
                for h in holds:
                    label = _label(template, scope, signed_delta, h)
                    key = (label, tuple(np.round(u, 3)))
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(CandidateSpec(label, template, scope, signed_delta, h, u.astype(np.float32)))
                    if max_candidates > 0 and len(out) >= max_candidates:
                        return out
    return out


def generate_labeled_candidates(
    nominal: np.ndarray,
    actuators: pd.DataFrame,
    phase: str,
    max_delta: float = 0.08,
    include_nominal: bool = False,
    state: np.ndarray | None = None,
    delta_values: tuple[float, ...] | None = None,
    hold_steps: tuple[int, ...] = (1, 2, 3),
    max_candidates: int = 64,
    priority_upstream_nodes: set[str] | None = None,
    priority_downstream_nodes: set[str] | None = None,
    allowed_templates: set[str] | list[str] | tuple[str, ...] | str | None = None,
    blocked_templates: set[str] | list[str] | tuple[str, ...] | str | None = None,
    allowed_scopes_by_template: dict[str, set[str] | list[str] | tuple[str, ...] | str] | None = None,
) -> list[tuple[str, np.ndarray]]:
    return [
        (spec.label, spec.action)
        for spec in generate_candidate_specs(
            nominal,
            actuators,
            phase,
            max_delta=max_delta,
            include_nominal=include_nominal,
            state=state,
            delta_values=delta_values,
            hold_steps=hold_steps,
            max_candidates=max_candidates,
            priority_upstream_nodes=priority_upstream_nodes,
            priority_downstream_nodes=priority_downstream_nodes,
            allowed_templates=allowed_templates,
            blocked_templates=blocked_templates,
            allowed_scopes_by_template=allowed_scopes_by_template,
        )
    ]


def generate_candidates(
    nominal: np.ndarray,
    actuators: pd.DataFrame,
    phase: str,
    max_delta: float = 0.08,
    group_count: int = 4,
    include_nominal: bool = False,
    state: np.ndarray | None = None,
    priority_upstream_nodes: set[str] | None = None,
    priority_downstream_nodes: set[str] | None = None,
    allowed_templates: set[str] | list[str] | tuple[str, ...] | str | None = None,
    blocked_templates: set[str] | list[str] | tuple[str, ...] | str | None = None,
    allowed_scopes_by_template: dict[str, set[str] | list[str] | tuple[str, ...] | str] | None = None,
) -> list[np.ndarray]:
    _ = group_count
    return [
        u
        for _, u in generate_labeled_candidates(
            nominal,
            actuators,
            phase,
            max_delta=max_delta,
            include_nominal=include_nominal,
            state=state,
            priority_upstream_nodes=priority_upstream_nodes,
            priority_downstream_nodes=priority_downstream_nodes,
            allowed_templates=allowed_templates,
            blocked_templates=blocked_templates,
            allowed_scopes_by_template=allowed_scopes_by_template,
        )
    ]
