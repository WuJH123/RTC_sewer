from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd


@dataclass
class PolicyContext:
    elapsed_min: float
    duration_min: int
    rainfall_mm_h: float
    phase: str
    previous_action: np.ndarray
    node_depths: Dict[str, float] | None = None
    node_max_depths: Dict[str, float] | None = None


def _first_nonempty(*values: object) -> str:
    for value in values:
        text = str(value or "").strip()
        if text and text.lower() != "nan":
            return text
    return ""


def _reference_node_for_row(row: "pd.Series", role: str) -> str:
    """Resolve the control/reference node id for an actuator.

    The Project6 actuator tables do not carry the legacy ``from_node``/
    ``to_node`` column names that earlier code assumed. Topology is instead
    described by ``storage_node`` / ``upstream_node`` / ``downstream_node``
    (retrofit asset table) and, at runtime, by ``from_node`` / ``to_node``
    attached from the INP link connectivity. Resolve against whatever is
    available so filling-degree policies see a real depth signal instead of a
    silent zero. An explicit ``efd_reference_node`` always wins.
    """
    explicit = _first_nonempty(row.get("efd_reference_node", ""))
    if explicit:
        return explicit
    storage_node = _first_nonempty(row.get("storage_node", ""))
    upstream = _first_nonempty(row.get("upstream_node", ""), row.get("from_node", ""))
    downstream = _first_nonempty(row.get("downstream_node", ""), row.get("to_node", ""))
    if role == "storage_inlet":
        # An inlet fills the downstream storage tank.
        return _first_nonempty(storage_node, downstream, upstream)
    if role == "storage_outlet":
        # An outlet drains the upstream storage tank.
        return _first_nonempty(storage_node, upstream, downstream)
    # Regulators / pumps / plain links: the node whose level they control most
    # directly is the upstream node.
    return _first_nonempty(upstream, storage_node, downstream)


def attach_reference_nodes(actuators: pd.DataFrame, inp_path: "str | Path") -> pd.DataFrame:
    """Attach ``from_node``/``to_node`` columns from the INP link topology.

    Only the few retrofit storage assets carry topology in the CSV contract, so
    the remaining regulators and pumps would otherwise have no resolvable
    reference node. Reading the INP link connectivity gives every managed
    facility a real control node without clobbering explicit asset-table values.
    """
    if actuators.empty or "actuator_id" not in actuators:
        return actuators
    try:
        from sewerrtc.io.inp_parser import parse_links, read_sections

        links = parse_links(read_sections(inp_path))
    except Exception:
        return actuators
    if links.empty or "link_id" not in links:
        return actuators
    topo = links.drop_duplicates("link_id").set_index("link_id")
    out = actuators.copy()
    ids = out["actuator_id"].astype(str)
    from_existing = out["from_node"].astype(str) if "from_node" in out else pd.Series("", index=out.index)
    to_existing = out["to_node"].astype(str) if "to_node" in out else pd.Series("", index=out.index)
    out["from_node"] = [
        _first_nonempty(str(topo.at[a, "from_node"]) if a in topo.index else "", from_existing.iloc[i])
        for i, a in enumerate(ids)
    ]
    out["to_node"] = [
        _first_nonempty(str(topo.at[a, "to_node"]) if a in topo.index else "", to_existing.iloc[i])
        for i, a in enumerate(ids)
    ]
    return out


def _safe_depths_for_actuators(actuators: pd.DataFrame, ctx: PolicyContext) -> np.ndarray:
    """Return local reference node depths for each actuator."""
    depths = ctx.node_depths or {}
    vals: list[float] = []
    role = actuators.get("storage_control_type", pd.Series("", index=actuators.index)).fillna("").astype(str)
    for i, row in actuators.reset_index(drop=True).iterrows():
        r = str(role.iloc[i]) if i < len(role) else ""
        nid = _reference_node_for_row(row, r)
        vals.append(float(depths.get(nid, 0.0)))
    return np.asarray(vals, dtype=np.float32)


def _efd_reference_fill(actuators: pd.DataFrame, ctx: PolicyContext) -> tuple[np.ndarray, np.ndarray]:
    """Compute Wuhan-specific EFD reference depth and filling degree.

    Astlingen's EFD rules compare named storage tank filling degrees. For the
    Wuhan INP we reconstruct the same engineering idea from the local actuator
    table: storage inlets use their downstream storage node, storage outlets
    use their upstream storage node, and other controllable links use the
    upstream node. Filling degree is depth / max_depth whenever INP metadata is
    available; a depth-percentile fallback is kept only for legacy tables.
    """
    depths = ctx.node_depths or {}
    max_depths = ctx.node_max_depths or {}
    rows = actuators.reset_index(drop=True)
    ref_nodes: list[str] = []
    ref_max: list[float] = []
    role = rows.get("storage_control_type", pd.Series("", index=rows.index)).fillna("").astype(str)
    for i, row in rows.iterrows():
        r = str(role.iloc[i]) if i < len(role) else ""
        nid = _reference_node_for_row(row, r)
        ref_nodes.append(nid)

        md = np.nan
        for key in ("efd_reference_max_depth", "storage_node_max_depth"):
            try:
                md = float(row.get(key, np.nan))
            except Exception:
                md = np.nan
            if np.isfinite(md) and md > 1e-6:
                break
        if not (np.isfinite(md) and md > 1e-6):
            try:
                md = float(max_depths.get(nid, np.nan))
            except Exception:
                md = np.nan
        ref_max.append(md if np.isfinite(md) and md > 1e-6 else np.nan)

    d = np.asarray([float(depths.get(nid, 0.0)) for nid in ref_nodes], dtype=np.float32)
    md_arr = np.asarray(ref_max, dtype=np.float32)
    fill = np.full_like(d, np.nan, dtype=np.float32)
    valid = np.isfinite(md_arr) & (md_arr > 1e-6)
    fill[valid] = d[valid] / md_arr[valid]
    if np.any(~np.isfinite(fill)):
        scale = max(1e-6, float(np.nanpercentile(d, 95))) if d.size else 1.0
        fill[~np.isfinite(fill)] = d[~np.isfinite(fill)] / scale
    return d.astype(np.float32), np.clip(fill, 0.0, 1.5).astype(np.float32)


def phase_from_time(elapsed_min: float, duration_min: int) -> str:
    if elapsed_min < 0.35 * duration_min:
        return "pre_peak"
    if elapsed_min <= duration_min:
        return "peak"
    return "recession"


class GenericActionPolicy:
    def __init__(self, policy_id: str, actuators: pd.DataFrame, seed: int = 2026):
        self.policy_id = policy_id
        self.actuators = actuators.reset_index(drop=True)
        self.rng = np.random.default_rng(seed)
        self.base = np.ones(len(self.actuators), dtype=np.float32)
        if len(self.base) == 0:
            self.base = np.zeros(0, dtype=np.float32)
        self.fixed_action = self._fixed_action_for_policy(policy_id)
        self.schedule = self._load_schedule_for_policy(policy_id)
        self.enforces_targets = self.fixed_action is not None or self.schedule is not None

    def _fixed_action_for_policy(self, policy_id: str) -> np.ndarray | None:
        n = len(self.actuators)
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        candidates = [
            f"{policy_id}_setting",
            f"{policy_id}_action",
            f"baseline_{policy_id}_setting",
        ]
        for col in candidates:
            if col not in self.actuators:
                continue
            vals = pd.to_numeric(self.actuators[col], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(vals).any():
                fallback = np.ones(n, dtype=float)
                vals = np.where(np.isfinite(vals), vals, fallback)
                return np.clip(vals, 0.0, 1.0).astype(np.float32)
        return None

    def _load_schedule_for_policy(self, policy_id: str) -> pd.DataFrame | None:
        csv_col = f"{policy_id}_schedule_csv"
        if csv_col not in self.actuators:
            return None
        raw = self.actuators[csv_col].dropna().astype(str)
        raw = raw[raw.str.strip().ne("")]
        if raw.empty:
            return None
        path = Path(raw.iloc[0])
        if not path.exists():
            return None
        try:
            sched = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            return None
        time_col = f"{policy_id}_schedule_time_column"
        unit_col = f"{policy_id}_schedule_time_unit"
        time_name = (
            str(self.actuators[time_col].dropna().astype(str).iloc[0])
            if time_col in self.actuators and self.actuators[time_col].dropna().size
            else "simtime (hr)"
        )
        time_unit = (
            str(self.actuators[unit_col].dropna().astype(str).iloc[0]).lower()
            if unit_col in self.actuators and self.actuators[unit_col].dropna().size
            else "hr"
        )
        if time_name not in sched:
            return None
        out = pd.DataFrame()
        out["elapsed_min"] = pd.to_numeric(sched[time_name], errors="coerce")
        if time_unit in {"hr", "hour", "hours"}:
            out["elapsed_min"] *= 60.0
        for aid in self.actuators["actuator_id"].astype(str):
            if aid not in sched:
                out[aid] = np.nan
                continue
            values = sched[aid].astype(str).str.strip().replace({"ON": "1.0", "OFF": "0.0", "on": "1.0", "off": "0.0"})
            out[aid] = pd.to_numeric(values, errors="coerce")
        out = out.dropna(subset=["elapsed_min"]).sort_values("elapsed_min").reset_index(drop=True)
        return out if not out.empty else None

    def _scheduled_action(self, elapsed_min: float) -> np.ndarray:
        if self.schedule is None or self.schedule.empty:
            return self.base.copy()
        times = self.schedule["elapsed_min"].to_numpy(dtype=float)
        j = int(np.searchsorted(times, float(elapsed_min), side="right") - 1)
        j = int(np.clip(j, 0, len(self.schedule) - 1))
        vals = []
        for aid in self.actuators["actuator_id"].astype(str):
            vals.append(float(self.schedule.iloc[j].get(aid, np.nan)))
        arr = np.asarray(vals, dtype=float)
        fallback = self.fixed_action if self.fixed_action is not None else self.base
        arr = np.where(np.isfinite(arr), arr, fallback)
        return np.clip(arr, 0.0, 1.0).astype(np.float32)

    def action(self, ctx: PolicyContext) -> np.ndarray:
        """Return a policy action while leaving non-deployed assets passive."""
        action = self._unmasked_action(ctx)
        if "control_enabled" not in self.actuators:
            return action
        enabled = self.actuators["control_enabled"].fillna(True).astype(bool).to_numpy()
        if enabled.size != action.size or enabled.all():
            return action
        default = pd.to_numeric(
            self.actuators.get("fail_safe_setting", pd.Series(1.0, index=self.actuators.index)),
            errors="coerce",
        ).fillna(1.0).to_numpy(dtype=np.float32)
        out = np.asarray(action, dtype=np.float32).copy()
        out[~enabled] = np.clip(default[~enabled], 0.0, 1.0)
        return out

    def _unmasked_action(self, ctx: PolicyContext) -> np.ndarray:
        n = len(self.actuators)
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        typ = self.actuators["link_type"].to_numpy(str)
        storage = self.actuators.get("near_storage", pd.Series(False, index=self.actuators.index)).to_numpy(bool)
        storage_role = self.actuators.get("storage_control_type", pd.Series("", index=self.actuators.index)).fillna("").to_numpy(str)
        storage_inlet = storage_role == "storage_inlet"
        storage_outlet = storage_role == "storage_outlet"
        non_pump = typ != "pump"
        if len(ctx.previous_action) == n:
            u = np.clip(np.asarray(ctx.previous_action, dtype=np.float32), 0.0, 1.0).copy()
        else:
            u = np.ones(n, dtype=np.float32)
        pid = self.policy_id
        if self.schedule is not None:
            return self._scheduled_action(ctx.elapsed_min)
        if self.fixed_action is not None:
            return self.fixed_action.copy()
        if pid in ("no_control", "internal_rules"):
            return ctx.previous_action.copy() if len(ctx.previous_action) == n else u
        if pid == "all_open":
            return u
        if pid == "all_closed_safe":
            u[:] = 0.15
            u[typ == "pump"] = 0.0
            return u
        if pid == "random_uniform":
            return self.rng.uniform(0.0, 1.0, n).astype(np.float32)
        if pid == "sobol_like":
            x = (np.sin(np.arange(n) * 12.9898 + ctx.elapsed_min * 0.071) * 43758.5453) % 1
            return x.astype(np.float32)
        if pid in ("storage_retain", "peak_storage_retain", "pre_peak_storage_retain"):
            u[storage & (typ != "pump")] = 0.15 if ctx.phase != "recession" else 0.55
            return u
        if pid == "storage_release":
            u[storage & (typ != "pump")] = 1.0 if ctx.phase == "recession" else 0.45
            return u
        if pid in ("pump_emptying", "recession_pump_emptying"):
            u[typ == "pump"] = 1.0 if ctx.phase == "recession" else 0.35
            return u
        if pid == "pump_throttle":
            u[typ == "pump"] = 0.25 if ctx.phase == "peak" else 0.85
            return u
        if pid == "one_at_a_time_pulse":
            u[:] = 1.0
            j = int(ctx.elapsed_min // 10) % n
            u[j] = 0.1 if int(ctx.elapsed_min // 20) % 2 == 0 else 1.0
            return u
        if pid == "actuator_group_pulse":
            u[:] = 1.0
            group = int(ctx.elapsed_min // 20) % 4
            mask = np.arange(n) % 4 == group
            u[mask] = 0.2 if ctx.phase != "recession" else 0.9
            return u
        if pid == "upstream_retain_downstream_release":
            u[storage & (typ != "pump")] = 0.2 if ctx.phase == "peak" else 0.8
            u[(typ == "pump")] = 0.2 if ctx.phase == "peak" else 1.0
            return u
        if pid == "priority_zone_protective":
            u[storage & (typ != "pump")] = 0.1 if ctx.phase in ("pre_peak", "peak") else 0.85
            u[typ == "pump"] = 0.2 if ctx.phase == "peak" else 1.0
            return u
        if pid == "priority_peak_storage_block":
            # Strong storage-retention counterfactual: close storage outlets and
            # throttle pumps during the storm peak, then release during recession.
            if ctx.phase in ("pre_peak", "peak"):
                u[storage_inlet] = 0.35
                u[storage_outlet] = 0.05
                u[storage & non_pump & ~(storage_inlet | storage_outlet)] = 0.10
                u[typ == "pump"] = 0.10
            else:
                u[storage & non_pump] = 0.90
                u[typ == "pump"] = 1.00
            return u.astype(np.float32)
        if pid == "priority_peak_storage_release":
            # Opposite counterfactual for action-direction learning.
            if ctx.phase in ("pre_peak", "peak"):
                u[storage_inlet] = 0.95
                u[storage_outlet] = 0.95
                u[storage & non_pump & ~(storage_inlet | storage_outlet)] = 0.90
                u[typ == "pump"] = 0.65
            else:
                u[storage & non_pump] = 0.45
                u[typ == "pump"] = 0.85
            return u.astype(np.float32)
        if pid == "pump_peak_shutdown":
            u[typ == "pump"] = 0.0 if ctx.phase in ("pre_peak", "peak") else 1.0
            u[storage & non_pump] = 0.25 if ctx.phase == "peak" else 0.85
            return u.astype(np.float32)
        if pid == "pump_peak_boost":
            u[typ == "pump"] = 1.0
            u[storage_inlet] = 0.75 if ctx.phase in ("pre_peak", "peak") else 0.45
            u[storage_outlet] = 0.95 if ctx.phase == "peak" else 0.60
            return u.astype(np.float32)
        if pid == "storage_group_wave":
            u[storage & non_pump] = 0.55
            sidx = np.where(storage & non_pump)[0]
            if len(sidx):
                group = int(ctx.elapsed_min // 15) % min(6, len(sidx))
                active = sidx[np.arange(len(sidx)) % min(6, len(sidx)) == group]
                u[active] = 0.05 if int(ctx.elapsed_min // 30) % 2 == 0 else 1.0
            u[typ == "pump"] = 0.35 if ctx.phase == "peak" else 1.0
            return u.astype(np.float32)
        if pid == "actuator_extreme_counterfactual":
            # Dense, group-wise extreme actions to enrich non-zero PFV deltas.
            u = GenericActionPolicy("auto_rbc", self.actuators).action(ctx)
            groups = 8
            group = int(ctx.elapsed_min // 10) % groups
            mask = np.arange(n) % groups == group
            u[mask] = 0.02 if int(ctx.elapsed_min // 20) % 2 == 0 else 1.0
            return np.clip(u, 0.0, 1.0).astype(np.float32)
        regulator = ((typ == "orifice") | (typ == "weir")) & ~(storage_inlet | storage_outlet)
        if pid == "regulator_restrict_wave":
            u[:] = 1.0
            ridx = np.where(regulator)[0]
            if len(ridx):
                group_count = min(8, len(ridx))
                group = int(ctx.elapsed_min // 10) % group_count
                active = ridx[np.arange(len(ridx)) % group_count == group]
                u[active] = 0.05 if ctx.phase in ("pre_peak", "peak") else 0.65
            return np.clip(u, 0.0, 1.0).astype(np.float32)
        if pid == "regulator_release_wave":
            u[:] = 0.45 if ctx.phase in ("pre_peak", "peak") else 0.70
            ridx = np.where(regulator)[0]
            if len(ridx):
                group_count = min(8, len(ridx))
                group = int(ctx.elapsed_min // 10) % group_count
                active = ridx[np.arange(len(ridx)) % group_count == group]
                u[active] = 1.0
            u[typ == "pump"] = 0.35 if ctx.phase == "peak" else 0.90
            return np.clip(u, 0.0, 1.0).astype(np.float32)
        if pid == "storage_inlet_outlet_sweep":
            u[:] = 0.75
            if int(ctx.elapsed_min // 15) % 2 == 0:
                u[storage_inlet] = 0.10 if ctx.phase in ("pre_peak", "peak") else 0.55
                u[storage_outlet] = 0.90 if ctx.phase == "recession" else 0.35
            else:
                u[storage_inlet] = 0.85
                u[storage_outlet] = 0.10 if ctx.phase in ("pre_peak", "peak") else 0.85
            u[typ == "pump"] = 0.30 if ctx.phase == "peak" else 0.90
            return np.clip(u, 0.0, 1.0).astype(np.float32)
        if pid == "pump_station_wave":
            u[:] = 0.75
            pidx = np.where(typ == "pump")[0]
            if len(pidx):
                group_count = min(6, len(pidx))
                group = int(ctx.elapsed_min // 10) % group_count
                active = pidx[np.arange(len(pidx)) % group_count == group]
                u[pidx] = 0.45 if ctx.phase in ("pre_peak", "peak") else 0.85
                u[active] = 0.05 if ctx.phase == "peak" else 1.0
            u[storage & non_pump] = 0.30 if ctx.phase == "peak" else 0.85
            return np.clip(u, 0.0, 1.0).astype(np.float32)
        if pid == "auto_rbc":
            _, fill = _efd_reference_fill(self.actuators, ctx)
            pump = typ == "pump"
            # Local hydraulic filling, rather than one catchment-wide rainfall
            # threshold, determines each facility's response.
            pump_setting = 0.20 + 0.80 * np.clip((fill - 0.30) / 0.60, 0.0, 1.0)
            if ctx.phase == "recession":
                pump_setting = np.maximum(pump_setting, np.where(fill > 0.20, 0.80, 0.20))
            u[pump] = pump_setting[pump]
            # Existing RTC in this network is dominated by regulated orifices
            # and weirs, not storage assets. Leaving those links at the
            # previous value made Auto-RBC hydraulically identical to
            # no_control despite recording different settings. Apply the
            # same local fill feedback to every controllable regulator.
            regulator = ((typ == "orifice") | (typ == "weir")) & ~storage
            u[regulator] = np.clip(1.0 - 0.80 * fill[regulator], 0.15, 1.0)
            u[storage_outlet] = np.clip(0.15 + 0.80 * fill[storage_outlet], 0.15, 0.95)
            u[storage_inlet] = np.clip(0.95 - 0.80 * fill[storage_inlet], 0.15, 0.95)
            other_storage = storage & non_pump & ~(storage_inlet | storage_outlet)
            u[other_storage] = np.clip(0.20 + 0.70 * fill[other_storage], 0.20, 0.90)
            return u.astype(np.float32)
        if pid in ("efd_static", "efd_storage_priority"):
            # Wuhan-specific EFD-like rule: high-filled local storage/control
            # nodes are released or protected according to their own INP max
            # depths. This borrows the EFD idea, not Astlingen's actual rules.
            d, fill = _efd_reference_fill(self.actuators, ctx)
            # EFD is an independent equal-filling benchmark and must never
            # silently delegate to Auto-RBC: doing so made the two baselines
            # byte-for-byte identical whenever reference depths resolved to
            # zero. With no actuators we simply hold the previous setting; when
            # the fill field is flat the equalization bands below still yield a
            # distinct medium-band profile that differs from Auto-RBC.
            if d.size == 0:
                return u
            storage_control = storage_inlet | storage_outlet
            target_mask = storage_control if np.any(storage_control) else storage
            target = float(np.nanmean(fill[target_mask])) if np.any(target_mask) else float(np.nanmean(fill))
            high = fill > target + 0.08
            low = fill < target - 0.08
            # Storage outlets: empty highly filled storages; retain low-filled ones.
            u[storage_outlet & high] = 0.90 if ctx.phase != "peak" else 0.75
            u[storage_outlet & low] = 0.15
            u[storage_outlet & ~(high | low)] = 0.55
            # Storage inlets: restrict inflow into already high-filled storages.
            u[storage_inlet & high] = 0.15
            u[storage_inlet & low] = 0.85
            u[storage_inlet & ~(high | low)] = 0.55
            # Other storage actuators follow the outlet-style equalization rule.
            other_storage = storage & non_pump & ~(storage_inlet | storage_outlet)
            u[other_storage & high] = 0.85
            u[other_storage & low] = 0.25
            u[other_storage & ~(high | low)] = 0.55
            regulator = ((typ == "orifice") | (typ == "weir")) & ~storage
            u[regulator & high] = 0.25
            u[regulator & low] = 0.90
            u[regulator & ~(high | low)] = 0.60
            # Pumps are intentionally conservative during the rainfall peak to
            # avoid turning EFD into an aggressive global dewatering strategy.
            if pid == "efd_static":
                u[typ == "pump"] = np.where(high[typ == "pump"], 0.80, 0.35)
                if ctx.phase == "recession":
                    u[typ == "pump"] = np.where(fill[typ == "pump"] > 0.20, 1.0, 0.35)
            else:
                # Storage-priority variant: pumps are mostly a recession
                # emptying device; useful as a safer benchmark for PFV-first RTC.
                if ctx.phase != "recession":
                    u[typ == "pump"] = np.where(high[typ == "pump"], 0.55, 0.25)
                else:
                    u[typ == "pump"] = np.where(fill[typ == "pump"] > 0.20, 1.0, 0.25)
            return np.clip(u, 0.0, 1.0).astype(np.float32)
        if pid in ("mixed_safe_random", "random_safe"):
            noise = self.rng.normal(0, 0.12, n)
            u = np.clip(GenericActionPolicy("auto_rbc", self.actuators).action(ctx) + noise, 0.0, 1.0)
            return u.astype(np.float32)
        return u
