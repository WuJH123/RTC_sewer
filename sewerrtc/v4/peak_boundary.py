from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sewerrtc.control.v4_candidate_generator import (
    _float_or,
    _is_binary,
    _semantics_by_id,
    project_candidate_schedule,
)
from sewerrtc.prompt3.gate5r_pipeline import (
    branch_state_hashes,
    hashes_match_across_branches,
    schedule_action_cost,
)
from sewerrtc.v4.labels import classify_labels, window_kpis
from sewerrtc.v4.partial_audit import HARD_AUTHENTICITY_COLUMNS


PEAK_FAMILIES = (
    "synchronized_pump_starts",
    "simultaneous_orifice_opening",
    "synchronized_storage_release",
    "remove_staggering",
    "front_loaded_peak_release",
    "same_downstream_bottleneck",
)


def _schedule_sha(schedule: np.ndarray) -> str:
    return hashlib.sha256(
        np.round(np.asarray(schedule, dtype=np.float64), 8).tobytes()
    ).hexdigest()


def verify_projection_constraints(
    projected: np.ndarray,
    anchor: np.ndarray,
    facility_ids: list[str],
    facility_semantics: pd.DataFrame,
    *,
    atol: float = 1e-8,
) -> dict[str, bool]:
    """Independently re-check a projected schedule against Engineering36.

    This is a verification pass, not a repair pass: it never modifies the
    schedule and mirrors the projector's binary/rate/dwell/interlock rules
    so preflight can prove pre-projection compliance from the plan alone.
    """
    projected = np.asarray(projected, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    semantics = _semantics_by_id(facility_ids, facility_semantics)
    binary_ok = True
    rate_ok = True
    dwell_ok = True
    for index, facility_id in enumerate(facility_ids):
        row = semantics[facility_id]
        lower = _float_or(row.get("lower_bound"), 0.0)
        upper = _float_or(row.get("upper_bound"), 1.0)
        if lower > upper:
            lower, upper = upper, lower
        column = projected[:, index]
        if _is_binary(facility_id, row):
            binary_ok = binary_ok and bool(
                np.all(
                    np.isclose(column, lower, atol=atol)
                    | np.isclose(column, upper, atol=atol)
                )
            )
            min_hold = max(1, int(_float_or(row.get("min_hold_steps"), 1)))
            last_change = -min_hold
            previous = float(anchor[0, index])
            for step in range(projected.shape[0]):
                value = float(column[step])
                if not np.isclose(value, previous, atol=atol):
                    if step - last_change < min_hold:
                        dwell_ok = False
                        break
                    last_change = step
                    previous = value
        else:
            rate_limit = _float_or(row.get("rate_limit"), 1.0)
            previous = float(anchor[0, index])
            for step in range(projected.shape[0]):
                if abs(float(column[step]) - previous) > rate_limit + atol:
                    rate_ok = False
                    break
                previous = float(column[step])
    groups: dict[str, dict[str, list[int]]] = {}
    for index, facility_id in enumerate(facility_ids):
        row = semantics[facility_id]
        group = str(row.get("interlock_group", "")).strip()
        role = str(row.get("storage_role", "")).strip().lower()
        if not group or group.lower() == "nan":
            continue
        bucket = groups.setdefault(group, {"inlet": [], "outlet": []})
        if role == "storage_inlet":
            bucket["inlet"].append(index)
        elif role == "storage_outlet":
            bucket["outlet"].append(index)
    interlock_ok = True
    for members in groups.values():
        for step in range(projected.shape[0]):
            inlet_open = any(
                projected[step, idx] > 0.01 for idx in members["inlet"]
            )
            outlet_open = any(
                projected[step, idx] > 0.01 for idx in members["outlet"]
            )
            if inlet_open and outlet_open:
                interlock_ok = False
                break
    return {
        "binary_semantics_ok": bool(binary_ok),
        "rate_limit_ok": bool(rate_ok),
        "dwell_ok": bool(dwell_ok),
        "interlock_ok": bool(interlock_ok),
    }


def build_peak_candidate_catalog(
    opportunities: pd.DataFrame,
    *,
    facility_ids: list[str],
    facility_semantics: pd.DataFrame,
    target_count: int = 60,
) -> pd.DataFrame:
    """Build deterministic Peak-stress candidates without future hydraulics."""
    required = {
        "event_id",
        "checkpoint_id",
        "checkpoint_min",
        "opportunity_class",
        "anchor_action_json",
        "active_facility_ids_json",
    }
    missing = required - set(opportunities)
    if missing:
        raise ValueError(f"Opportunity pool missing: {sorted(missing)}")
    source = opportunities[
        opportunities["opportunity_class"].eq("responsive")
    ].copy()
    if source.empty:
        raise ValueError("Peak candidates require responsive checkpoints")
    semantics = facility_semantics.set_index("facility_id", drop=False)
    pumps = [
        facility
        for facility in facility_ids
        if facility in semantics.index
        and str(semantics.loc[facility].get("actuator_type", "")).lower()
        == "pump"
    ]
    orifices = [
        facility
        for facility in facility_ids
        if facility in semantics.index
        and str(semantics.loc[facility].get("actuator_type", "")).lower()
        in {"orifice", "weir"}
    ]
    storage_outlets = [
        facility
        for facility in facility_ids
        if facility in semantics.index
        and str(semantics.loc[facility].get("storage_role", "")).lower()
        == "storage_outlet"
    ]
    continuous = [
        facility
        for facility in facility_ids
        if facility in semantics.index
        and str(
            semantics.loc[facility].get("binary_or_continuous", "")
        ).lower()
        != "binary"
    ]
    rows: list[dict] = []
    seen: set[str] = set()
    attempts = 0
    maximum_attempts = max(500, int(target_count) * 50)
    while len(rows) < int(target_count) and attempts < maximum_attempts:
        checkpoint = source.iloc[attempts % len(source)]
        family = PEAK_FAMILIES[attempts % len(PEAK_FAMILIES)]
        k_target = (2, 4, 6, 8)[(attempts // len(PEAK_FAMILIES)) % 4]
        hold_steps = (4, 6)[attempts % 2]
        anchor_vector = np.asarray(
            json.loads(str(checkpoint["anchor_action_json"])), dtype=float
        )
        if anchor_vector.size != len(facility_ids):
            raise ValueError("anchor action does not match Engineering36 order")
        anchor = np.repeat(anchor_vector.reshape(1, -1), 12, axis=0)
        active = [
            str(item)
            for item in json.loads(
                str(checkpoint["active_facility_ids_json"])
            )
            if str(item) in facility_ids
        ]
        if family == "synchronized_pump_starts":
            preferred = pumps
        elif family == "simultaneous_orifice_opening":
            preferred = orifices
        elif family == "synchronized_storage_release":
            preferred = storage_outlets
        else:
            preferred = active
        ordered = list(dict.fromkeys([*preferred, *active, *continuous, *facility_ids]))
        chosen = ordered[: min(int(k_target), len(ordered))]
        requested = anchor.copy()
        amplitude = 0.55 + 0.4 * ((attempts % 37) / 36.0)
        for facility in chosen:
            index = facility_ids.index(facility)
            binary = (
                facility in semantics.index
                and str(
                    semantics.loc[facility].get(
                        "binary_or_continuous", ""
                    )
                ).lower()
                == "binary"
            )
            target = 1.0 if binary else amplitude
            if np.isclose(target, anchor_vector[index]):
                target = 0.0 if target > 0.5 else 1.0
            requested[:hold_steps, index] = target
        # A continuous channel makes the stress lattice informative even when
        # native pump settings are already at binary extremes.
        if continuous:
            index = facility_ids.index(continuous[attempts % len(continuous)])
            requested[:hold_steps, index] = amplitude
        requested_sha = _schedule_sha(requested)
        attempts += 1
        if requested_sha in seen:
            continue
        projected, projection = project_candidate_schedule(
            requested,
            anchor,
            facility_ids,
            facility_semantics,
            max_k=8,
        )
        if np.allclose(projected, anchor):
            continue
        constraint_flags = verify_projection_constraints(
            projected, anchor, facility_ids, facility_semantics
        )
        # Fail-closed: a candidate whose projection cannot be independently
        # re-verified never enters the plan.
        if not all(constraint_flags.values()):
            continue
        seen.add(requested_sha)
        base_kwargs = json.loads(
            str(checkpoint.get("source_runner_kwargs", "{}"))
        )
        base_kwargs.update(
            {
                "override_start_min": float(checkpoint["checkpoint_min"]),
                "post_action": projected.tolist(),
                "stop_after_override_min": 120.0,
                "prefix_history_min": 60.0,
                "decision_interval_sec": 600,
                "control_step_sec": 300,
                "post_control_mode": "external_override",
                "hotstart_dir": None,
            }
        )
        sample_id = (
            f"peak__{checkpoint['checkpoint_id']}__{len(rows):03d}"
        )
        rows.append(
            {
                **checkpoint.to_dict(),
                "sample_id": sample_id,
                "case_id": sample_id,
                "family": family,
                "k_target": int(k_target),
                "hold_steps": int(hold_steps),
                **constraint_flags,
                "requested_schedule_json": json.dumps(requested.tolist()),
                "projected_schedule_json": json.dumps(projected.tolist()),
                "anchor_schedule_json": json.dumps(anchor.tolist()),
                "requested_schedule_sha256": requested_sha,
                "projected_schedule_sha256": projection[
                    "projected_schedule_hash"
                ],
                "runner_function": "run_swmm_fixed_action",
                "runner_kwargs": json.dumps(
                    base_kwargs, separators=(",", ":")
                ),
            }
        )
    if len(rows) < int(target_count):
        raise ValueError(
            f"could generate only {len(rows)} unique Peak candidates"
        )
    return pd.DataFrame(rows)


def build_peak_boundary_plan(
    candidates: pd.DataFrame, minimum: int = 30, maximum: int = 60
) -> pd.DataFrame:
    required = {
        "event_id",
        "checkpoint_id",
        "family",
        "requested_schedule_sha256",
    }
    missing = required - set(candidates)
    if missing:
        raise ValueError(f"candidate catalog missing: {sorted(missing)}")
    targeted = candidates[candidates["family"].isin(PEAK_FAMILIES)].copy()
    targeted = targeted.drop_duplicates(
        ["event_id", "checkpoint_id", "requested_schedule_sha256"]
    )
    if len(targeted) < int(minimum):
        raise ValueError(
            f"need at least {minimum} Peak-boundary candidates, found {len(targeted)}"
        )
    selected = targeted.head(int(maximum)).reset_index(drop=True)
    # Peak-boundary events are development-pool only (never formal), so the
    # split assignment is a constant and split isolation holds by design.
    selected["split"] = "development"
    if not {
        "runner_function",
        "runner_kwargs",
        "projected_schedule_json",
        "anchor_schedule_json",
    }.issubset(selected):
        return selected
    rows: list[dict] = []
    for _, candidate in selected.iterrows():
        base = json.loads(str(candidate["runner_kwargs"]))
        projected = json.loads(str(candidate["projected_schedule_json"]))
        anchor = json.loads(str(candidate["anchor_schedule_json"]))
        for branch in (
            "candidate",
            "no_control",
            "dynamic_internal_rules",
            "hold_previous",
        ):
            kwargs = dict(base)
            kwargs["post_control_mode"] = (
                "native_rules"
                if branch == "dynamic_internal_rules"
                else "external_override"
            )
            if branch == "candidate":
                kwargs["post_action"] = projected
            elif branch == "no_control":
                width = len(projected[0])
                kwargs["post_action"] = [[1.0] * width for _ in range(12)]
            else:
                kwargs["post_action"] = anchor
            sample_id = str(candidate.get("sample_id", candidate["case_id"]))
            rows.append(
                {
                    **candidate.to_dict(),
                    "sample_id": sample_id,
                    "case_id": f"{sample_id}__{branch}",
                    "branch": branch,
                    "runner_function": "run_swmm_fixed_action",
                    "runner_kwargs": json.dumps(
                        kwargs, separators=(",", ":")
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_peak_boundary_dataset(
    run_manifest: pd.DataFrame,
    *,
    priority_nodes: list[str],
    facility_ids: list[str],
    scientific_margin: dict[str, float],
    dead_zone: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build exact Peak labels only from complete same-state four-branch cases."""
    required = {
        "sample_id",
        "event_id",
        "checkpoint_id",
        "checkpoint_min",
        "branch",
        "family",
        "status",
        "detail_path",
    }
    missing = required - set(run_manifest)
    if missing:
        raise ValueError(f"Peak run manifest missing: {sorted(missing)}")
    accepted: list[dict] = []
    rejected: list[dict] = []
    required_branches = {
        "candidate",
        "no_control",
        "dynamic_internal_rules",
        "hold_previous",
    }
    for sample_id, group in run_manifest.groupby("sample_id"):
        branches = set(group.loc[group["status"].eq("pass"), "branch"])
        if branches != required_branches:
            rejected.append(
                {
                    "sample_id": sample_id,
                    "rejection_reason": "four_branch_incomplete",
                }
            )
            continue
        details: dict[str, pd.DataFrame] = {}
        paths_ok = True
        for _, branch_row in group.iterrows():
            path = Path(str(branch_row["detail_path"]))
            if not path.exists():
                paths_ok = False
                break
            details[str(branch_row["branch"])] = pd.read_csv(path)
        if not paths_ok or set(details) != required_branches:
            rejected.append(
                {
                    "sample_id": sample_id,
                    "rejection_reason": "branch_detail_missing",
                }
            )
            continue
        first = group.iloc[0]
        checkpoint = float(first["checkpoint_min"])
        hashes = {
            branch: branch_state_hashes(
                detail,
                checkpoint_min=checkpoint,
                facility_ids=facility_ids,
            )
            for branch, detail in details.items()
        }
        same_state = hashes_match_across_branches(
            hashes,
            keys=(
                "prefix_history_sha256",
                "checkpoint_pre_action_sha256",
            ),
        )
        if not same_state:
            rejected.append(
                {
                    "sample_id": sample_id,
                    "rejection_reason": "same_state_hash_mismatch",
                }
            )
            continue
        kpis = {
            branch: window_kpis(
                detail, priority_nodes, checkpoint, dt_sec=300
            )
            for branch, detail in details.items()
        }
        candidate = kpis["candidate"]
        delta_pfv = candidate["PFV"] - kpis["no_control"]["PFV"]
        delta_tfv = candidate["TFV"] - kpis["dynamic_internal_rules"]["TFV"]
        delta_peak = (
            candidate["peak_TFV_rate"]
            - kpis["dynamic_internal_rules"]["peak_TFV_rate"]
        )
        projected = np.asarray(
            json.loads(str(first.get("projected_schedule_json", "[]"))),
            dtype=float,
        )
        anchor = np.asarray(
            json.loads(str(first.get("anchor_schedule_json", "[]"))),
            dtype=float,
        )
        action_cost = (
            schedule_action_cost(projected, anchor)
            if projected.size and projected.shape == anchor.shape
            else 0.0
        )
        labels = classify_labels(
            delta_pfv,
            delta_tfv,
            delta_peak,
            scientific_margin=scientific_margin,
            dead_zone=dead_zone,
            action_cost=action_cost,
        )
        candidate_detail = details["candidate"]
        elapsed = pd.to_numeric(
            candidate_detail["elapsed_min"], errors="coerce"
        )
        post = candidate_detail[
            (elapsed > checkpoint) & (elapsed <= checkpoint + 120.0)
        ]
        actual_columns = [
            column
            for column in (
                [f"actual_setting:{item}" for item in facility_ids]
                + [f"a:{item}" for item in facility_ids]
            )
            if column in post
        ]
        # Prefer explicit actual settings and fall back to a:* only for old
        # authoritative runner output.
        explicit = [
            column
            for column in actual_columns
            if column.startswith("actual_setting:")
        ]
        actual_columns = explicit or [
            column for column in actual_columns if column.startswith("a:")
        ]
        actual_sha = _schedule_sha(
            post[actual_columns].to_numpy(float)
            if actual_columns
            else np.empty((0, 0))
        )
        readback_columns = [
            f"readback_setting:{item}"
            for item in facility_ids
            if f"readback_setting:{item}" in post
        ]
        readback_ok = (
            len(explicit) == len(facility_ids)
            and len(readback_columns) == len(facility_ids)
            and np.allclose(
                post[explicit].to_numpy(float),
                post[readback_columns].to_numpy(float),
                atol=1e-8,
                equal_nan=False,
            )
        )
        accepted.append(
            {
                **first.to_dict(),
                "actual_schedule_sha256": actual_sha,
                "state_hash_match": True,
                "readback_ok": readback_ok,
                "action_cost": float(action_cost),
                "delta_pfv_h120_vs_no_control": float(delta_pfv),
                "delta_tfv_h120_vs_dynamic_internal": float(delta_tfv),
                "delta_peak_h120_vs_dynamic_internal": float(delta_peak),
                **labels,
            }
        )
    return pd.DataFrame(accepted), pd.DataFrame(rejected)


def _post_window(
    detail: pd.DataFrame, checkpoint: float, horizon: float = 120.0
) -> pd.DataFrame:
    elapsed = pd.to_numeric(detail["elapsed_min"], errors="coerce")
    return detail[(elapsed > checkpoint) & (elapsed <= checkpoint + horizon)]


def _actual_setting_columns(
    frame: pd.DataFrame, facility_ids: list[str]
) -> tuple[list[str], bool]:
    explicit = [
        f"actual_setting:{item}"
        for item in facility_ids
        if f"actual_setting:{item}" in frame
    ]
    if len(explicit) == len(facility_ids):
        return explicit, True
    fallback = [
        f"a:{item}" for item in facility_ids if f"a:{item}" in frame
    ]
    if len(fallback) == len(facility_ids):
        return fallback, False
    return (explicit or fallback), False


def _local_response_magnitude(
    candidate_detail: pd.DataFrame,
    reference_detail: pd.DataFrame,
    checkpoint: float,
    facility_ids: list[str],
    horizon: float = 120.0,
) -> float:
    """L1 distance between candidate and reference local facility flow.

    Uses per-facility ``flow:*`` columns over the H120 window and falls back
    to the system ``tfv_rate_m3s`` when facility flows are unavailable; branch
    rows are aligned positionally on the shared 5-min grid.
    """
    cand = _post_window(candidate_detail, checkpoint, horizon)
    ref = _post_window(reference_detail, checkpoint, horizon)
    columns = [
        f"flow:{item}"
        for item in facility_ids
        if f"flow:{item}" in cand and f"flow:{item}" in ref
    ]
    if not columns:
        columns = [
            column
            for column in ("tfv_rate_m3s",)
            if column in cand and column in ref
        ]
    if not columns:
        return 0.0
    rows = min(len(cand), len(ref))
    if rows == 0:
        return 0.0
    left = cand[columns].to_numpy(float)[:rows]
    right = ref[columns].to_numpy(float)[:rows]
    return float(np.nansum(np.abs(left - right)))


def build_peak_partial_bundle(
    plan: pd.DataFrame,
    completions: pd.DataFrame,
    *,
    priority_nodes: list[str],
    facility_ids: list[str],
    scientific_margin: dict[str, float],
    dead_zone: dict[str, float],
) -> dict:
    """Reduce completed four-branch Peak cases into gate-ready samples.

    Partial mode only reads samples whose four branches each carry a valid
    completion marker; a sample missing any branch is ``pending`` -- never
    missing and never a failure. Reduction mirrors ``build_peak_boundary_dataset``
    and additionally proves each hard-authenticity column so the partial gate
    can judge a Peak sample the same way the full gate does. A completed sample
    that fails any hard check is rejected fail-closed; no-op and duplicate
    actual schedules are funnelled out exactly like the formal builder.
    """
    if "sample_id" not in plan or "case_id" not in plan or "branch" not in plan:
        raise ValueError("plan needs sample_id, case_id and branch")
    required_branches = {
        "candidate",
        "no_control",
        "dynamic_internal_rules",
        "hold_previous",
    }
    plan = plan.copy()
    plan["case_id"] = plan["case_id"].astype(str)
    comp = (
        completions.copy()
        if completions is not None and not completions.empty
        else pd.DataFrame(columns=["case_id"])
    )
    if "case_id" in comp:
        comp["case_id"] = comp["case_id"].astype(str)
        comp_by_case = {
            str(row["case_id"]): row for _, row in comp.iterrows()
        }
    else:
        comp_by_case = {}

    accepted_rows: list[dict] = []
    hard_failed_rows: list[dict] = []
    noop_rows: list[dict] = []
    branch_rows: list[dict] = []
    pending_ids: list[str] = []
    completed_samples = 0

    for sample_id, group in plan.groupby("sample_id"):
        branch_ids = {
            str(row["branch"]): str(row["case_id"])
            for _, row in group.iterrows()
        }
        present = {
            branch: comp_by_case[case_id]
            for branch, case_id in branch_ids.items()
            if case_id in comp_by_case
        }
        if set(present) != required_branches:
            pending_ids.append(str(sample_id))
            continue
        completed_samples += 1
        for branch, row in present.items():
            record = row.to_dict()
            record["branch_role"] = branch
            record["is_reference_branch"] = branch != "candidate"
            branch_rows.append(record)
        first = present["candidate"]
        base = first.to_dict()
        statuses = {b: str(r.get("status", "")) for b, r in present.items()}
        details: dict[str, pd.DataFrame] = {}
        details_ok = all(
            statuses[b] == "pass" for b in required_branches
        )
        if details_ok:
            for branch, row in present.items():
                path = Path(str(row.get("detail_path", "")))
                if not path.exists():
                    details_ok = False
                    break
                details[branch] = pd.read_csv(path)
        if not details_ok or set(details) != required_branches:
            hard_failed_rows.append(
                {
                    **base,
                    "sample_id": sample_id,
                    "branch_role": "candidate",
                    "rejection_reason": "four_branch_incomplete_or_failed",
                    **{column: False for column in HARD_AUTHENTICITY_COLUMNS},
                }
            )
            continue

        checkpoint = float(base.get("checkpoint_min"))
        hashes = {
            branch: branch_state_hashes(
                detail, checkpoint_min=checkpoint, facility_ids=facility_ids
            )
            for branch, detail in details.items()
        }
        same_state = hashes_match_across_branches(
            hashes,
            keys=("prefix_history_sha256", "checkpoint_pre_action_sha256"),
        )
        prefix_ok = (
            len({h.get("prefix_history_sha256") for h in hashes.values()})
            == 1
        )
        kpis = {
            branch: window_kpis(detail, priority_nodes, checkpoint, dt_sec=300)
            for branch, detail in details.items()
        }
        cand = kpis["candidate"]
        delta_pfv = cand["PFV"] - kpis["no_control"]["PFV"]
        delta_tfv = cand["TFV"] - kpis["dynamic_internal_rules"]["TFV"]
        delta_peak = (
            cand["peak_TFV_rate"]
            - kpis["dynamic_internal_rules"]["peak_TFV_rate"]
        )
        kpi_ok = all(
            np.isfinite(
                [value["PFV"], value["TFV"], value["peak_TFV_rate"]]
            ).all()
            for value in kpis.values()
        )
        projected = np.asarray(
            json.loads(str(base.get("projected_schedule_json", "[]"))),
            dtype=float,
        )
        anchor = np.asarray(
            json.loads(str(base.get("anchor_schedule_json", "[]"))),
            dtype=float,
        )
        action_cost = (
            schedule_action_cost(projected, anchor)
            if projected.size and projected.shape == anchor.shape
            else 0.0
        )
        labels = classify_labels(
            delta_pfv,
            delta_tfv,
            delta_peak,
            scientific_margin=scientific_margin,
            dead_zone=dead_zone,
            action_cost=action_cost,
        )
        candidate_detail = details["candidate"]
        post = _post_window(candidate_detail, checkpoint)
        actual_columns, explicit = _actual_setting_columns(post, facility_ids)
        actual_matrix = (
            post[actual_columns].to_numpy(float)
            if actual_columns
            else np.empty((0, 0))
        )
        actual_sha = _schedule_sha(actual_matrix)
        anchor_vector = (
            anchor[0] if anchor.ndim == 2 and anchor.size else np.zeros(0)
        )
        actual_distance = (
            float(np.abs(actual_matrix - anchor_vector).sum())
            if actual_matrix.size and anchor_vector.size == actual_matrix.shape[1]
            else 0.0
        )
        readback_columns = [
            f"readback_setting:{item}"
            for item in facility_ids
            if f"readback_setting:{item}" in post
        ]
        readback_ok = bool(
            explicit
            and len(readback_columns) == len(facility_ids)
            and np.allclose(
                post[
                    [f"actual_setting:{item}" for item in facility_ids]
                ].to_numpy(float),
                post[readback_columns].to_numpy(float),
                atol=1e-8,
                equal_nan=False,
            )
        )
        local_magnitude = _local_response_magnitude(
            details["candidate"],
            details["hold_previous"],
            checkpoint,
            facility_ids,
        )
        chain_ok = all(
            f"{prefix}:{item}" in candidate_detail
            for item in facility_ids
            for prefix in (
                "requested_setting",
                "target_setting",
                "actual_setting",
                "readback_setting",
            )
        )

        def _no_hotstart(row: pd.Series) -> bool:
            result = row.get("result")
            result = result if isinstance(result, dict) else {}
            return (
                not bool(result.get("hotstart_used", False))
                and int(result.get("use_hotstart_call_count", 0) or 0) == 0
                and int(result.get("save_hotstart_call_count", 0) or 0) == 0
            )

        no_hotstart = all(_no_hotstart(row) for row in present.values())

        def _inp_path(row: pd.Series) -> str | None:
            try:
                return json.loads(str(row.get("runner_kwargs", "{}"))).get(
                    "inp_path"
                )
            except (ValueError, TypeError):
                return None

        physical_ok = (
            len({_inp_path(row) for row in present.values()}) == 1
            and _inp_path(first) is not None
        )
        rainfall_ok = (
            len({str(row.get("rainfall_sha256")) for row in present.values()})
            == 1
            and bool(base.get("rainfall_sha256"))
        )
        elapsed = pd.to_numeric(
            candidate_detail["elapsed_min"], errors="coerce"
        )
        h120_ok = bool(
            float(elapsed.max()) >= checkpoint + 120.0 - 1e-6 and len(post) >= 1
        )
        k_target = int(base.get("k_target", 0) or 0)
        actuator_ok = bool(base.get("binary_semantics_ok")) and bool(
            base.get("rate_limit_ok")
        )
        engineering_ok = (
            bool(base.get("binary_semantics_ok"))
            and bool(base.get("rate_limit_ok"))
            and bool(base.get("dwell_ok"))
            and bool(base.get("interlock_ok"))
        )
        hard = {
            "completion_valid": all(
                statuses[b] == "pass" for b in required_branches
            )
            and all(bool(row.get("input_sha")) for row in present.values()),
            "four_branches_complete": True,
            "same_state_ok": bool(same_state),
            "physical_sha_ok": bool(physical_ok),
            "rainfall_sha_ok": bool(rainfall_ok),
            "prefix_sha_ok": bool(prefix_ok),
            "action_stage_chain_complete": bool(chain_ok),
            "no_hotstart": bool(no_hotstart),
            "k_le_8": 1 <= k_target <= 8,
            "actuator_semantics_ok": actuator_ok,
            "engineering_limits_ok": engineering_ok,
            "h120_window_complete": h120_ok,
            "kpi_recompute_ok": bool(kpi_ok),
            "reference_cache_sha_ok": bool(same_state),
        }
        output_isolated = all(
            branch_ids[branch] in str(present[branch].get("detail_path", ""))
            for branch in required_branches
        ) and len({branch_ids[branch] for branch in required_branches}) == len(
            required_branches
        )
        is_noop = actual_distance <= 1e-9
        row = {
            **base,
            "branch_role": "candidate",
            "is_reference_branch": False,
            "candidate_family": base.get("family"),
            "K": k_target,
            "checkpoint_role": base.get("opportunity_class"),
            "rainfall_phase": base.get("phase"),
            "actual_schedule_sha256": actual_sha,
            "actual_action_distance": actual_distance,
            "local_response_magnitude": local_magnitude,
            "is_noop": bool(is_noop),
            "output_isolated": bool(output_isolated),
            "state_hash_match": bool(same_state),
            "readback_ok": readback_ok,
            "action_cost": float(action_cost),
            "delta_pfv_h120_vs_no_control": float(delta_pfv),
            "delta_tfv_h120_vs_dynamic_internal": float(delta_tfv),
            "delta_peak_h120_vs_dynamic_internal": float(delta_peak),
            **hard,
            **labels,
        }
        if not all(hard.values()):
            hard_failed_rows.append(
                {**row, "rejection_reason": "hard_authenticity_failed"}
            )
        elif is_noop:
            noop_rows.append(
                {**row, "rejection_reason": "no_op_not_accepted"}
            )
        else:
            accepted_rows.append(row)

    accepted_df = pd.DataFrame(accepted_rows)
    dup_keys = ["event_id", "checkpoint_id", "actual_schedule_sha256"]
    if len(accepted_df) and all(key in accepted_df for key in dup_keys):
        dup_mask = accepted_df.duplicated(dup_keys, keep="first")
    else:
        dup_mask = pd.Series(False, index=accepted_df.index)
    duplicates_df = accepted_df[dup_mask].copy()
    if len(duplicates_df):
        duplicates_df["rejection_reason"] = "duplicate_actual_schedule"
    final_accepted = accepted_df[~dup_mask].copy()

    rejected = pd.concat(
        [pd.DataFrame(hard_failed_rows), pd.DataFrame(noop_rows)],
        ignore_index=True,
    )
    pending = (
        plan[plan["sample_id"].isin(pending_ids)]
        .drop_duplicates("sample_id")
        .reset_index(drop=True)
    )
    return {
        "sample_manifest": final_accepted.reset_index(drop=True),
        "branch_manifest": pd.DataFrame(branch_rows).reset_index(drop=True),
        "rejected": rejected.reset_index(drop=True),
        "actual_duplicates": duplicates_df.reset_index(drop=True),
        "pending": pending,
        "missing_confirmed": plan.head(0).copy(),
        "completed_total": int(completed_samples),
        "hard_violation_total": int(len(hard_failed_rows)),
    }


def audit_peak_boundary(samples: pd.DataFrame) -> dict:
    required = {
        "event_id",
        "checkpoint_id",
        "actual_schedule_sha256",
        "peak_noninferior",
        "pfv_safe",
        "family",
    }
    missing = required - set(samples)
    if missing:
        return {"status": "blocked", "missing_columns": sorted(missing)}
    unique = samples.drop_duplicates(
        ["event_id", "checkpoint_id", "actual_schedule_sha256"]
    )
    degraded = unique[~unique["peak_noninferior"].astype(bool)]
    safe = degraded[degraded["pfv_safe"].astype(bool)]
    checks = {
        "events_at_least_3": degraded["event_id"].nunique() >= 3,
        "checkpoints_at_least_6": degraded.groupby(
            ["event_id", "checkpoint_id"]
        ).ngroups
        >= 6,
        "peak_degraded_30_to_60": 30 <= len(degraded) <= 60,
        "pfv_safe_peak_hard_negative_at_least_10": len(safe) >= 10,
        "families_at_least_2": degraded["family"].nunique() >= 2,
    }
    return {
        "status": "pass" if all(checks.values()) else "scientific_fail",
        "checks": checks,
        "peak_degraded": int(len(degraded)),
        "pfv_safe_peak_hard_negative": int(len(safe)),
    }


def peak_constraint_binding_audit(samples: pd.DataFrame) -> dict:
    if "peak_noninferior" not in samples:
        raise ValueError("peak_noninferior is required")
    degraded = samples[~samples["peak_noninferior"].astype(bool)]
    return {
        "status": "binding_or_unobserved" if degraded.empty else "not_binding",
        "peak_degraded": int(len(degraded)),
        "events_searched": int(samples["event_id"].nunique())
        if "event_id" in samples
        else 0,
        "checkpoints_searched": int(
            samples.groupby(["event_id", "checkpoint_id"]).ngroups
        )
        if {"event_id", "checkpoint_id"}.issubset(samples)
        else 0,
        "remove_peak_constraint": False,
        "scientific_margin_changed": False,
    }
