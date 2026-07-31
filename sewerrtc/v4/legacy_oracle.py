"""Gate P3 legacy oracle compatibility audit (read-only evidence scan).

Implements the ``AuditLegacyOracleCompatibility`` stage of
``docs/contracts/PROJECT6_V4_PILOT_FEASIBILITY_GATE_P3.json``: every legacy
oracle / Gate5 / ablation / physical-feasibility evidence row is checked
against the 13 frozen compatibility dimensions.  Only rows that pass ALL
dimensions may serve as positive-control replays; incompatible rows may only
seed the exact feasibility search and must never be used as labels.

The audit is fail-closed: any dimension that cannot be positively proven
from the legacy evidence itself is recorded as a failure with an explicit
``unverifiable`` evidence string.  Nothing in this module mutates legacy
outputs or the frozen Pilot v1/v2 evidence.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

COMPATIBILITY_DIMENSIONS = (
    "network_sha256",
    "no_dwf_scenario",
    "event_rainfall_sha256",
    "checkpoint_identity",
    "engineering36_scope",
    "k_at_most_8",
    "reference_semantics",
    "dynamic_internal_truly_dynamic",
    "h120_window",
    "actual_readback",
    "margin_dead_zone",
    "no_hotstart",
    "provenance",
)

SEED_USE = "search_seed_only"
POSITIVE_CONTROL_USE = "positive_control_replay"
SEED_FAMILY = "replay_compatible_oracle_seed"

# Source kinds understood by the row evaluator.
KIND_ORACLE_CASE = "oracle_case"
KIND_GATE0_PROOF = "gate0_proof"
KIND_V4_MANIFEST = "v4_manifest"

_CURRENT_REFERENCE_SEMANTICS = (
    "delta_pfv_h120_vs_no_control"
    "+delta_tfv_h120_vs_dynamic_internal"
    "+delta_peak_h120_vs_dynamic_internal"
)

_H120_OK_VALUES = {"ok", "pass", "complete", "completed", "full", "labeled"}


def _as_bool(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _fail(evidence: str) -> tuple[bool, str]:
    return False, evidence


def _ok(evidence: str) -> tuple[bool, str]:
    return True, evidence


def _resolve_schedule_path(
    schedule_csv: str, *, source_dir: Path, legacy_root: Path
) -> Path | None:
    raw = str(schedule_csv or "").strip()
    if not raw or raw.lower() == "nan":
        return None
    candidate = Path(raw)
    probes = (
        [candidate]
        if candidate.is_absolute()
        else [source_dir / raw, legacy_root / raw]
    )
    for probe in probes:
        if probe.exists():
            return probe
    return None


def _evaluate_oracle_case_row(row: pd.Series, ctx: dict) -> dict[str, tuple[bool, str]]:
    dims: dict[str, tuple[bool, str]] = {}
    inp_sha = str(row.get("inp_sha256", "")).strip()
    if inp_sha == ctx["network_sha256"]:
        dims["network_sha256"] = _ok("inp_sha256 matches frozen network")
    else:
        dims["network_sha256"] = _fail(
            f"episode-case inp_sha256 {inp_sha[:12]} != frozen network "
            f"{ctx['network_sha256'][:12]}"
        )
    dims["no_dwf_scenario"] = _fail(
        "unverifiable: DWF scenario not recorded in legacy case row"
    )
    event_id = str(row.get("event_id", "")).strip()
    rain = str(row.get("rainfall_sha256", "")).strip()
    expected_rain = ctx["rainfall_by_event"].get(event_id)
    if expected_rain is None:
        dims["event_rainfall_sha256"] = _fail(
            f"event {event_id} not in current event inventory"
        )
    elif rain == expected_rain:
        dims["event_rainfall_sha256"] = _ok("rainfall sha matches inventory")
    else:
        dims["event_rainfall_sha256"] = _fail(
            f"rainfall sha {rain[:12]} != inventory {expected_rain[:12]}"
        )
    dims["checkpoint_identity"] = _fail(
        "episode-level evidence: no checkpoint_id / checkpoint_state_sha256"
    )
    dims["engineering36_scope"] = _fail(
        "unverifiable: 12x36 Engineering36 schedule scope not recorded"
    )
    k_raw = pd.to_numeric(
        row.get("max_simultaneous_deviations"), errors="coerce"
    )
    if pd.notna(k_raw) and float(k_raw) <= ctx["max_k"]:
        dims["k_at_most_8"] = _ok(
            f"max_simultaneous_deviations={float(k_raw):g}<= {ctx['max_k']}"
        )
    else:
        dims["k_at_most_8"] = _fail(
            f"max_simultaneous_deviations={k_raw} exceeds or missing"
        )
    dims["reference_semantics"] = _fail(
        "absolute episode PFV/TFV metrics; not "
        + _CURRENT_REFERENCE_SEMANTICS
    )
    dims["dynamic_internal_truly_dynamic"] = _fail(
        "no dynamic-internal-rules reference branch recorded"
    )
    dims["h120_window"] = _fail(
        "full-episode metric window; no H120 checkpoint window"
    )
    dims["actual_readback"] = _fail(
        "no actual/readback schedule evidence recorded"
    )
    dims["margin_dead_zone"] = _fail(
        "labels not computed under frozen margins/dead-zones"
    )
    dims["no_hotstart"] = _fail(
        "unverifiable: hotstart usage not recorded in legacy case row"
    )
    provenance_ok = (
        str(row.get("status", "")).strip().lower() == "success"
        and _as_bool(row.get("runtime_executed"))
        and _as_bool(row.get("authoritative_swmm"))
    )
    constraint_mode = str(row.get("constraint_mode", "")).strip()
    if provenance_ok and constraint_mode == "constrained":
        dims["provenance"] = _ok(
            "success + runtime_executed + authoritative_swmm, constrained"
        )
    elif provenance_ok:
        dims["provenance"] = _fail(
            f"relaxed-constraint mode '{constraint_mode}' breaks provenance "
            "for positive-control use"
        )
    else:
        dims["provenance"] = _fail(
            "status/runtime_executed/authoritative_swmm not all proven"
        )
    return dims


def _evaluate_gate0_row(row: pd.Series, ctx: dict) -> dict[str, tuple[bool, str]]:
    dims: dict[str, tuple[bool, str]] = {}
    unverifiable = {
        "network_sha256": "no inp_sha256 recorded in gate0 proof row",
        "no_dwf_scenario": "DWF scenario not recorded",
        "event_rainfall_sha256": "no rainfall_sha256 recorded",
        "checkpoint_identity": "event-level proof: no checkpoint identity",
        "engineering36_scope": "schedule scope not recorded",
        "k_at_most_8": "K not recorded",
        "reference_semantics": "oracle PFV/TFV absolute; not "
        + _CURRENT_REFERENCE_SEMANTICS,
        "dynamic_internal_truly_dynamic": "no DI reference branch recorded",
        "h120_window": "no H120 checkpoint window recorded",
        "margin_dead_zone": "labels not under frozen margins/dead-zones",
        "no_hotstart": "hotstart usage not recorded",
    }
    for name, why in unverifiable.items():
        dims[name] = _fail(f"unverifiable: {why}")
    if _as_bool(row.get("readback_ok")):
        dims["actual_readback"] = _ok("readback_ok=true recorded")
    else:
        dims["actual_readback"] = _fail("readback_ok not proven")
    if _as_bool(row.get("authoritative_swmm")):
        dims["provenance"] = _ok("authoritative_swmm=true recorded")
    else:
        dims["provenance"] = _fail("authoritative_swmm not proven")
    return dims


def _evaluate_v4_manifest_row(row: pd.Series, ctx: dict) -> dict[str, tuple[bool, str]]:
    dims: dict[str, tuple[bool, str]] = {}
    dims["network_sha256"] = _fail(
        "unverifiable: no network/inp sha256 column in v4 dataset manifest"
    )
    dims["no_dwf_scenario"] = _fail(
        "unverifiable: DWF scenario not recorded in v4 manifest"
    )
    event_id = str(row.get("event_id", "")).strip()
    if event_id in ctx["rainfall_by_event"]:
        dims["event_rainfall_sha256"] = _fail(
            "event known to inventory but rainfall sha not recorded in row"
        )
    else:
        dims["event_rainfall_sha256"] = _fail(
            f"event {event_id} not in current event inventory"
        )
    checkpoint_id = str(row.get("checkpoint_id", "")).strip()
    if (event_id, checkpoint_id) in ctx["pilot_state_keys"]:
        dims["checkpoint_identity"] = _fail(
            "checkpoint id matches a pilot state but "
            "checkpoint_state_sha256 not recorded; same-state unproven"
        )
    else:
        dims["checkpoint_identity"] = _fail(
            "checkpoint not in current pilot state catalog"
        )
    dims["engineering36_scope"] = _fail(
        "unverifiable: 12x36 Engineering36 scope not recorded"
    )
    k_raw = pd.to_numeric(row.get("k_value"), errors="coerce")
    if pd.notna(k_raw) and float(k_raw) <= ctx["max_k"]:
        dims["k_at_most_8"] = _ok(f"k_value={float(k_raw):g} <= {ctx['max_k']}")
    else:
        dims["k_at_most_8"] = _fail(f"k_value={k_raw} exceeds or missing")
    contract = str(row.get("v4_label_contract", "")).strip()
    dims["reference_semantics"] = _fail(
        f"label contract '{contract}' != current "
        + _CURRENT_REFERENCE_SEMANTICS
    )
    dims["dynamic_internal_truly_dynamic"] = _fail(
        "unverifiable: internal_rules branch present but dynamic response "
        "proof not recorded"
    )
    h120 = str(row.get("h120_label_status", "")).strip().lower()
    if h120 in _H120_OK_VALUES:
        dims["h120_window"] = _ok(f"h120_label_status={h120}")
    else:
        dims["h120_window"] = _fail(f"h120_label_status={h120 or 'missing'}")
    if _as_bool(row.get("actual_action_present")):
        dims["actual_readback"] = _ok("actual_action_present=true")
    else:
        dims["actual_readback"] = _fail("actual_action_present not proven")
    dims["margin_dead_zone"] = _fail(
        "labels computed under legacy contract, not frozen "
        "margins/dead-zones"
    )
    if str(row.get("hotstart_used_for_label", "")).strip().lower() == "false":
        dims["no_hotstart"] = _ok("hotstart_used_for_label=False")
    else:
        dims["no_hotstart"] = _fail("hotstart_used_for_label not False")
    if _as_bool(row.get("runtime_executed")):
        dims["provenance"] = _ok("runtime_executed=true")
    else:
        dims["provenance"] = _fail("runtime_executed not proven")
    return dims


_ROW_EVALUATORS = {
    KIND_ORACLE_CASE: _evaluate_oracle_case_row,
    KIND_GATE0_PROOF: _evaluate_gate0_row,
    KIND_V4_MANIFEST: _evaluate_v4_manifest_row,
}

_ROW_ID_COLUMNS = {
    KIND_ORACLE_CASE: "case_id",
    KIND_GATE0_PROOF: "event_id",
    KIND_V4_MANIFEST: "sample_id",
}


def evaluate_legacy_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    kind: str,
    context: dict,
    source_dir: Path,
    legacy_root: Path,
) -> pd.DataFrame:
    """One output row per legacy evidence row with all 13 dimension verdicts."""
    if kind not in _ROW_EVALUATORS:
        raise ValueError(f"unknown legacy source kind: {kind}")
    evaluator = _ROW_EVALUATORS[kind]
    id_column = _ROW_ID_COLUMNS[kind]
    rows: list[dict] = []
    for index, row in frame.iterrows():
        dims = evaluator(row, context)
        missing = [name for name in COMPATIBILITY_DIMENSIONS if name not in dims]
        if missing:
            raise ValueError(f"evaluator for {kind} missed dimensions {missing}")
        failed = [
            name for name in COMPATIBILITY_DIMENSIONS if not dims[name][0]
        ]
        schedule_path = None
        if kind == KIND_ORACLE_CASE:
            schedule_path = _resolve_schedule_path(
                row.get("schedule_csv", ""),
                source_dir=source_dir,
                legacy_root=legacy_root,
            )
        provenance_ok = dims["provenance"][0]
        seed_usable = bool(
            kind == KIND_ORACLE_CASE
            and schedule_path is not None
            and str(row.get("status", "")).strip().lower() == "success"
            and _as_bool(row.get("runtime_executed"))
            and _as_bool(row.get("authoritative_swmm"))
        )
        record = {
            "source": source,
            "source_kind": kind,
            "row_index": int(index),
            "evidence_id": str(row.get(id_column, "")),
            "event_id": str(row.get("event_id", "")),
            "checkpoint_id": str(row.get("checkpoint_id", "")),
            "constraint_mode": str(row.get("constraint_mode", "")),
            "schedule_sha256": str(row.get("schedule_sha256", "")),
            "schedule_path_resolved": (
                str(schedule_path) if schedule_path is not None else ""
            ),
            "fully_compatible": not failed,
            "failed_dimensions": ";".join(failed),
            "seed_usable": seed_usable,
            "provenance_ok": provenance_ok,
            "allowed_use": (
                POSITIVE_CONTROL_USE if not failed else SEED_USE
            )
            if seed_usable or not failed
            else "none",
        }
        for name in COMPATIBILITY_DIMENSIONS:
            passed, evidence = dims[name]
            record[f"dim_{name}"] = passed
            record[f"dim_{name}_evidence"] = evidence
        rows.append(record)
    return pd.DataFrame(rows)


def build_replay_seed_plan(evaluated: pd.DataFrame) -> pd.DataFrame:
    """Seed-only replay plan rows for the Round A oracle-seed family."""
    if evaluated.empty:
        return pd.DataFrame(
            columns=[
                "seed_id",
                "source",
                "evidence_id",
                "event_id",
                "schedule_path_resolved",
                "schedule_sha256",
                "constraint_mode",
                "candidate_family",
                "use",
                "requires_projection",
                "label_use_forbidden",
            ]
        )
    seeds = evaluated[evaluated["seed_usable"]].copy()
    plan = pd.DataFrame(
        {
            "seed_id": (
                seeds["source"].astype(str)
                + "::"
                + seeds["evidence_id"].astype(str)
            ),
            "source": seeds["source"].astype(str),
            "evidence_id": seeds["evidence_id"].astype(str),
            "event_id": seeds["event_id"].astype(str),
            "schedule_path_resolved": seeds["schedule_path_resolved"],
            "schedule_sha256": seeds["schedule_sha256"],
            "constraint_mode": seeds["constraint_mode"],
            "candidate_family": SEED_FAMILY,
            "use": SEED_USE,
            "requires_projection": True,
            "label_use_forbidden": True,
        }
    )
    return plan.reset_index(drop=True)


def audit_legacy_oracle_compatibility(
    sources: dict[str, tuple[str, pd.DataFrame, Path]],
    *,
    network_sha256: str,
    rainfall_by_event: dict[str, str],
    pilot_state_keys: set[tuple[str, str]],
    legacy_root: Path,
    max_k: int = 8,
    missing_sources: list[str] | None = None,
) -> dict:
    """Full legacy compatibility audit over all discovered evidence sources.

    ``sources`` maps source name -> (kind, frame, source_dir).  Returns the
    compatible/incompatible frames, the seed replay plan and the audit
    payload.  Compatibility is strictly all-13-dimensions; nothing here ever
    promotes unverifiable evidence.
    """
    context = {
        "network_sha256": str(network_sha256),
        "rainfall_by_event": rainfall_by_event,
        "pilot_state_keys": pilot_state_keys,
        "max_k": int(max_k),
    }
    frames: list[pd.DataFrame] = []
    per_source: dict[str, dict] = {}
    for name, (kind, frame, source_dir) in sources.items():
        evaluated = evaluate_legacy_frame(
            frame,
            source=name,
            kind=kind,
            context=context,
            source_dir=source_dir,
            legacy_root=legacy_root,
        )
        frames.append(evaluated)
        per_source[name] = {
            "kind": kind,
            "rows": int(len(evaluated)),
            "fully_compatible": int(evaluated["fully_compatible"].sum())
            if not evaluated.empty
            else 0,
            "seed_usable": int(evaluated["seed_usable"].sum())
            if not evaluated.empty
            else 0,
        }
    combined = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["fully_compatible", "seed_usable"])
    )
    if combined.empty:
        compatible = combined.copy()
        incompatible = combined.copy()
    else:
        compatible = combined[combined["fully_compatible"]].reset_index(
            drop=True
        )
        incompatible = combined[~combined["fully_compatible"]].reset_index(
            drop=True
        )
    replay_plan = build_replay_seed_plan(combined)
    dimension_fail_counts = {
        name: int((~combined[f"dim_{name}"]).sum())
        if f"dim_{name}" in combined
        else 0
        for name in COMPATIBILITY_DIMENSIONS
    }
    audit = {
        "stage": "AuditLegacyOracleCompatibility",
        "contract_id": "project6_v4_pilot_feasibility_gate_p3",
        "read_only": True,
        "dimensions": list(COMPATIBILITY_DIMENSIONS),
        "network_sha256": str(network_sha256),
        "sources": per_source,
        "missing_sources": sorted(missing_sources or []),
        "evidence_rows": int(len(combined)),
        "fully_compatible_rows": int(len(compatible)),
        "incompatible_rows": int(len(incompatible)),
        "seed_usable_rows": int(len(replay_plan)),
        "dimension_fail_counts": dimension_fail_counts,
        "policy": {
            "fully_compatible_use": POSITIVE_CONTROL_USE,
            "incompatible_use": SEED_USE,
            "labels_from_incompatible": "forbidden",
        },
        "status": "pass" if len(combined) > 0 else "blocked",
    }
    return {
        "combined": combined,
        "compatible": compatible,
        "incompatible": incompatible,
        "replay_plan": replay_plan,
        "audit": audit,
    }
