"""Build a bounded development-only fast-E2E pool from existing V4 evidence.

This builder is intentionally *not* an exhaustive historical file recovery.  It
uses already-structured Project6 artifacts first:

* strict reusable case/physical manifests;
* the frozen/unified V4 lineage table when present;
* accepted Train1600/V4 sample manifests discovered under project6_dual_reference_v4.

The scientific grouping key is rainfall/event identity, never file count.  Fresh
formal-blind/challenge events are excluded fail-closed.  The output is a
selection/audit manifest; it does not relabel source/unknown data as formal
Wuhan evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sewerrtc.v4.train_v4_loader import ACCEPTANCE_GATE_COLUMNS, compute_acceptance

ROOT = Path(r"E:\RTC_sewer\Project6")
DATA = ROOT / "outputs/project6_dual_reference_v4/final_v4/v42_paper/data_reuse"
V4_ROOT = ROOT / "outputs/project6_dual_reference_v4/final_v4"
OUT = ROOT / "outputs/project6_dual_reference_v4/final_v4/v42_paper/fast_e2e_64plus/core_pool"
FORMAL_ADAPTER = ROOT / "outputs/rainfall_library_v8_storage_variablepump/rainfall_event_table.formal_adapter.json"
RAIN_TABLE = ROOT / "outputs/rainfall_library_v8_storage_variablepump/rainfall_event_table.csv"
ROLES = ("candidate", "no_control", "dynamic_internal", "hold_previous")
RESERVED_SPLIT_TOKENS = ("formal_blind", "challenge", "reserved_evaluation")
SOURCE_PRIORITY = {
    "train1600_accepted": 0,
    "unified_development": 1,
    "target_strict": 2,
    "source_strict": 3,
    "fast_core_compatible": 4,
}


def yes(v: object) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes", "y", "t"}


def json_ids(v: object) -> list[str]:
    try:
        return [str(x) for x in json.loads(str(v))]
    except Exception:
        return []


def _text(v: object) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()


def _checkpoint(row: pd.Series) -> float:
    value = pd.to_numeric(row.get("checkpoint_min", np.nan), errors="coerce")
    if pd.notna(value):
        return float(value)
    checkpoint_id = _text(row.get("checkpoint_id", ""))
    if "__" in checkpoint_id:
        value = pd.to_numeric(checkpoint_id.rsplit("__", 1)[-1], errors="coerce")
        if pd.notna(value):
            return float(value)
    return float("nan")


def _resolved_forcing(case: pd.Series) -> bool:
    """Prefer the authoritative alignment copy after pandas merge."""
    for name in ("same_forcing_pass_y", "same_forcing_pass", "same_forcing_pass_x"):
        if name in case.index:
            return yes(case.get(name, False))
    return False


def _rain_group(row: pd.Series) -> str:
    for name in ("rainfall_sha256", "rainfall_fingerprint", "split_group_key"):
        value = _text(row.get(name, ""))
        if value:
            return value
    return _text(row.get("event_id", ""))


def _state_key(row: pd.Series) -> str:
    """Same-state key must not contain candidate/case identity."""
    for name in ("prefix_state_hash", "state_key"):
        value = _text(row.get(name, ""))
        if value:
            return value
    payload = "|".join(
        [
            _rain_group(row),
            f"{_checkpoint(row):.6f}",
            _text(row.get("network_sha256", "")),
            _text(row.get("event_id", "")),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _h3_schedule_sha(row: pd.Series) -> str:
    """Hash the actually projected/readback H3 schedule when no stored SHA exists."""
    for name in ("actual_schedule_sha256", "candidate_action_sha", "action_readback_sha256"):
        value = _text(row.get(name, ""))
        if value:
            return value
    for name in ("projected_schedule_json", "requested_schedule_json"):
        value = row.get(name, None)
        if value is None or _text(value) == "":
            continue
        try:
            arr = np.asarray(json.loads(str(value)), dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 36)
            h3 = np.ascontiguousarray(arr[:3], dtype=np.float64)
            if h3.size and np.isfinite(h3).all():
                return hashlib.sha256(h3.tobytes(order="C")).hexdigest()
        except Exception:
            continue
    return ""


def _load_reserved_events(root: Path, v4_root: Path) -> tuple[set[str], dict[str, Any]]:
    reserved: set[str] = set()
    audit: dict[str, Any] = {
        "formal_adapter_path": str(root / FORMAL_ADAPTER.relative_to(ROOT)),
        "formal_adapter_found": False,
        "formal_adapter_events": 0,
        "event_ledger_found": False,
        "event_ledger_reserved_events": 0,
    }
    adapter = root / FORMAL_ADAPTER.relative_to(ROOT)
    if adapter.exists():
        try:
            payload = json.loads(adapter.read_text(encoding="utf-8"))
            split = str(payload.get("split", "")).casefold()
            event_ids = {str(x) for x in payload.get("event_ids", [])}
            if any(token in split for token in RESERVED_SPLIT_TOKENS):
                reserved.update(event_ids)
            audit["formal_adapter_found"] = True
            audit["formal_adapter_split"] = str(payload.get("split", ""))
            audit["formal_adapter_events"] = len(event_ids)
        except Exception as exc:
            audit["formal_adapter_error"] = f"{type(exc).__name__}: {exc}"
    ledger = v4_root / "inventory/event_usage_ledger.csv"
    if ledger.exists():
        try:
            frame = pd.read_csv(ledger)
            if {"event_id", "assigned_split"}.issubset(frame.columns):
                mask = frame["assigned_split"].fillna("").astype(str).str.casefold().apply(
                    lambda x: any(token in x for token in RESERVED_SPLIT_TOKENS)
                )
                ids = set(frame.loc[mask, "event_id"].astype(str))
                reserved.update(ids)
                audit["event_ledger_found"] = True
                audit["event_ledger_reserved_events"] = len(ids)
        except Exception as exc:
            audit["event_ledger_error"] = f"{type(exc).__name__}: {exc}"
    audit["reserved_event_union"] = len(reserved)
    return reserved, audit


def _discover_train_manifests(v4_root: Path) -> list[Path]:
    """Cheap metadata discovery only; never scans/reads raw trajectory bodies."""
    explicit = [
        v4_root / "train1600_v3/dataset/train1600_v3_sample_manifest.csv",
        v4_root / "train1600/dataset/train1600_sample_manifest.csv",
    ]
    found = {p.resolve() for p in explicit if p.exists()}
    try:
        result = subprocess.run(
            [
                "rg",
                "--files",
                "-uu",
                "-g",
                "*sample_manifest*.csv",
                "-g",
                "*sample_manifest*.parquet",
                str(v4_root),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode in (0, 1):
            for line in result.stdout.splitlines():
                p = Path(line.strip())
                if p.exists():
                    found.add(p.resolve())
    except FileNotFoundError:
        for p in v4_root.rglob("*sample_manifest*.csv"):
            found.add(p.resolve())
        for p in v4_root.rglob("*sample_manifest*.parquet"):
            found.add(p.resolve())
    return sorted(found, key=lambda p: str(p).casefold())


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def _add_train_manifest_rows(
    rows: list[dict[str, Any]],
    path: Path,
    reserved_events: set[str],
) -> dict[str, Any]:
    frame = _read_table(path)
    result: dict[str, Any] = {
        "path": str(path),
        "rows": int(len(frame)),
        "accepted": 0,
        "usable_candidate_rows": 0,
        "unique_events": int(frame["event_id"].astype(str).nunique()) if "event_id" in frame else 0,
    }
    if frame.empty or not {"case_id", "event_id"}.issubset(frame.columns):
        result["status"] = "not_candidate_lineage_manifest"
        return result
    if not set(ACCEPTANCE_GATE_COLUMNS).issubset(frame.columns):
        result["status"] = "missing_acceptance_contract"
        return result
    accepted = frame.loc[compute_acceptance(frame)].copy()
    result["accepted"] = int(len(accepted))
    result["accepted_events"] = int(accepted["event_id"].astype(str).nunique())
    added = 0
    for _, item in accepted.iterrows():
        event_id = _text(item.get("event_id", ""))
        if not event_id or event_id in reserved_events:
            continue
        split = _text(item.get("split", "")).casefold()
        if any(token in split for token in RESERVED_SPLIT_TOKENS):
            continue
        group = _rain_group(item)
        action_sig = _h3_schedule_sha(item)
        checkpoint = _checkpoint(item)
        if not group or not action_sig or not np.isfinite(checkpoint):
            continue
        state = _state_key(item)
        case_id = _text(item.get("case_id", ""))
        rows.append(
            {
                "rainfall_group_key": group,
                "event_id": event_id,
                "counterfactual_state_key": state,
                "checkpoint_min": checkpoint,
                "candidate_trajectory_key": case_id,
                "no_control_trajectory_key": f"{state}|no_control",
                "dynamic_internal_trajectory_key": f"{state}|dynamic_internal",
                "hold_previous_trajectory_key": f"{state}|hold_previous",
                "candidate_action_signature": action_sig,
                "fast_e2e_admission_tier": "train1600_accepted",
                "domain_id": _text(item.get("domain_id", "v4_train1600_development")),
                "source_role": "development",
                "source_manifest": str(path),
                "case_id": case_id,
                "development_only": True,
            }
        )
        added += 1
    result["usable_candidate_rows"] = added
    result["status"] = "used"
    return result


def _add_strict_rows(
    rows: list[dict[str, Any]],
    cases: pd.DataFrame,
    physical: pd.DataFrame,
    reserved_events: set[str],
) -> tuple[dict[str, int], int]:
    by_id = physical.set_index("physical_identity_sha256", drop=False)
    tiers = {"target_strict": 0, "source_strict": 0, "fast_core_compatible": 0}
    reserved = 0
    for _, case in cases.iterrows():
        event_id = _text(case.get("event_id", ""))
        if _text(case.get("source_role", "")) == "reserved_evaluation" or event_id in reserved_events:
            reserved += 1
            continue
        ids = [i for i in json_ids(case.get("branch_physical_ids", "[]")) if i in by_id.index]
        if not ids:
            continue
        subset = by_id.loc[ids]
        if isinstance(subset, pd.Series):
            subset = subset.to_frame().T
        role_frames = {
            role: subset[subset.get("branch_role", pd.Series("", index=subset.index)).astype(str) == role]
            for role in ROLES
        }
        if any(frame.empty for frame in role_frames.values()):
            continue
        core = all(
            yes(case.get(c, False))
            for c in ("four_reference_complete", "same_state_numeric_pass", "four_reference_finite_pass", "core_trajectory_targets")
        ) and _resolved_forcing(case)
        if not core:
            continue
        tier = (
            "target_strict"
            if yes(case.get("eligible_counterfactual_flood"))
            else "source_strict"
            if yes(case.get("eligible_source_domain_counterfactual_aux"))
            else "fast_core_compatible"
        )
        tiers[tier] += 1
        checkpoint = _checkpoint(case)
        group = _rain_group(case)
        state = _state_key(case)
        if not group or not np.isfinite(checkpoint):
            continue
        candidates = role_frames["candidate"].drop_duplicates("action_readback_sha256")
        nc = role_frames["no_control"].iloc[0]
        di = role_frames["dynamic_internal"].iloc[0]
        hold = role_frames["hold_previous"].iloc[0]
        for _, cand in candidates.iterrows():
            action_sig = _text(cand.get("action_readback_sha256", ""))
            if not action_sig:
                continue
            rows.append(
                {
                    "rainfall_group_key": group,
                    "event_id": event_id,
                    "counterfactual_state_key": state,
                    "checkpoint_min": checkpoint,
                    "candidate_trajectory_key": _text(cand.get("physical_identity_sha256", cand.name)),
                    "no_control_trajectory_key": _text(nc.get("physical_identity_sha256", nc.name)),
                    "dynamic_internal_trajectory_key": _text(di.get("physical_identity_sha256", di.name)),
                    "hold_previous_trajectory_key": _text(hold.get("physical_identity_sha256", hold.name)),
                    "candidate_action_signature": action_sig,
                    "fast_e2e_admission_tier": tier,
                    "domain_id": _text(case.get("domain_id", "")),
                    "source_role": _text(case.get("source_role", "")),
                    "source_manifest": "strict_reusable_pool",
                    "case_id": _text(case.get("case_id", "")),
                    "development_only": True,
                }
            )
    return tiers, reserved


def _add_unified_rows(
    rows: list[dict[str, Any]],
    unified: Path,
    reserved_events: set[str],
) -> dict[str, Any]:
    result = {"path": str(unified), "found": unified.exists(), "rows": 0, "usable_candidate_rows": 0}
    if not unified.exists():
        return result
    lineage = pd.read_parquet(unified)
    result["rows"] = int(len(lineage))
    added = 0
    for _, item in lineage.iterrows():
        event_id = _text(item.get("event_id", ""))
        if event_id in reserved_events:
            continue
        split = _text(item.get("split", "")).casefold()
        if any(token in split for token in RESERVED_SPLIT_TOKENS):
            continue
        group = _rain_group(item)
        checkpoint = _checkpoint(item)
        action_sig = _h3_schedule_sha(item)
        state = _state_key(item)
        if not group or not action_sig or not np.isfinite(checkpoint):
            continue
        rows.append(
            {
                "rainfall_group_key": group,
                "event_id": event_id,
                "counterfactual_state_key": state,
                "checkpoint_min": checkpoint,
                "candidate_trajectory_key": _text(item.get("candidate_trajectory_sha", item.get("candidate_id", ""))),
                "no_control_trajectory_key": _text(item.get("trajectory_no_control_sha", item.get("ref_nc_action_sha", ""))),
                "dynamic_internal_trajectory_key": _text(item.get("trajectory_dynamic_internal_sha", item.get("ref_di_action_sha", ""))),
                "hold_previous_trajectory_key": _text(item.get("trajectory_hold_previous_sha", item.get("ref_hold_action_sha", ""))),
                "candidate_action_signature": action_sig,
                "fast_e2e_admission_tier": "unified_development",
                "domain_id": _text(item.get("domain_id", "unified_development")),
                "source_role": "development",
                "source_manifest": str(unified),
                "case_id": _text(item.get("case_id", item.get("candidate_id", ""))),
                "development_only": True,
            }
        )
        added += 1
    result["usable_candidate_rows"] = added
    result["unique_rainfall_groups"] = int(lineage.get("rainfall_fingerprint", pd.Series(dtype=str)).astype(str).nunique())
    return result


def _stable_group_split(groups: list[str], seed: int) -> pd.DataFrame:
    ranked = sorted(groups, key=lambda g: (hashlib.sha256(f"{seed}:{g}".encode()).hexdigest(), g))
    n_val = max(1, int(round(0.2 * len(ranked)))) if len(ranked) >= 2 else 0
    val = set(ranked[:n_val])
    return pd.DataFrame(
        {
            "rainfall_group_key": ranked,
            "split": ["validation" if g in val else "train" for g in ranked],
        }
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=ROOT)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--v4-root", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--min-checkpoint-min", type=float, default=120.0)
    ap.add_argument("--candidates-per-state", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = args.project_root
    data_dir = args.data_dir or (root / DATA.relative_to(ROOT))
    v4_root = args.v4_root or (root / V4_ROOT.relative_to(ROOT))
    output_dir = args.output_dir or (root / OUT.relative_to(ROOT))

    reserved_events, reserved_audit = _load_reserved_events(root, v4_root)
    rows: list[dict[str, Any]] = []

    strict_tiers = {"target_strict": 0, "source_strict": 0, "fast_core_compatible": 0}
    strict_reserved = 0
    cases_path = data_dir / "reusable_case_manifest.parquet"
    physical_path = data_dir / "reusable_pool_manifest.parquet"
    if cases_path.exists() and physical_path.exists():
        cases = pd.read_parquet(cases_path)
        physical = pd.read_parquet(physical_path)
        strict_tiers, strict_reserved = _add_strict_rows(rows, cases, physical, reserved_events)

    unified = root / "data/v42_final_unified/sample_lineage.parquet"
    unified_audit = _add_unified_rows(rows, unified, reserved_events)

    train_manifest_audits: list[dict[str, Any]] = []
    for path in _discover_train_manifests(v4_root):
        try:
            train_manifest_audits.append(_add_train_manifest_rows(rows, path, reserved_events))
        except Exception as exc:
            train_manifest_audits.append(
                {"path": str(path), "status": "read_error", "error": f"{type(exc).__name__}: {exc}"}
            )

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        manifest = pd.DataFrame(
            columns=[
                "rainfall_group_key",
                "event_id",
                "counterfactual_state_key",
                "checkpoint_min",
                "candidate_action_signature",
                "fast_e2e_admission_tier",
            ]
        )
    manifest["checkpoint_min"] = pd.to_numeric(manifest["checkpoint_min"], errors="coerce")
    manifest["source_priority"] = manifest.get("fast_e2e_admission_tier", pd.Series(index=manifest.index, dtype=str)).map(SOURCE_PRIORITY).fillna(99)
    manifest = manifest.sort_values(
        ["source_priority", "rainfall_group_key", "counterfactual_state_key", "candidate_action_signature"],
        kind="mergesort",
    ).drop_duplicates(
        ["rainfall_group_key", "counterfactual_state_key", "candidate_action_signature"],
        keep="first",
    ).reset_index(drop=True)

    late = manifest[manifest.checkpoint_min.ge(float(args.min_checkpoint_min))].copy()
    counts = (
        late.groupby(["rainfall_group_key", "counterfactual_state_key"])["candidate_action_signature"].nunique()
        if not late.empty
        else pd.Series(dtype=int)
    )
    usable_states = counts[counts.ge(int(args.candidates_per_state))]
    groups = sorted(usable_states.index.get_level_values(0).unique()) if len(usable_states) else []

    selected_rows: list[pd.DataFrame] = []
    for group in groups:
        state_counts = usable_states.loc[group]
        if isinstance(state_counts, pd.Series):
            best_state = sorted(state_counts.index, key=lambda s: (-int(state_counts.loc[s]), str(s)))[0]
        else:
            best_state = str(state_counts.name)
        sub = late[(late.rainfall_group_key.astype(str) == str(group)) & (late.counterfactual_state_key.astype(str) == str(best_state))]
        sub = sub.sort_values(["source_priority", "candidate_action_signature"], kind="mergesort")
        selected_rows.append(sub.head(max(int(args.candidates_per_state), 3)))
    selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else manifest.iloc[0:0].copy()

    split = _stable_group_split(groups, int(args.seed))
    train_groups = int((split.split == "train").sum()) if not split.empty else 0
    val_groups = int((split.split == "validation").sum()) if not split.empty else 0

    source_counts = manifest.get("fast_e2e_admission_tier", pd.Series(dtype=str)).value_counts().to_dict()
    current_rain_table_rows = None
    rain_table = root / RAIN_TABLE.relative_to(ROOT)
    if rain_table.exists():
        try:
            current_rain_table_rows = int(len(pd.read_csv(rain_table)))
        except Exception:
            current_rain_table_rows = None

    audit = {
        "development_only": True,
        "formal_mainline_authorized": False,
        "admission_policy": "structured_v4_manifests_first_no_exhaustive_raw_scan",
        "total_candidate_rows_after_dedup": int(len(manifest)),
        "unique_rainfall_groups": int(manifest.rainfall_group_key.astype(str).nunique()) if not manifest.empty else 0,
        "checkpoint_ge120_groups": int(late.rainfall_group_key.astype(str).nunique()) if not late.empty else 0,
        "states_ge_checkpoint_gate": int(late.counterfactual_state_key.astype(str).nunique()) if not late.empty else 0,
        "states_with_candidate_choice": int(len(usable_states)),
        "usable_groups": int(len(groups)),
        "selected_rows_one_state_per_group": int(len(selected)),
        "candidate_count_distribution": {str(k): int(v) for k, v in counts.value_counts().sort_index().items()},
        "source_tier_counts": {str(k): int(v) for k, v in source_counts.items()},
        "strict_source_tier_case_counts": strict_tiers,
        "strict_reserved_excluded": int(strict_reserved),
        "train_rainfall_groups": train_groups,
        "validation_rainfall_groups": val_groups,
        "train_gt64": bool(train_groups >= 65),
        "rainfall_event_table_rows_current": current_rain_table_rows,
        "reserved_event_policy": reserved_audit,
        "unified_lineage": unified_audit,
        "train_manifest_audits": train_manifest_audits,
        "important_note": (
            "sample/case count is not rainfall diversity; formal-blind adapter events are excluded. "
            "The potential run is authorised only when >=65 independent rainfall groups are in Step2 train."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(output_dir / "FAST_CORE_CASE_MANIFEST.parquet", index=False)
    selected.to_parquet(output_dir / "FAST_CORE_SELECTED_CASES.parquet", index=False)
    split.to_csv(output_dir / "FAST_CORE_RAINFALL_GROUPS.csv", index=False)
    (output_dir / "FAST_CORE_POOL_AUDIT.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
