"""Prepare Project6 V4.2 Formal F2 metadata and untouched evaluation ledger.

The central invariant is that *data eligibility* and *historical contamination*
are different concepts:

- ``formal_step1_allowed`` / ``formal_step2_allowed`` say whether a source may be
  used by a model after the F2 ledger has assigned roles;
- ``historically_revealed`` says whether the rainfall itself has already been
  exposed by prior development/evaluation and therefore cannot be selected as a
  new F2 Calibration/Locked/Challenge/Blind rainfall;
- current Step2 rows that are already admitted (or pending raw re-admission) are
  always contamination because F2 training will consume them;
- event_inventory is an evaluation-selection authority, not contamination;
- opportunity_pool is pre-control checkpoint-selection metadata. Its rows may be
  used only after the ledger is frozen; evaluation-role rainfalls must never be
  promoted to Step1 auxiliary training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.v4.formal_f2 import (
    DEFAULT_COUNTS,
    FORMAL_GENERATION_ID,
    FORMAL_TRAIN_MIN_GROUPS,
    assert_zero_split_overlap,
    build_event_ledger,
    canonical_rain_group,
    explicit_step1_roles,
    formal_step2_metadata_pool,
    load_registry,
    manifest_source_rows,
    pool_summary,
    read_table,
    resolve_source_files,
    split_overlap_matrix,
    text,
    yes,
)


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


def _inventory(root: Path, registry: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    """Return the first event inventory that exposes a real rainfall identity."""
    spec = dict(registry.get("sources", {}).get("event_inventory", {}) or {})
    for path in resolve_source_files(root, spec):
        if path.suffix.lower() not in {".csv", ".parquet"}:
            continue
        frame = read_table(path)
        groups = {canonical_rain_group(r) for r in frame.to_dict("records")}
        groups.discard("")
        if groups:
            return frame, str(path)
    return pd.DataFrame(), ""


def _bool_column(frame: pd.DataFrame, name: str, default: bool = False) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    return frame[name].map(yes).astype(bool)


def _historical_contamination(source_all: pd.DataFrame) -> pd.DataFrame:
    """Rows whose rainfall must be excluded from new F2 evaluation.

    A source being *eligible* for Step1 is not by itself evidence that every
    rainfall in the source has already been consumed. Conversely, any row that
    is already part of the Step2 training population is contamination even if a
    registry author accidentally marks the source as not historically revealed.
    """
    if source_all.empty:
        return source_all.copy()
    revealed = _bool_column(source_all, "historically_revealed")
    step2_allowed = _bool_column(source_all, "formal_step2_allowed")
    admitted = _bool_column(source_all, "step2_accepted_from_manifest")
    pending = _bool_column(source_all, "raw_readmission_required")
    step2_training_population = step2_allowed & (admitted | pending)
    out = source_all.loc[revealed | step2_training_population].copy()
    return out


def _groups_for_events(frame: pd.DataFrame, events: set[str]) -> set[str]:
    if frame.empty or not events:
        return set()
    event_col = next((c for c in ("event_id", "rainfall_event_id") if c in frame.columns), None)
    if event_col is None:
        return set()
    subset = frame.loc[frame[event_col].astype(str).isin(events)]
    return {g for g in (canonical_rain_group(r) for r in subset.to_dict("records")) if g}


def _reserved(
    root: Path,
    source_all: pd.DataFrame,
    inventory: pd.DataFrame,
) -> tuple[set[str], set[str], dict[str, Any]]:
    """Resolve historical reserved event IDs to rainfall groups fail-closed.

    The legacy Formal-Blind adapter stores event IDs.  The old implementation
    tried to recover rainfall SHA only from the rainfall table/source rows and
    could therefore report ``reserved_event_count=36`` with
    ``reserved_rainfall_group_count=0``.  The event inventory is now an explicit
    third authority for the event->rainfall mapping.
    """
    adapter = root / "outputs/rainfall_library_v8_storage_variablepump/rainfall_event_table.formal_adapter.json"
    table = adapter.with_name("rainfall_event_table.csv")
    events: set[str] = set()
    groups: set[str] = set()
    audit: dict[str, Any] = {
        "adapter_path": str(adapter),
        "adapter_found": adapter.exists(),
        "rainfall_table_path": str(table),
        "rainfall_table_found": table.exists(),
    }
    if adapter.exists():
        payload = json.loads(adapter.read_text(encoding="utf-8"))
        split = text(payload.get("split", "")).casefold()
        if any(token in split for token in ("blind", "reserved", "challenge")):
            events.update(str(x) for x in payload.get("event_ids", []) if text(x))
        audit.update({"adapter_split": payload.get("split"), "reserved_event_count": len(events)})

    if table.exists() and events:
        groups.update(_groups_for_events(pd.read_csv(table, low_memory=False), events))
    groups.update(_groups_for_events(inventory, events))
    groups.update(_groups_for_events(source_all, events))

    audit["reserved_rainfall_group_count"] = len(groups)
    audit["reserved_event_ids_without_rainfall_group"] = max(0, len(events) - len(groups))
    return events, groups, audit


def _contamination_audit(source_all: pd.DataFrame, contamination: pd.DataFrame, ledger: pd.DataFrame) -> dict[str, Any]:
    def n_groups(frame: pd.DataFrame) -> int:
        if frame.empty or "rainfall_group_key" not in frame.columns:
            return 0
        return int(frame.loc[frame.rainfall_group_key.astype(str).ne(""), "rainfall_group_key"].astype(str).nunique())

    by_source: dict[str, int] = {}
    if not contamination.empty and "source_id" in contamination.columns:
        for sid, grp in contamination.groupby("source_id", sort=True):
            by_source[str(sid)] = n_groups(grp)
    roles = ledger.formal_f2_role.astype(str) if not ledger.empty else pd.Series(dtype=str)
    return {
        "all_source_rows": int(len(source_all)),
        "all_source_rainfall_groups": n_groups(source_all),
        "historical_contamination_rows": int(len(contamination)),
        "historical_contamination_rainfall_groups": n_groups(contamination),
        "historical_contamination_groups_by_source": by_source,
        "ledger_unused_untouched_groups": int(roles.eq("unused_untouched").sum()),
        "ledger_auxiliary_historical_groups": int(roles.eq("auxiliary").sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--registry", type=Path, default=PROJECT_ROOT / "configs/v42_formal_source_registry_f2.yaml")
    ap.add_argument(
        "--step1-window-manifest",
        type=Path,
        default=PROJECT_ROOT / "outputs/project6_dual_reference_v4/final_v4/v42_paper/step1_gat/dataset/step1_window_manifest.parquet",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/prepare",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--step1-validation-fraction", type=float, default=0.15)
    ap.add_argument("--min-train-rainfall-groups", type=int, default=FORMAL_TRAIN_MIN_GROUPS)
    ap.add_argument("--calibration-groups", type=int, default=DEFAULT_COUNTS["calibration"])
    ap.add_argument("--locked-groups", type=int, default=DEFAULT_COUNTS["locked_validation"])
    ap.add_argument("--challenge-groups", type=int, default=DEFAULT_COUNTS["challenge"])
    ap.add_argument("--blind-groups", type=int, default=DEFAULT_COUNTS["formal_blind"])
    args = ap.parse_args()

    registry = load_registry(args.registry)
    source_all, source_audits = manifest_source_rows(args.project_root, registry)
    inventory, inventory_path = _inventory(args.project_root, registry)
    reserved_events, reserved_groups, reserved_audit = _reserved(args.project_root, source_all, inventory)

    # Contamination is derived from actual historical reveal/training status, not
    # from mere eligibility.  This is the bug fixed by this revision.
    contamination = _historical_contamination(source_all)

    ledger = build_event_ledger(
        contamination,
        inventory=inventory,
        historical_reserved_groups=sorted(reserved_groups),
        seed=args.seed,
        evaluation_counts={
            "calibration": args.calibration_groups,
            "locked_validation": args.locked_groups,
            "challenge": args.challenge_groups,
            "formal_blind": args.blind_groups,
        },
    )
    assert_zero_split_overlap(ledger)

    # Training metadata must be built from the full registry population after the
    # ledger is frozen.  formal_step2_metadata_pool admits only role=train rows,
    # so evaluation/unused rainfalls cannot leak into Step2.
    step2 = formal_step2_metadata_pool(source_all, ledger)

    if not args.step1_window_manifest.exists():
        raise FileNotFoundError(args.step1_window_manifest)
    base = read_table(args.step1_window_manifest)
    if "rainfall_sha256" in base:
        rainfall_sha = base.rainfall_sha256.fillna("").astype(str).str.strip()
        use = rainfall_sha.ne("")
        if use.any():
            base["legacy_split_group_key"] = base.split_group_key.astype(str)
            base.loc[use, "split_group_key"] = rainfall_sha.loc[use]
    provisional = explicit_step1_roles(
        base,
        ledger,
        validation_fraction=args.step1_validation_fraction,
        split_seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Keep the full registry rows for the subsequent Step1 expansion.  The
    # Step1 builder uses the frozen ledger to exclude evaluation/unused rainfalls
    # from model training; writing only contamination rows here previously made
    # source eligibility and contamination inseparable.
    _write(source_all, args.output_dir / "FORMAL_F2_SOURCE_ROWS.parquet")
    _write(contamination, args.output_dir / "FORMAL_F2_CONTAMINATION_ROWS.parquet")
    _write(ledger, args.output_dir / "FORMAL_F2_EVENT_LEDGER.csv")
    _write(provisional, args.output_dir / "FORMAL_F2_STEP1_WINDOW_MANIFEST.parquet")
    _write(step2, args.output_dir / "FORMAL_F2_STEP2_METADATA_POOL.parquet")
    pd.DataFrame(source_audits).to_csv(args.output_dir / "FORMAL_F2_SOURCE_AUDIT.csv", index=False)

    summary = pool_summary(provisional, step2, ledger)
    summary.update(
        {
            "status": "pass",
            "development_only": False,
            "formal_mainline_authorized": False,
            "registry_path": str(args.registry),
            "event_inventory_path": inventory_path,
            "reserved_audit": reserved_audit,
            "contamination_audit": _contamination_audit(source_all, contamination, ledger),
            "source_count": len(registry.get("sources", {})),
            "resolved_manifest_count": sum(1 for x in source_audits if x.get("status") == "read"),
            "required_min_train_rainfall_groups": args.min_train_rainfall_groups,
            "raw_readmission_pending_rows": int(
                step2.get("raw_readmission_pending", pd.Series(dtype=bool)).astype(bool).sum()
            )
            if not step2.empty
            else 0,
            "step1_provisional_only": True,
        }
    )

    reasons: list[str] = []
    warnings: list[str] = []
    if summary["formal_train_ledger_groups"] < args.min_train_rainfall_groups:
        reasons.append("formal_train_ledger_groups_below_minimum")
    if summary["step1_target_train_groups"] < args.min_train_rainfall_groups:
        warnings.append("provisional_step1_target_groups_below_minimum_expand_from_structured_physical_sources")
    if summary["step2_train_rainfall_groups"] < args.min_train_rainfall_groups:
        reasons.append("step2_metadata_groups_below_minimum_before_raw_readmission")
    if any(int(v) for v in split_overlap_matrix(ledger).values()):
        reasons.append("rainfall_split_overlap")

    required_counts = [
        ("calibration", args.calibration_groups),
        ("locked_validation", args.locked_groups),
        ("challenge", args.challenge_groups),
        ("formal_blind", args.blind_groups),
    ]
    for role, required in required_counts:
        actual = int(summary["evaluation_group_counts"].get(role, 0))
        if actual < required:
            reasons.append(f"{role}_untouched_group_shortfall:{actual}<{required}")

    if reserved_events and not reserved_groups:
        reasons.append("historical_reserved_events_could_not_map_to_rainfall_groups")
    if reasons:
        summary["status"] = "fail"
    summary["reasons"] = reasons
    summary["warnings"] = warnings

    (args.output_dir / "FORMAL_F2_PREPARE_AUDIT.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    for role in ("train", "calibration", "locked_validation", "challenge", "formal_blind"):
        (args.output_dir / f"{role}_groups.json").write_text(
            json.dumps(
                {
                    "formal_generation_id": FORMAL_GENERATION_ID,
                    "groups": sorted(
                        ledger.loc[ledger.formal_f2_role.eq(role), "rainfall_group_key"].astype(str)
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if not reasons else 3


if __name__ == "__main__":
    raise SystemExit(main())
