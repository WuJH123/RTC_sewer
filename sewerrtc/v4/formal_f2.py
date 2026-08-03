"""Project6 V4.2 Formal F2 data utilities.

Current scientific split policy
-------------------------------
The split authority is the rainfall SHA/fingerprint.  For the current V4.2
execution generation, a rainfall group is either used for model development or
held out for Calibration/Locked/Challenge/Test.  Historical labels from older
experiments are retained as lineage metadata but do not by themselves make a
rainfall unusable.

The invariant that matters for this generation is zero rainfall-group overlap
between current model development and current held-out evaluation roles.  This
is the familiar train/validation/test principle applied at independent-rainfall
level rather than row level.

Other invariants:
- Step1 target roles are explicit (not inferred from folder/domain names);
- Step2 admission is source-specific and fail-closed;
- old split labels such as ``formal_blind`` are metadata only; they cannot
  silently override the current frozen ledger;
- missing hydraulic targets are never fabricated.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from sewerrtc.v4.train_v4_loader import ACCEPTANCE_GATE_COLUMNS, compute_acceptance

FORMAL_GENERATION_ID = "PROJECT6_V42_FORMAL_F2"
FORMAL_TRAIN_MIN_GROUPS = 65
DEFAULT_COUNTS = {
    "calibration": 12,
    "locked_validation": 16,
    "challenge": 12,
    "formal_blind": 24,
}
RAIN_COLUMNS = (
    "rainfall_sha256",
    "rainfall_fingerprint",
    "rainfall_group_key",
    "split_group_key",
)
STATE_HASH_COLUMNS = ("prefix_state_hash", "prefix_sha256", "state_key")
ACTION_HASH_COLUMNS = (
    "actual_schedule_sha256",
    "candidate_action_sha",
    "action_readback_sha256",
    "projected_schedule_sha256",
)
DETAIL_COLUMNS = (
    "detail_path",
    "source_detail_path",
    "candidate_detail_path",
    "source_detail_path_candidate",
)
LEGACY_RESERVED_TOKENS = ("formal_blind", "challenge", "reserved_evaluation")


def text(v: Any) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip()


def yes(v: Any) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return bool(v)
    if isinstance(v, (float, np.floating)):
        return bool(np.isfinite(v) and v != 0.0)
    return text(v).casefold() in {"true", "1", "1.0", "yes", "y", "t"}


def read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    return (
        pd.read_parquet(p)
        if p.suffix.lower() == ".parquet"
        else pd.read_csv(p, low_memory=False)
    )


def sha256_file(path: str | Path, chunk_bytes: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def canonical_rain_group(
    row: Mapping[str, Any], *, allow_event_fallback: bool = False
) -> str:
    for c in RAIN_COLUMNS:
        v = text(row.get(c, ""))
        if v:
            return v
    if allow_event_fallback:
        v = text(row.get("event_id", row.get("rainfall_event_id", "")))
        return f"event:{v}" if v else ""
    return ""


def event_id_of(row: Mapping[str, Any]) -> str:
    return text(row.get("event_id", row.get("rainfall_event_id", "")))


def checkpoint_of(row: Mapping[str, Any]) -> float:
    for c in ("checkpoint_min", "elapsed_min", "anchor_min"):
        if c in row:
            v = pd.to_numeric(row.get(c), errors="coerce")
            if pd.notna(v):
                return float(v)
    raw = text(row.get("checkpoint_id", ""))
    if "__" in raw:
        v = pd.to_numeric(raw.rsplit("__", 1)[-1], errors="coerce")
        if pd.notna(v):
            return float(v)
    return float("nan")


def state_key_of(row: Mapping[str, Any]) -> str:
    for c in STATE_HASH_COLUMNS:
        v = text(row.get(c, ""))
        if v:
            return v
    rain = canonical_rain_group(row, allow_event_fallback=True)
    event = event_id_of(row)
    cp = checkpoint_of(row)
    net = text(row.get("network_sha256", row.get("physical_sha256", "")))
    if not rain or not np.isfinite(cp):
        return ""
    return hashlib.sha256(f"{rain}|{event}|{cp:.6f}|{net}".encode()).hexdigest()


def action_key_of(row: Mapping[str, Any]) -> str:
    for c in ACTION_HASH_COLUMNS:
        v = text(row.get(c, ""))
        if v:
            return v
    for c in (
        "projected_schedule_json",
        "requested_schedule_json",
        "actual_schedule_json",
    ):
        raw = row.get(c)
        if not text(raw):
            continue
        try:
            a = np.asarray(json.loads(str(raw)), dtype=np.float64)
            if a.ndim == 1 and a.size % 36 == 0:
                a = a.reshape(-1, 36)
            if a.ndim == 2 and a.shape[1] == 36:
                return hashlib.sha256(
                    np.ascontiguousarray(a[:3]).tobytes()
                ).hexdigest()
        except Exception:
            pass
    return ""


def load_registry(path: str | Path) -> dict[str, Any]:
    raw = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        obj = yaml.safe_load(raw)
    except Exception:
        obj = json.loads(raw)
    if not isinstance(obj, dict) or not isinstance(obj.get("sources"), Mapping):
        raise ValueError("formal F2 registry must contain sources mapping")
    return obj


def resolve_source_files(
    project_root: str | Path, spec: Mapping[str, Any]
) -> list[Path]:
    root, found = Path(project_root), {}
    for key in ("paths", "globs"):
        vals = spec.get(key, []) or []
        vals = [vals] if isinstance(vals, str) else vals
        for pattern in vals:
            p = root / str(pattern)
            hits = (
                root.glob(str(pattern))
                if any(ch in str(pattern) for ch in "*?[]")
                else ([p] if p.is_file() else [])
            )
            for hit in hits:
                if hit.is_file():
                    found[str(hit.resolve()).casefold()] = hit.resolve()
    return sorted(found.values(), key=lambda p: str(p).casefold())


def _split_text(row: Mapping[str, Any]) -> str:
    return " ".join(
        text(row.get(c, ""))
        for c in ("split", "assigned_split", "formal_split", "source_role", "role")
    ).casefold()


def row_is_reserved(row: Mapping[str, Any]) -> bool:
    """Return whether an *old* source row carries a reserved-evaluation label.

    This is lineage metadata only.  Current-generation admission/splitting is
    controlled by the frozen F2 ledger, not by this historical label.
    """
    s = _split_text(row)
    return any(t in s for t in LEGACY_RESERVED_TOKENS) or text(
        row.get("source_role", "")
    ).casefold() == "reserved_evaluation"


def source_acceptance_mask(
    frame: pd.DataFrame, source_id: str, spec: Mapping[str, Any]
) -> np.ndarray:
    if frame.empty:
        return np.zeros(0, dtype=bool)
    policy = text(spec.get("step2_admission", "none")).casefold()
    gates_present = set(ACCEPTANCE_GATE_COLUMNS).issubset(frame.columns)
    if policy in {"acceptance_14", "current_14_gate"}:
        if not gates_present:
            return np.zeros(len(frame), dtype=bool)
        mask = np.asarray(compute_acceptance(frame), dtype=bool).copy()
    elif policy == "pilot_v3_training":
        if "eligible_for_training" not in frame.columns:
            return np.zeros(len(frame), dtype=bool)
        eligible = frame["eligible_for_training"].map(yes).to_numpy(dtype=bool)
        mask = (
            np.logical_and(
                eligible, np.asarray(compute_acceptance(frame), dtype=bool)
            )
            if gates_present
            else np.zeros(len(frame), dtype=bool)
        )
    elif policy in {"raw_readmission_required", "none", ""}:
        return np.zeros(len(frame), dtype=bool)
    else:
        raise ValueError(f"unsupported Step2 admission for {source_id}: {policy}")
    # Important: old split labels do not remove otherwise valid rows here.
    return np.asarray(mask, dtype=bool)


def manifest_source_rows(
    project_root: str | Path, registry: Mapping[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows, audits = [], []
    for sid, raw_spec in registry["sources"].items():
        spec, seen = dict(raw_spec or {}), set()
        for path in resolve_source_files(project_root, spec):
            if path.suffix.lower() not in {".csv", ".parquet"}:
                continue
            fsha = sha256_file(path)
            if fsha in seen:
                audits.append(
                    {
                        "source_id": sid,
                        "path": str(path),
                        "status": "duplicate_manifest_content_skipped",
                        "manifest_sha256": fsha,
                    }
                )
                continue
            seen.add(fsha)
            try:
                df = read_table(path)
            except Exception as exc:
                audits.append(
                    {
                        "source_id": sid,
                        "path": str(path),
                        "status": f"unreadable:{type(exc).__name__}",
                    }
                )
                continue
            policy = text(spec.get("step2_admission", "none")).casefold()
            gates_present = set(ACCEPTANCE_GATE_COLUMNS).issubset(df.columns)
            mask = (
                source_acceptance_mask(df, str(sid), spec)
                if spec.get("formal_step2_allowed", False)
                else np.zeros(len(df), bool)
            )
            raw_pending_count = 0
            for pos, (_, r) in enumerate(df.iterrows()):
                d = r.to_dict()
                pilot_pending = (
                    policy == "pilot_v3_training"
                    and yes(d.get("eligible_for_training", False))
                    and not gates_present
                )
                raw_pending = policy == "raw_readmission_required" or pilot_pending
                if raw_pending:
                    raw_pending_count += 1
                rows.append(
                    {
                        "source_id": str(sid),
                        "source_manifest": str(path),
                        "source_manifest_sha256": fsha,
                        "source_row_number": pos,
                        "event_id": event_id_of(d),
                        "rainfall_group_key": canonical_rain_group(d),
                        "checkpoint_min": checkpoint_of(d),
                        "state_key": state_key_of(d),
                        "action_key": action_key_of(d),
                        "formal_step1_allowed": bool(
                            spec.get("formal_step1_allowed", False)
                        ),
                        "formal_step2_allowed": bool(
                            spec.get("formal_step2_allowed", False)
                        ),
                        "auxiliary_only": bool(spec.get("auxiliary_only", False)),
                        "historically_revealed": bool(
                            spec.get("historically_revealed", True)
                        ),
                        "legacy_reserved_label": bool(row_is_reserved(d)),
                        "step2_accepted_from_manifest": bool(mask[pos]),
                        "step2_admission_policy": policy,
                        "raw_readmission_required": bool(raw_pending),
                        "source_split": _split_text(d),
                        "case_id": text(
                            d.get("case_id", d.get("candidate_id", ""))
                        ),
                        "detail_path": next(
                            (
                                text(d.get(c, ""))
                                for c in DETAIL_COLUMNS
                                if text(d.get(c, ""))
                            ),
                            "",
                        ),
                    }
                )
            audits.append(
                {
                    "source_id": sid,
                    "path": str(path),
                    "status": "read",
                    "rows": len(df),
                    "step2_accepted_rows": int(mask.sum()),
                    "raw_readmission_candidate_rows": int(raw_pending_count),
                    "unique_event_ids": int(df["event_id"].astype(str).nunique())
                    if "event_id" in df
                    else 0,
                    "unique_rainfall_groups": len(
                        {
                            g
                            for g in (
                                canonical_rain_group(x)
                                for x in df.to_dict("records")
                            )
                            if g
                        }
                    ),
                    "manifest_sha256": fsha,
                }
            )
    return pd.DataFrame(rows), audits


def _rank(groups: Iterable[str], salt: str) -> list[str]:
    return sorted(
        {text(g) for g in groups if text(g)},
        key=lambda g: (hashlib.sha256(f"{salt}:{g}".encode()).hexdigest(), g),
    )


def build_event_ledger(
    source_rows: pd.DataFrame,
    *,
    inventory: pd.DataFrame | None = None,
    historical_reserved_groups: Iterable[str] = (),
    seed: int = 42,
    evaluation_counts: Mapping[str, int] | None = None,
    minimum_train_groups: int = FORMAL_TRAIN_MIN_GROUPS,
) -> pd.DataFrame:
    """Freeze one current-generation rainfall-group train/holdout partition.

    Historical use is retained as metadata but does not exclude a group.  Held-
    out roles are selected first, preferring inventory groups that are not part
    of the current Step2 training-capable population.  If more evaluation groups
    are needed, training-capable groups may be held out only while preserving the
    requested minimum number of training groups.
    """
    counts = dict(DEFAULT_COUNTS)
    if evaluation_counts:
        counts.update({k: int(v) for k, v in evaluation_counts.items()})

    hist = set(
        source_rows.get("rainfall_group_key", pd.Series(dtype=str)).astype(str)
    )
    hist.discard("")
    historical_reserved = {
        text(x) for x in historical_reserved_groups if text(x)
    }
    allowed = source_rows.get(
        "formal_step2_allowed", pd.Series(False, index=source_rows.index)
    ).astype(bool)
    admitted = source_rows.get(
        "step2_accepted_from_manifest", pd.Series(False, index=source_rows.index)
    ).astype(bool)
    pending = source_rows.get(
        "raw_readmission_required", pd.Series(False, index=source_rows.index)
    ).astype(bool)
    train_capable = set(
        source_rows.loc[
            allowed & (admitted | pending), "rainfall_group_key"
        ].astype(str)
    )
    train_capable.discard("")

    inv_rows = []
    if inventory is not None:
        for d in inventory.to_dict("records"):
            g = canonical_rain_group(d)
            if g:
                inv_rows.append(
                    {
                        "rainfall_group_key": g,
                        "inventory_event_id": event_id_of(d),
                        "rainfall_family": text(
                            d.get(
                                "rainfall_family",
                                d.get("pattern", d.get("rainfall_pattern", "")),
                            )
                        ),
                        "duration_min": pd.to_numeric(
                            d.get(
                                "duration_min",
                                d.get("rainfall_duration_min", np.nan),
                            ),
                            errors="coerce",
                        ),
                    }
                )
    inv = (
        pd.DataFrame(inv_rows).drop_duplicates("rainfall_group_key")
        if inv_rows
        else pd.DataFrame()
    )
    inv_groups = (
        set(inv["rainfall_group_key"].astype(str)) if not inv.empty else set()
    )

    # Evaluation requires a rainfall definition, so the inventory is the
    # selection authority. Prefer groups that do not consume current training
    # capacity. Historical exposure is intentionally not part of the gate.
    outside_train = _rank(inv_groups - train_capable, f"f2-eval-outside-{seed}")
    spare_count = max(0, len(train_capable) - int(minimum_train_groups))
    spare_train = _rank(inv_groups & train_capable, f"f2-eval-spare-{seed}")[
        :spare_count
    ]
    remaining = outside_train + spare_train

    role: dict[str, str] = {}
    selected_eval: set[str] = set()
    for name in ("calibration", "locked_validation", "challenge", "formal_blind"):
        chosen, remaining = remaining[: counts[name]], remaining[counts[name] :]
        role.update({g: name for g in chosen})
        selected_eval.update(chosen)

    train = train_capable - selected_eval
    role.update({g: "train" for g in train})

    inv_map = (
        inv.set_index("rainfall_group_key").to_dict("index") if not inv.empty else {}
    )
    grouped = source_rows.groupby("rainfall_group_key") if not source_rows.empty else None
    out = []
    for g in sorted(hist | inv_groups | historical_reserved):
        grp = (
            grouped.get_group(g)
            if grouped is not None and g in grouped.groups
            else pd.DataFrame()
        )
        invd = inv_map.get(g, {})
        final_role = role.get(
            g, "auxiliary" if g in hist else "unused_holdout"
        )
        revealed = (
            bool(grp.get("historically_revealed", pd.Series(dtype=bool)).astype(bool).any())
            if not grp.empty
            else False
        )
        legacy_reserved = (
            bool(grp.get("legacy_reserved_label", pd.Series(dtype=bool)).astype(bool).any())
            if not grp.empty
            else False
        )
        out.append(
            {
                "formal_generation_id": FORMAL_GENERATION_ID,
                "rainfall_group_key": g,
                "rainfall_sha256": g,
                "formal_f2_role": final_role,
                "current_generation_holdout": final_role
                in {"calibration", "locked_validation", "challenge", "formal_blind"},
                "current_generation_training_seen": g in train,
                "model_training_seen": g in train,
                "historically_seen": g in hist,
                "historically_revealed": revealed,
                "historically_reserved": g in historical_reserved or legacy_reserved,
                "historical_status_is_split_gate": False,
                "source_datasets": json.dumps(
                    sorted(set(grp["source_id"].astype(str))) if not grp.empty else []
                ),
                "historical_event_ids": json.dumps(
                    sorted(
                        {
                            x
                            for x in grp.get(
                                "event_id", pd.Series(dtype=str)
                            ).astype(str)
                            if x
                        }
                    )
                ),
                "inventory_event_id": text(invd.get("inventory_event_id", "")),
                "rainfall_family": text(invd.get("rainfall_family", "")),
                "duration_min": invd.get("duration_min", np.nan),
            }
        )
    return (
        pd.DataFrame(out)
        .sort_values(["formal_f2_role", "rainfall_group_key"], kind="mergesort")
        .reset_index(drop=True)
    )


def split_overlap_matrix(ledger: pd.DataFrame) -> dict[str, int]:
    roles = [
        "train",
        "calibration",
        "locked_validation",
        "challenge",
        "formal_blind",
    ]
    sets = {
        r: set(
            ledger.loc[
                ledger.formal_f2_role.astype(str).eq(r), "rainfall_group_key"
            ].astype(str)
        )
        for r in roles
    }
    return {
        f"{a}__{b}": len(sets[a] & sets[b])
        for i, a in enumerate(roles)
        for b in roles[i + 1 :]
    }


def assert_zero_split_overlap(ledger: pd.DataFrame) -> None:
    bad = {k: v for k, v in split_overlap_matrix(ledger).items() if v}
    if bad:
        raise RuntimeError(f"Formal F2 rainfall split overlap: {bad}")


def explicit_step1_roles(
    window_manifest: pd.DataFrame,
    ledger: pd.DataFrame,
    *,
    validation_fraction: float = 0.15,
    split_seed: int = 42,
) -> pd.DataFrame:
    if "split_group_key" not in window_manifest:
        raise KeyError("Step1 manifest missing split_group_key")
    role_map = dict(
        zip(
            ledger.rainfall_group_key.astype(str),
            ledger.formal_f2_role.astype(str),
        )
    )
    out = window_manifest.copy()
    g = out.split_group_key.astype(str)
    f2 = g.map(role_map).fillna("auxiliary")
    candidates = sorted(set(g[f2.eq("train")]))
    ranked = _rank(candidates, f"f2-step1-{split_seed}")
    n_val = (
        max(1, int(round(validation_fraction * len(ranked))))
        if len(ranked) >= 2
        else 0
    )
    val, train = set(ranked[:n_val]), set(ranked[n_val:])
    out["formal_generation_id"] = FORMAL_GENERATION_ID
    out["formal_split"], out["step1_domain_role"] = (
        "auxiliary",
        "auxiliary_pretrain",
    )
    out.loc[
        g.isin(train), ["formal_split", "step1_domain_role"]
    ] = ["train", "target_formal"]
    out.loc[g.isin(val), ["formal_split", "step1_domain_role"]] = [
        "validation",
        "target_formal",
    ]
    for r in ("calibration", "locked_validation", "challenge", "formal_blind"):
        m = f2.eq(r)
        out.loc[m, "formal_split"] = r
        out.loc[m, "step1_domain_role"] = (
            "target_formal_calibration" if r == "calibration" else "reserved_blind"
        )
    return out


def formal_step2_metadata_pool(
    source_rows: pd.DataFrame, ledger: pd.DataFrame
) -> pd.DataFrame:
    if source_rows.empty:
        return source_rows.copy()
    role = dict(
        zip(
            ledger.rainfall_group_key.astype(str),
            ledger.formal_f2_role.astype(str),
        )
    )
    f = source_rows.copy()
    f["formal_f2_role"] = f.rainfall_group_key.astype(str).map(role).fillna(
        "excluded"
    )
    admitted = f.step2_accepted_from_manifest.astype(bool)
    pending = f.raw_readmission_required.astype(bool)
    base = (
        f.formal_step2_allowed.astype(bool)
        & f.formal_f2_role.eq("train")
        & (admitted | pending)
        & f.rainfall_group_key.astype(str).ne("")
    )
    identified_authorized = (
        admitted & f.state_key.astype(str).ne("") & f.action_key.astype(str).ne("")
    )
    identifiable_pending = pending & f.case_id.astype(str).ne("")
    out = f.loc[base & (identified_authorized | identifiable_pending)].copy()
    out["formal_generation_id"] = FORMAL_GENERATION_ID
    out["training_admission_authorized"] = out.step2_accepted_from_manifest.astype(bool)
    out["raw_readmission_pending"] = out.raw_readmission_required.astype(bool)
    out["_dedup_key"] = np.where(
        out.raw_readmission_pending.astype(bool),
        "raw|"
        + out.source_manifest.astype(str)
        + "|"
        + out.source_row_number.astype(str),
        "accepted|"
        + out.rainfall_group_key.astype(str)
        + "|"
        + out.state_key.astype(str)
        + "|"
        + out.action_key.astype(str),
    )
    return (
        out.sort_values(
            [
                "rainfall_group_key",
                "source_id",
                "source_manifest",
                "source_row_number",
            ],
            kind="mergesort",
        )
        .drop_duplicates("_dedup_key", keep="first")
        .drop(columns=["_dedup_key"])
        .reset_index(drop=True)
    )


def pool_summary(
    step1: pd.DataFrame, step2: pd.DataFrame, ledger: pd.DataFrame
) -> dict[str, Any]:
    train1 = step1[
        (step1.step1_domain_role.astype(str) == "target_formal")
        & (step1.formal_split.astype(str) == "train")
    ]
    val1 = step1[
        (step1.step1_domain_role.astype(str) == "target_formal")
        & (step1.formal_split.astype(str) == "validation")
    ]
    return {
        "formal_generation_id": FORMAL_GENERATION_ID,
        "split_policy": "current_generation_rainfall_group_holdout",
        "historical_status_is_split_gate": False,
        "step1_rows": len(step1),
        "step1_target_train_groups": int(
            train1.split_group_key.astype(str).nunique()
        ),
        "step1_target_validation_groups": int(
            val1.split_group_key.astype(str).nunique()
        ),
        "step1_auxiliary_groups": int(
            step1.loc[
                step1.step1_domain_role.astype(str).eq("auxiliary_pretrain"),
                "split_group_key",
            ]
            .astype(str)
            .nunique()
        ),
        "step2_metadata_rows": len(step2),
        "step2_train_rainfall_groups": int(
            step2.rainfall_group_key.astype(str).nunique()
        )
        if not step2.empty
        else 0,
        "formal_train_ledger_groups": int(
            ledger.formal_f2_role.astype(str).eq("train").sum()
        ),
        "evaluation_group_counts": {
            r: int(ledger.formal_f2_role.astype(str).eq(r).sum())
            for r in (
                "calibration",
                "locked_validation",
                "challenge",
                "formal_blind",
            )
        },
        "split_overlap": split_overlap_matrix(ledger),
    }
