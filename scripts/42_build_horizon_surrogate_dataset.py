from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np

from sewerrtc.control.horizon_action_features import ACTION_FEATURE_COLUMNS
from sewerrtc.control.horizon_rollout import build_horizon_samples_from_detail
from sewerrtc.control.actuator_scope import select_actuators_for_scope
from sewerrtc.data.dataset_fingerprint import source_file_fingerprint
from sewerrtc.data.gat_feature_cache import gat_feature_cache_path
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config, resolve_gat_model_path
from sewerrtc.models.temporal_graph_surrogate import TARGET_COLUMNS


def _detail_event_id(path: Path) -> str:
    name = path.name
    suffix = "_detail.csv"
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    event_id, sep, _policy = name.rpartition("__")
    return event_id if sep else name


def _detail_policy_id(path: Path) -> str:
    name = path.name.removesuffix("_detail.csv")
    _event_id, sep, policy_id = name.rpartition("__")
    return policy_id if sep else ""


def _collect_detail_files(
    root: Path,
    max_files: int = 0,
    source_scope: str = "all",
    allowed_event_ids: set[str] | None = None,
    cfg: dict | None = None,
) -> list[Path]:
    source_scope = str(source_scope or "all")
    cfg_files: list[Path] = []
    if cfg is not None:
        if source_scope in {"generic_trajectories", "all"}:
            try:
                cfg_files.extend(sorted((cfg_path(cfg, "outputs.data_bank_train") / "trajectories").glob("*_detail.csv")))
            except Exception:
                pass
        if source_scope in {"closed_loop", "all"}:
            try:
                closed_root = cfg_path(cfg, "outputs.closed_loop")
                cfg_files.extend(sorted(closed_root.glob("formal/*/proposed/*__proposed_detail.csv")))
                cfg_files.extend(sorted(closed_root.glob("formal/*/baselines/*/*__*_detail.csv")))
                cfg_files.extend(sorted(closed_root.glob("debug/*/proposed/*__proposed_detail.csv")))
                cfg_files.extend(sorted(closed_root.glob("debug/*/baselines/*/*__*_detail.csv")))
            except Exception:
                pass
    if cfg is not None and source_scope in {"generic_trajectories", "closed_loop"}:
        files = cfg_files
        unique = []
        seen = set()
        for p in files:
            key = str(p.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(p)
        if allowed_event_ids is not None:
            allowed = {str(x) for x in allowed_event_ids}
            unique = [p for p in unique if _detail_event_id(p) in allowed]
        if max_files:
            unique = unique[: int(max_files)]
        return unique
    if source_scope == "generic_trajectories":
        patterns = [
            "data_bank_train_paired_no_controls/trajectories/*_detail.csv",
        ]
    elif source_scope == "closed_loop":
        patterns = [
            "closed_loop_paired_no_controls/formal/*/proposed/*__proposed_detail.csv",
            "closed_loop_paired_no_controls/formal/*/baselines/*/*__*_detail.csv",
            "closed_loop_paired_no_controls/debug/*/proposed/*__proposed_detail.csv",
            "closed_loop_paired_no_controls/debug/*/baselines/*/*__*_detail.csv",
        ]
    else:
        patterns = [
            "closed_loop_paired_no_controls/formal/*/proposed/*__proposed_detail.csv",
            "closed_loop_paired_no_controls/formal/*/baselines/*/*__*_detail.csv",
            "closed_loop_paired_no_controls/debug/*/proposed/*__proposed_detail.csv",
            "closed_loop_paired_no_controls/debug/*/baselines/*/*__*_detail.csv",
            "data_bank_train_paired_no_controls/trajectories/*_detail.csv",
        ]
    files: list[Path] = []
    files.extend(cfg_files)
    for pat in patterns:
        files.extend(sorted((root / "outputs").glob(pat)))
    unique = []
    seen = set()
    for p in files:
        key = str(p.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    if allowed_event_ids is not None:
        allowed = {str(x) for x in allowed_event_ids}
        unique = [p for p in unique if _detail_event_id(p) in allowed]
    if max_files:
        unique = unique[: int(max_files)]
    return unique


def _safe_write_frame(df: pd.DataFrame, path: Path) -> tuple[Path, str, str]:
    ensure_dir(path.parent)
    try:
        df.to_parquet(path, index=False)
        return path, "parquet", ""
    except Exception as exc:
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path, "csv", repr(exc)


def _classify_write_issue(path: Path, error: str, chunk_index: int | None = None) -> dict:
    text = str(error or "")
    category = "write_fallback"
    severity = "warning"
    if "Unable to find a usable engine" in text and ("pyarrow" in text or "fastparquet" in text):
        category = "parquet_engine_missing_csv_fallback"
    issue = {
        "detail_file": str(path),
        "error": text,
        "category": category,
        "severity": severity,
    }
    if chunk_index is not None:
        issue["chunk_index"] = int(chunk_index)
    return issue


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _chunk_path(chunk_dir: Path, index: int) -> Path:
    return chunk_dir / f"chunk_{index:05d}.parquet"


def _preferred_existing_chunk_path(default_chunk_path: Path) -> Path:
    csv_path = default_chunk_path.with_suffix(".csv")
    if csv_path.exists():
        return csv_path
    return default_chunk_path if default_chunk_path.exists() else csv_path


def _chunk_event_filter_ok(df: pd.DataFrame, allowed_event_ids: set[str]) -> tuple[bool, str]:
    if "event_id" not in df:
        return False, "missing_event_id_column"
    observed = set(df["event_id"].dropna().astype(str).unique())
    stale = sorted(observed - {str(x) for x in allowed_event_ids})
    if stale:
        preview = ",".join(stale[:5])
        return False, f"stale_event_ids={preview}"
    return True, ""


def _chunk_source_fingerprint_matches(recorded: object, source_files: list[Path]) -> bool:
    text = str(recorded or "").strip()
    return bool(text) and text == source_file_fingerprint(source_files)


def _write_chunk_manifest(records: list[dict], manifest_path: Path) -> None:
    pd.DataFrame(records).to_csv(manifest_path, index=False)


def _apply_gat_features(
    samples: pd.DataFrame,
    detail_path: Path,
    cache_dir: Path | None,
    history_steps: int,
    require: bool,
    verify_source_fingerprint: bool = True,
) -> pd.DataFrame:
    if cache_dir is None:
        if require:
            raise FileNotFoundError("GAT reconstructed feature cache directory was not configured")
        return samples
    path = gat_feature_cache_path(cache_dir, detail_path)
    if not path.exists():
        if require:
            raise FileNotFoundError(f"Missing GAT reconstructed features: {path}")
        return samples
    data = np.load(path, allow_pickle=False)
    if verify_source_fingerprint:
        expected = source_file_fingerprint([detail_path])
        if str(data["source_fingerprint"].item()) != expected:
            raise ValueError(f"Stale GAT feature cache fingerprint for {detail_path}")
    starts = pd.to_numeric(samples["row_index"], errors="raise").to_numpy(np.int64)
    if starts.size and int(starts.max()) >= int(data["row_count"].item()):
        raise ValueError(f"GAT feature cache row mismatch for {detail_path}")
    for col in (
        "current_depth_mean",
        "current_depth_p95",
        "current_depth_max",
        "priority_depth_mean",
        "priority_depth_max",
    ):
        samples[col] = np.asarray(data[col], dtype=np.float32)[starts]
    priority_max = np.asarray(data["priority_depth_max"], dtype=np.float32)
    history_start = np.maximum(0, starts - int(history_steps) + 1)
    samples["priority_depth_trend"] = priority_max[starts] - priority_max[history_start]
    samples["state_feature_source"] = "gat_reconstructed_sparse_sensors"
    return samples


def _build_one_detail(args: tuple) -> tuple[pd.DataFrame | None, dict | None]:
    (
        p, priority_nodes, horizon_steps, history_steps, dt_sec, stride, chunk_index,
        actuators, priority_to_actuators, gat_cache_dir, require_gat_features, verify_source_fingerprint,
        reference_detail_path,
    ) = args
    try:
        df = build_horizon_samples_from_detail(
            p,
            priority_nodes,
            horizon_steps=horizon_steps,
            history_steps=history_steps,
            dt_sec=dt_sec,
            stride=stride,
            actuators=actuators,
            priority_to_actuators=priority_to_actuators,
            reference_detail_path=reference_detail_path,
        )
        state_detail_path = Path(reference_detail_path) if reference_detail_path else p
        df = _apply_gat_features(
            df, state_detail_path, gat_cache_dir, history_steps, require_gat_features,
            verify_source_fingerprint=verify_source_fingerprint,
        )
        return df, None
    except Exception as exc:
        return None, {"detail_file": str(p), "error": repr(exc), "chunk_index": chunk_index}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan.yaml")
    ap.add_argument("--horizon-steps", type=int, default=0)
    ap.add_argument("--history-steps", type=int, default=0)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max-detail-files", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=250)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--gat-feature-cache-dir", default="")
    ap.add_argument("--require-gat-features", action="store_true")
    ap.add_argument(
        "--trust-gat-feature-cache",
        action="store_true",
        help="Skip repeated detail-content hashing after the verified importer and GAT cache have completed.",
    )
    ap.add_argument("--skip-combine", action="store_true")
    ap.add_argument(
        "--source-scope",
        choices=["all", "generic_trajectories", "closed_loop"],
        default="",
        help="Restrict detail-file discovery. Use generic_trajectories for formal GAT-MPC training to avoid stale closed-loop results.",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    configured_gat_cache = (cfg.get("outputs", {}) or {}).get("gat_features", "outputs/gat_reconstructed_features")
    gat_cache_dir = Path(args.gat_feature_cache_dir) if args.gat_feature_cache_dir else root / configured_gat_cache
    gat_model_path = resolve_gat_model_path(cfg)
    hcfg = cfg.get("horizon_surrogate", {}) or {}
    horizon_steps = int(args.horizon_steps or hcfg.get("horizon_steps", 6))
    history_steps = int(args.history_steps or hcfg.get("history_steps", 3))
    max_files = int(args.max_detail_files or hcfg.get("max_detail_files", 0) or 0)
    priority_nodes = [
        x.strip()
        for x in (cfg_path(cfg, "outputs.design") / "priority_nodes.txt").read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    source_scope = str(args.source_scope or hcfg.get("source_scope", "all") or "all")
    rainfall_event_table = cfg_path(cfg, "outputs.rainfall") / "rainfall_event_table.csv"
    rain_events = pd.read_csv(rainfall_event_table)
    allowed_event_ids = set(rain_events["event_id"].astype(str)) if "event_id" in rain_events else set()
    unfiltered_files = _collect_detail_files(root, max_files=0, source_scope=source_scope, cfg=cfg)
    files = _collect_detail_files(
        root,
        max_files=max_files,
        source_scope=source_scope,
        allowed_event_ids=allowed_event_ids,
        cfg=cfg,
    )
    reference_by_event = {
        _detail_event_id(path): path
        for path in unfiltered_files
        if _detail_policy_id(path) == "no_control" and _detail_event_id(path) in allowed_event_ids
    }
    missing_reference_events = sorted({_detail_event_id(path) for path in files} - set(reference_by_event))
    if missing_reference_events:
        raise FileNotFoundError(
            "Every horizon sample requires a same-event No-control reference detail; "
            f"missing={missing_reference_events[:10]}"
        )
    schedule_path = cfg_path(cfg, "outputs.data_bank_train") / "trajectory_schedule.csv"
    if source_scope == "generic_trajectories" and schedule_path.exists():
        schedule = pd.read_csv(schedule_path)
        allowed_keys = set(zip(schedule["event_id"].astype(str), schedule["policy_id"].astype(str)))
        files = [
            p for p in files
            if (lambda parts: bool(parts[1]) and (parts[0], parts[2]) in allowed_keys)(
                p.stem.removesuffix("_detail").rpartition("__")
            )
        ]
    actuator_path = cfg_path(cfg, "outputs.audit") / "actuator_table.csv"
    actuators = pd.read_csv(actuator_path) if actuator_path.exists() else None
    if actuators is not None:
        actuator_scope = str((cfg.get("controller", {}) or {}).get("actuator_scope", "existing_rtc"))
        actuators = select_actuators_for_scope(actuators, actuator_scope)
    network_out = Path((cfg.get("outputs", {}) or {}).get("network", "outputs/network"))
    if not network_out.is_absolute():
        network_out = root / network_out
    priority_to_actuators_path = network_out / "priority_to_actuator_candidates.csv"
    priority_to_actuators = pd.read_csv(priority_to_actuators_path) if priority_to_actuators_path.exists() else None
    out_rel = hcfg.get("output_dataset", "data/surrogate/horizon_mpc_dataset.parquet")
    out_path = root / out_rel
    smoke_suffix = f"_smoke{max_files}" if max_files > 0 else ""
    if smoke_suffix:
        out_path = out_path.with_name(out_path.stem + smoke_suffix + out_path.suffix)
    chunk_size = max(1, int(args.chunk_size or 250))
    workers = max(1, int(args.workers or 1))
    chunk_dir = ensure_dir(out_path.parent / f"{out_path.stem}_chunks")
    manifest_path = chunk_dir / "chunk_manifest.csv"
    prior_manifest = pd.DataFrame()
    if manifest_path.exists() and manifest_path.stat().st_size > 0:
        try:
            prior_manifest = pd.read_csv(manifest_path)
        except Exception:
            prior_manifest = pd.DataFrame()
    prior_by_chunk = {
        int(row["chunk_index"]): row
        for _, row in prior_manifest.iterrows()
        if pd.notna(row.get("chunk_index"))
    }
    chunk_records = []
    failures = []
    warnings = []
    total_samples = 0
    total_files_processed = 0
    expected_chunks = (len(files) + chunk_size - 1) // chunk_size if files else 0
    for chunk_index, start in enumerate(range(0, len(files), chunk_size)):
        chunk_files = files[start : start + chunk_size]
        reference_files = [reference_by_event[_detail_event_id(path)] for path in chunk_files]
        fingerprint_files = list(dict.fromkeys([*chunk_files, *reference_files]))
        if args.require_gat_features:
            fingerprint_files.append(gat_model_path)
        source_fingerprint = source_file_fingerprint(fingerprint_files)
        default_chunk_path = _chunk_path(chunk_dir, chunk_index)
        existing = _preferred_existing_chunk_path(default_chunk_path)
        if args.resume and existing.exists():
            reuse_ok = False
            n_samples = 0
            read_error = ""
            try:
                reused = _read_frame(existing)
                n_samples = int(len(reused))
                missing_features = [c for c in ACTION_FEATURE_COLUMNS if c not in reused.columns]
                missing_targets = [c for c in TARGET_COLUMNS if c not in reused.columns]
                missing_effects = [f"effect_{c}" for c in TARGET_COLUMNS if f"effect_{c}" not in reused.columns]
                missing_references = [f"reference_{c}" for c in TARGET_COLUMNS if f"reference_{c}" not in reused.columns]
                event_filter_ok, event_filter_error = _chunk_event_filter_ok(reused, allowed_event_ids)
                prior = prior_by_chunk.get(chunk_index)
                source_filter_ok = prior is not None and str(prior.get("source_fingerprint", "")) == source_fingerprint
                gat_feature_ok = (not args.require_gat_features) or (
                    "state_feature_source" in reused
                    and reused["state_feature_source"].astype(str).eq("gat_reconstructed_sparse_sensors").all()
                )
                semantics_ok = (
                    "action_semantics" in reused
                    and reused["action_semantics"].astype(str).eq("absolute_from_no_control_reference").all()
                )
                reuse_ok = (
                    n_samples > 0 and not missing_features and not missing_targets
                    and not missing_effects and not missing_references
                    and event_filter_ok and source_filter_ok and gat_feature_ok and semantics_ok
                )
                if missing_features:
                    read_error = f"stale_schema_missing_action_features={missing_features[:5]}"
                if missing_targets:
                    read_error = f"stale_schema_missing_targets={missing_targets[:5]}"
                if missing_effects or missing_references:
                    read_error = "stale_schema_missing_paired_no_control_effect_labels"
                if not event_filter_ok:
                    read_error = event_filter_error
                if not source_filter_ok:
                    read_error = "source_fingerprint_mismatch"
                if not gat_feature_ok:
                    read_error = "missing_or_invalid_gat_reconstructed_features"
                if not semantics_ok:
                    read_error = "missing_or_invalid_action_semantics"
            except Exception as exc:
                read_error = repr(exc)
            if reuse_ok:
                chunk_records.append(
                    {
                        "chunk_index": chunk_index,
                        "chunk_file": str(existing),
                        "detail_files": len(chunk_files),
                        "samples": n_samples,
                        "status": "reused",
                        "source_fingerprint": source_fingerprint,
                    }
                )
                total_samples += n_samples
                total_files_processed += len(chunk_files)
                _write_chunk_manifest(chunk_records, manifest_path)
                print(f"[horizon_dataset] reused chunk {chunk_index + 1}/{expected_chunks} samples={n_samples}", flush=True)
                continue
            print(
                f"[horizon_dataset] rebuilding chunk {chunk_index + 1}/{expected_chunks}; "
                f"existing={existing} samples={n_samples} read_error={read_error}",
                flush=True,
            )
        rows = []
        chunk_failures = []
        jobs = [
            (
                p,
                priority_nodes,
                horizon_steps,
                history_steps,
                int(cfg["experiment"]["control_step_sec"]),
                int(args.stride),
                chunk_index,
                actuators,
                priority_to_actuators,
                gat_cache_dir,
                bool(args.require_gat_features),
                not bool(args.trust_gat_feature_cache),
                reference_by_event[_detail_event_id(p)],
            )
            for p in chunk_files
        ]
        if workers > 1 and len(jobs) > 1:
            with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
                futures = [pool.submit(_build_one_detail, job) for job in jobs]
                for fut in as_completed(futures):
                    df, failure = fut.result()
                    if failure:
                        chunk_failures.append(failure)
                    elif df is not None and not df.empty:
                        rows.append(df)
        else:
            for job in jobs:
                df, failure = _build_one_detail(job)
                if failure:
                    chunk_failures.append(failure)
                elif df is not None and not df.empty:
                    rows.append(df)
        chunk_df = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
        saved_chunk, chunk_fmt, write_error = _safe_write_frame(chunk_df, default_chunk_path)
        if write_error:
            warnings.append(_classify_write_issue(default_chunk_path, write_error, chunk_index=chunk_index))
        failures.extend(chunk_failures)
        n_samples = int(len(chunk_df))
        chunk_records.append(
            {
                "chunk_index": chunk_index,
                "chunk_file": str(saved_chunk),
                "detail_files": len(chunk_files),
                "samples": n_samples,
                "status": "written",
                "format": chunk_fmt,
                "failures": len(chunk_failures),
                "warnings": 1 if write_error else 0,
                "workers": workers,
                "source_fingerprint": source_fingerprint,
            }
        )
        total_samples += n_samples
        total_files_processed += len(chunk_files)
        _write_chunk_manifest(chunk_records, manifest_path)
        print(
            f"[horizon_dataset] chunk {chunk_index + 1}/{expected_chunks} "
            f"files={len(chunk_files)} samples={n_samples} total_samples={total_samples}",
            flush=True,
        )
    chunk_manifest = pd.DataFrame(chunk_records)
    chunk_manifest.to_csv(manifest_path, index=False)
    saved_path = out_path
    fmt = "parquet"
    combine_failures = []
    combine_warnings = []
    if not args.skip_combine:
        present_chunks = [Path(str(p)) for p in chunk_manifest.get("chunk_file", pd.Series(dtype=str)).tolist() if str(p)]
        if len(present_chunks) != expected_chunks:
            combine_failures.append(
                {
                    "detail_file": str(chunk_dir),
                    "error": f"incomplete_chunks: present={len(present_chunks)} expected={expected_chunks}",
                }
            )
        else:
            frames = []
            for p in present_chunks:
                try:
                    frames.append(_read_frame(p))
                except Exception as exc:
                    combine_failures.append({"detail_file": str(p), "error": f"combine_read_failed: {exc!r}"})
            dataset = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
            saved_path, fmt, write_error = _safe_write_frame(dataset, out_path)
            if write_error:
                combine_warnings.append(_classify_write_issue(out_path, write_error))
            total_samples = int(len(dataset))
    failures.extend(combine_failures)
    warnings.extend(combine_warnings)
    audit = {
        "detail_files_seen": len(files),
        "samples": int(total_samples),
        "horizon_steps": horizon_steps,
        "history_steps": history_steps,
        "stride": int(args.stride),
        "chunk_size": int(chunk_size),
        "workers": int(workers),
        "source_scope": source_scope,
        "trajectory_schedule": str(schedule_path) if schedule_path.exists() else "",
        "state_feature_source": "gat_reconstructed_sparse_sensors" if args.require_gat_features else "swmm_full_state",
        "gat_feature_cache_dir": str(gat_cache_dir),
        "gat_model": str(gat_model_path) if args.require_gat_features else "",
        "rainfall_event_table": str(rainfall_event_table),
        "rainfall_event_count": int(len(allowed_event_ids)),
        "detail_files_skipped_by_event_filter": int(len(unfiltered_files) - len(files)) if not max_files else int(
            len([p for p in unfiltered_files if _detail_event_id(p) not in allowed_event_ids])
        ),
        "chunk_dir": str(chunk_dir),
        "chunk_manifest": str(manifest_path),
        "expected_chunks": int(expected_chunks),
        "chunks_recorded": int(len(chunk_records)),
        "resume": bool(args.resume),
        "skip_combine": bool(args.skip_combine),
        "output_dataset": str(saved_path),
        "format": fmt,
        "action_feature_columns": ACTION_FEATURE_COLUMNS,
        "actuator_table": str(actuator_path) if actuator_path.exists() else "",
        "priority_to_actuators": str(priority_to_actuators_path) if priority_to_actuators_path.exists() else "",
        "failures": failures[:50],
        "failure_count": len(failures),
        "warnings": warnings[:50],
        "warning_count": len(warnings),
        "targets": TARGET_COLUMNS,
        "effect_targets": [f"effect_{c}" for c in TARGET_COLUMNS],
        "reference_policy": "no_control",
        "action_semantics": "absolute_from_no_control_reference",
    }
    surrogate_out = Path((cfg.get("outputs", {}) or {}).get("surrogate", "outputs/surrogate"))
    if not surrogate_out.is_absolute():
        surrogate_out = root / surrogate_out
    out_dir = ensure_dir(surrogate_out)
    audit_name = f"horizon_dataset_audit{smoke_suffix}.json" if smoke_suffix else "horizon_dataset_audit.json"
    (out_dir / audit_name).write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
