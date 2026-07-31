from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd


def _detail_event_id(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_detail"):
        stem = stem[: -len("_detail")]
    event_id, sep, _policy = stem.rpartition("__")
    return event_id if sep else stem


def _normalise_policy_list(values: Iterable[str] | str | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        raw = values.split(",")
    else:
        raw = list(values)
    out: list[str] = []
    for value in raw:
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def build_transition_cache(
    trajectory_dir: str | Path,
    out_npz: str | Path,
    max_files: int = 0,
    time_stride: int = 1,
    horizon_steps: int = 6,
    priority_nodes: list[str] | None = None,
    dt_sec: int = 300,
    baseline_policy: str = "",
    allowed_event_ids: set[str] | Sequence[str] | None = None,
    allowed_event_policy_keys: set[tuple[str, str]] | None = None,
    reference_policies: Iterable[str] | str | None = None,
) -> dict:
    trajectory_dir = Path(trajectory_dir)
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    files_seen = sorted(trajectory_dir.glob("*_detail.csv"))
    allowed = {str(x) for x in allowed_event_ids} if allowed_event_ids is not None else None
    if allowed is not None:
        files = [p for p in files_seen if _detail_event_id(p) in allowed]
        stale_files = [p for p in files_seen if _detail_event_id(p) not in allowed]
    else:
        files = list(files_seen)
        stale_files = []
    if allowed_event_policy_keys is not None:
        allowed_keys = {(str(e), str(p)) for e, p in allowed_event_policy_keys}
        selected = []
        for path in files:
            stem = path.stem.removesuffix("_detail")
            event_id, sep, policy_id = stem.rpartition("__")
            if sep and (event_id, policy_id) in allowed_keys:
                selected.append(path)
        files = selected
    if max_files:
        files = files[:max_files]
    ref_policies = _normalise_policy_list(reference_policies)
    if not ref_policies and str(baseline_policy or "").strip():
        ref_policies = _normalise_policy_list(baseline_policy)
    baseline_label = (
        "multi_reference"
        if len(ref_policies) > 1
        else (ref_policies[0] if len(ref_policies) == 1 else "none")
    )
    priority_nodes = set(priority_nodes or [])
    X_state, X_action, X_action_seq, X_rain, X_rain_seq, Y_next, Y_seq, Y_risk, Y_delta, Y_current = [], [], [], [], [], [], [], [], [], []
    node_cols = None
    action_cols = None
    sources = []
    source_event_ids = []
    source_policy_ids = []
    trajectories: dict[tuple[str, str], dict] = {}
    skipped_unreadable = []

    def _event_policy(fp: Path, df: pd.DataFrame) -> tuple[str, str]:
        if "event_id" in df.columns and "policy_id" in df.columns and len(df):
            return str(df["event_id"].iloc[0]), str(df["policy_id"].iloc[0])
        stem = fp.stem
        if stem.endswith("_detail"):
            stem = stem[: -len("_detail")]
        if "__" in stem:
            e, p = stem.split("__", 1)
            return e, p
        return stem, "unknown"

    def _extract(fp: Path) -> tuple[tuple[str, str], dict] | None:
        df = pd.read_csv(fp)
        if len(df) < 2:
            return None
        h_cols = [c for c in df.columns if c.startswith("h:")]
        a_cols = [c for c in df.columns if c.startswith("a:")]
        f_cols = [c for c in df.columns if c.startswith("flood:")]
        if not h_cols or not a_cols:
            return None
        key = _event_policy(fp, df)
        return key, {"df": df, "h_cols": h_cols, "a_cols": a_cols, "f_cols": f_cols, "file": fp}

    for fp in files:
        try:
            item = _extract(fp)
        except Exception as exc:
            skipped_unreadable.append({"detail_file": str(fp), "error": repr(exc)})
            continue
        if item is None:
            continue
        key, data = item
        trajectories[key] = data
        if node_cols is None:
            node_cols = data["h_cols"]
            action_cols = data["a_cols"]

    events = sorted({e for e, _ in trajectories})
    H = int(max(1, horizon_steps))

    def _arrays(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        df = data["df"]
        h_cols = [c for c in (node_cols or data["h_cols"]) if c in df.columns]
        a_cols = [c for c in (action_cols or data["a_cols"]) if c in df.columns]
        h = df[h_cols].fillna(0).to_numpy(np.float32)
        a = df[a_cols].fillna(1).to_numpy(np.float32)
        if a.shape[1] < len(action_cols or []):
            full_a = np.ones((len(df), len(action_cols or [])), dtype=np.float32)
            col_index = {c: j for j, c in enumerate(action_cols or [])}
            for j, c in enumerate(a_cols):
                full_a[:, col_index[c]] = a[:, j]
            a = full_a
        rain = df[["rainfall_mm_h"]].fillna(0).to_numpy(np.float32)
        f_cols = data["f_cols"]
        flood = df[f_cols].fillna(0).to_numpy(np.float32) if f_cols else np.zeros((len(df), 1), np.float32)
        priority_f_cols = [c for c in f_cols if c.split(":", 1)[1] in priority_nodes]
        pflood = df[priority_f_cols].fillna(0).to_numpy(np.float32) if priority_f_cols else np.zeros((len(df), 1), np.float32)
        return h, a, rain, flood.sum(axis=1), pflood.sum(axis=1)

    sampled_events = set()
    missing_reference_events = []
    for event in events:
        reference_arrays = []
        for ref_policy in ref_policies:
            ref_key = (event, ref_policy)
            if ref_key in trajectories:
                rh, ra, rrain, rtotal, rpriority = _arrays(trajectories[ref_key])
                reference_arrays.append((ref_policy, rh, rrain, rtotal, rpriority))
        if ref_policies and not reference_arrays:
            missing_reference_events.append(event)
            continue
        event_items = [(p, d) for (e, p), d in trajectories.items() if e == event]
        for policy, data in event_items:
            h, a, rain, total_rate, priority_rate = _arrays(data)
            lengths = [len(h), len(a)]
            for _rp, rh, rrain, rtotal, rpriority in reference_arrays:
                lengths.extend([len(rh), len(rrain), len(rtotal), len(rpriority)])
            n = min(lengths)
            if n <= H:
                continue
            for i in range(0, n - H, max(1, time_stride)):
                cand_tfv_h = float(total_rate[i + 1 : i + H + 1].sum() * dt_sec)
                cand_pfv_h = float(priority_rate[i + 1 : i + H + 1].sum() * dt_sec)
                cand_peak_h = float(total_rate[i + 1 : i + H + 1].max())
                if reference_arrays:
                    ref_tfv = [
                        float(rtotal[i + 1 : i + H + 1].sum() * dt_sec)
                        for _rp, _rh, _rrain, rtotal, _rpriority in reference_arrays
                    ]
                    ref_pfv = [
                        float(rpriority[i + 1 : i + H + 1].sum() * dt_sec)
                        for _rp, _rh, _rrain, _rtotal, rpriority in reference_arrays
                    ]
                    ref_peak = [
                        float(rtotal[i + 1 : i + H + 1].max())
                        for _rp, _rh, _rrain, rtotal, _rpriority in reference_arrays
                    ]
                    base_pfv_h = float(min(ref_pfv))
                    base_tfv_h = float(min(ref_tfv))
                    base_peak_h = float(min(ref_peak))
                else:
                    base_pfv_h = cand_pfv_h
                    base_tfv_h = cand_tfv_h
                    base_peak_h = cand_peak_h
                X_state.append(h[i])
                X_action.append(a[i])
                X_action_seq.append(a[i : i + H])
                X_rain.append(rain[i])
                X_rain_seq.append(rain[i : i + H])
                Y_next.append(h[i + 1])
                Y_seq.append(h[i + 1 : i + H + 1])
                Y_risk.append([cand_pfv_h, cand_tfv_h, cand_peak_h])
                Y_delta.append([cand_pfv_h - base_pfv_h, cand_tfv_h - base_tfv_h, cand_peak_h - base_peak_h])
                Y_current.append([base_pfv_h, base_tfv_h, base_peak_h])
                sources.append(f"{data['file'].name}:{i}:refs={','.join(ref_policies) or 'absolute'}")
                source_event_ids.append(event)
                source_policy_ids.append(policy)
                sampled_events.add(event)
    if not X_state:
        raise RuntimeError(f"No transition samples found under {trajectory_dir}")
    np.savez_compressed(
        out_npz,
        state=np.asarray(X_state, dtype=np.float32),
        action=np.asarray(X_action, dtype=np.float32),
        action_seq=np.asarray(X_action_seq, dtype=np.float32),
        rain=np.asarray(X_rain, dtype=np.float32),
        rain_seq=np.asarray(X_rain_seq, dtype=np.float32),
        next_state=np.asarray(Y_next, dtype=np.float32),
        target_seq=np.asarray(Y_seq, dtype=np.float32),
        risk_horizon=np.asarray(Y_risk, dtype=np.float32),
        risk_delta=np.asarray(Y_delta, dtype=np.float32),
        current_risk_ref=np.asarray(Y_current, dtype=np.float32),
        node_cols=np.asarray(node_cols or [], dtype=object),
        action_cols=np.asarray(action_cols or [], dtype=object),
        sources=np.asarray(sources, dtype=object),
        event_ids=np.asarray(source_event_ids, dtype=object),
        policy_ids=np.asarray(source_policy_ids, dtype=object),
    )
    meta = {
        "trajectory_dir": str(trajectory_dir),
        "files": len(files),
        "files_seen": len(files_seen),
        "files_used": len(trajectories),
        "skipped_stale_detail_files": len(stale_files),
        "skipped_unreadable_detail_files": len(skipped_unreadable),
        "samples": len(X_state),
        "nodes": len(node_cols or []),
        "actions": len(action_cols or []),
        "horizon_steps": int(horizon_steps),
        "dt_sec": int(dt_sec),
        "baseline_policy": baseline_label,
        "reference_policies": ref_policies,
        "reference_aggregation": (
            "min_reference_each_metric"
            if len(ref_policies) > 1
            else ("single_reference" if len(ref_policies) == 1 else "absolute_no_reference")
        ),
        "events_seen": len(events),
        "paired_events": len(sampled_events),
        "missing_reference_events": len(missing_reference_events),
        "allowed_event_count": len(allowed) if allowed is not None else None,
        "stale_detail_examples": [str(p.name) for p in stale_files[:20]],
        "unreadable_detail_examples": skipped_unreadable[:20],
        "out_npz": str(out_npz),
    }
    out_npz.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_npz(path: str | Path, keys: Iterable[str] | None = None) -> dict:
    wanted = None if keys is None else {str(k) for k in keys}
    with np.load(path, allow_pickle=True) as arr:
        selected = arr.files if wanted is None else [k for k in arr.files if k in wanted]
        return {k: arr[k] for k in selected}
