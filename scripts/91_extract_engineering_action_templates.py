from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def _parse_actuators(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            raw = json.loads(text.replace("'", '"'))
            if isinstance(raw, list):
                return [str(item).strip() for item in raw if str(item).strip()]
        except json.JSONDecodeError:
            pass
        text = text.strip("[]")
    return [item.strip().strip("'\"") for item in text.split(",") if item.strip().strip("'\"")]


def _magnitude_from_label(text: str, fallback: float = 0.12) -> tuple[float, float]:
    m = re.search(r"(?:^|\|)d=(?P<delta>[+-]?\d+(?:\.\d+)?)", text)
    if not m:
        return abs(float(fallback)), 1.0
    delta = float(m.group("delta"))
    return abs(delta), -1.0 if delta < 0.0 else 1.0


def _template_from_label(label: str, target_actuators: object = None, *, default_tier: int = 1) -> dict[str, object] | None:
    text = str(label)
    m = re.match(r"tier(?P<tier>[12])_(?P<profile>ramp|early_hold|delayed_hold|ramp_restore|early_then_restore|pulse)_(?P<aid>.+)_(?P<mag>\d+\.\d+)_(?P<direction>[+-]\d)", text)
    if m:
        return {
            "label": f"{m.group('profile')}_{m.group('aid')}_{m.group('mag')}_{m.group('direction')}",
            "kind": "continuous_profile",
            "tier": int(m.group("tier")),
            "actuators": [m.group("aid")],
            "profile": m.group("profile"),
            "magnitude": float(m.group("mag")),
            "direction": -1.0 if m.group("direction").startswith("-") else 1.0,
            "rationale": "High-frequency closed-loop action template extracted from prior evidence.",
        }
    m = re.match(r"tier(?P<tier>[12])_binary_pump_(?P<aid>.+)_from_h(?P<start>\d+)", text)
    if m:
        return {
            "label": f"binary_pump_{m.group('aid')}_from_h{m.group('start')}",
            "kind": "binary_pump",
            "tier": int(m.group("tier")),
            "actuators": [m.group("aid")],
            "start_step": int(m.group("start")),
            "rationale": "Binary pump template extracted from prior closed-loop evidence.",
        }
    actuators = _parse_actuators(target_actuators)
    if "actuator=" in text and not actuators:
        m = re.search(r"(?:^|\|)actuator=(?P<aid>[^|]+)", text)
        if m:
            actuators = [m.group("aid").strip()]
    if not actuators:
        return None
    magnitude, direction = _magnitude_from_label(text)
    priority_match = re.search(r"(?:^|\|)priority=(?P<priority>[^|]+)", text)
    priority = priority_match.group("priority") if priority_match else "unknown"
    if text.startswith("priority_group_regulator_restrict_then_restore"):
        profile = "early_then_restore"
    elif text.startswith("priority_group_regulator_pulse_release_if_safe"):
        profile = "pulse"
    elif text.startswith("priority_group_restrict"):
        profile = "early_then_restore"
    elif text.startswith("priority_group_release"):
        profile = "early_then_restore"
    elif text.startswith("regulator_restrict_then_restore"):
        profile = "early_then_restore"
    elif text.startswith("regulator_release_if_safe"):
        profile = "early_then_restore"
    elif text.startswith("regulator_pulse_release_if_safe"):
        profile = "pulse"
    else:
        return None
    label_slug = re.sub(r"[^A-Za-z0-9_.+-]+", "_", f"{profile}_{priority}_{len(actuators)}_{magnitude:.3f}_{direction:+.0f}").strip("_")
    return {
        "label": label_slug,
        "kind": "continuous_profile",
        "tier": int(default_tier),
        "actuators": actuators,
        "profile": profile,
        "magnitude": float(magnitude),
        "direction": float(direction),
        "priority_node": priority,
        "rationale": "Frozen Project6 v8 closed-loop group action template; used as Tier 1 engineering evidence.",
    }
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract reusable engineering action templates from controller history.")
    parser.add_argument("--history-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--max-templates", type=int, default=12)
    parser.add_argument("--allowed-events", default="")
    parser.add_argument("--template-tier", type=int, default=1)
    args = parser.parse_args()

    history_dir = Path(args.history_dir)
    allowed_events = {x.strip() for x in args.allowed_events.split(",") if x.strip()}
    rows: list[pd.DataFrame] = []
    for path in sorted(history_dir.glob("*__controller_history.csv")):
        frame = pd.read_csv(path)
        if allowed_events:
            frame = frame[frame["event_id"].astype(str).isin(allowed_events)].copy()
        if not frame.empty:
            rows.append(frame)
    if not rows:
        raise FileNotFoundError(f"no controller history rows found under {history_dir}")
    history = pd.concat(rows, ignore_index=True)
    label_column = "selected_label" if "selected_label" in history.columns else "selected_sequence_label"
    if label_column not in history.columns:
        raise KeyError("controller history must contain selected_label or selected_sequence_label")
    gate_series = (
        history["selected_gate_pass"]
        if "selected_gate_pass" in history.columns
        else pd.Series([True] * len(history), index=history.index)
    )
    active = history[
        gate_series.fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])
        & ~history[label_column].astype(str).isin(["deployment_reliability_no_control", "hold_native", "reference_no_control"])
    ].copy()
    grouped = (
        active.groupby([label_column, "target_actuators"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    counts = active[label_column].astype(str).value_counts()
    templates = []
    seen = set()
    for _, row in grouped.iterrows():
        label = str(row[label_column])
        count = int(row["count"])
        if int(count) < int(args.min_count):
            continue
        template = _template_from_label(label, row.get("target_actuators", ""), default_tier=int(args.template_tier))
        if template is None:
            continue
        key = json.dumps({
            "kind": template.get("kind"),
            "tier": template.get("tier"),
            "actuators": template.get("actuators"),
            "profile": template.get("profile"),
            "magnitude": template.get("magnitude"),
            "direction": template.get("direction"),
            "start_step": template.get("start_step"),
        }, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        template["source_label"] = label
        template["source_count"] = int(count)
        templates.append(template)
        if len(templates) >= int(args.max_templates):
            break
    out = {
        "history_dir": str(history_dir),
        "active_rows": int(len(active)),
        "label_column": label_column,
        "min_count": int(args.min_count),
        "templates": templates,
        "label_counts": counts.head(30).to_dict(),
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
