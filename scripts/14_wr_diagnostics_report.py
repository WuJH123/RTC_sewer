from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config


def _read_csv(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _read_json(path: Path) -> dict:
    if path.exists() and path.stat().st_size > 0:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _last_or_best(df: pd.DataFrame, metric: str) -> dict:
    if df.empty:
        return {}
    if metric in df:
        row = df.sort_values(metric, ascending=False).iloc[0]
    else:
        row = df.iloc[-1]
    return {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in row.to_dict().items()}


def _table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    """Render a table without requiring pandas' optional tabulate dependency."""
    if df.empty:
        return "_No rows._"
    show = df.head(max_rows).copy() if max_rows else df.copy()
    try:
        return show.to_markdown(index=False)
    except Exception:
        return "```text\n" + show.to_string(index=False) + "\n```"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan.yaml")
    ap.add_argument("--mode", choices=["debug", "formal"], default="debug")
    ap.add_argument("--run-tag", default="", help="Optional run-specific diagnostics subdirectory.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    diag = cfg_path(cfg, "outputs.diagnostics")
    out = diag / args.mode
    if args.run_tag:
        out = out / args.run_tag
    out = ensure_dir(out)

    gate = _read_json(out / "gate_summary.json")
    policy = _read_csv(out / "strategy_policy_summary.csv")
    runtime = _read_csv(out / "runtime_and_stability_diagnostics.csv")
    gat = _read_csv(diag / "gat_reconstruction_report.csv")
    surrogate = _read_csv(diag / "surrogate_training_history.csv")
    surrogate_dist = _read_csv(diag / "surrogate_delta_split_distribution.csv")
    residual_train = _read_csv(diag / "residual_action_value_training_report.csv")
    residual_group = _read_csv(diag / "residual_action_value_group_report.csv")
    residual_balance = _read_json(diag / "residual_action_value_balance.json")
    residual_overall = _read_json(diag / "residual_counterfactuals" / "residual_counterfactual_overall.json")
    residual_template = _read_csv(diag / "residual_counterfactuals" / "residual_counterfactual_by_template.csv")
    residual_tier = _read_csv(diag / "residual_counterfactuals" / "residual_counterfactual_by_delta_tier.csv")
    preflight = _read_json(out / "wr_preflight_gate.json")
    selected_actions = _read_csv(out / "selected_action_diagnostics.csv")
    failure_attribution = _read_csv(out / "failure_action_attribution.csv")
    action_guard = _read_csv(out / "action_template_outcomes" / "action_template_empirical_guard_table.csv")

    report = {
        "mode": args.mode,
        "run_tag": args.run_tag,
        "gate": gate,
        "preflight": preflight,
        "gat": gat.to_dict(orient="records") if not gat.empty else [],
        "surrogate_best_by_pfv_direction": _last_or_best(surrogate, "PFV_nz_dir"),
        "surrogate_delta_distribution": surrogate_dist.to_dict(orient="records") if not surrogate_dist.empty else [],
        "residual_action_value_best": _last_or_best(residual_train, "PFV_direction_accuracy"),
        "residual_action_value_group_rows": int(len(residual_group)),
        "residual_action_value_balance": residual_balance,
        "residual_counterfactual_overall": residual_overall,
        "residual_counterfactual_by_delta_tier": residual_tier.to_dict(orient="records") if not residual_tier.empty else [],
        "policy_summary": policy.to_dict(orient="records") if not policy.empty else [],
        "runtime_rows": int(len(runtime)),
        "selected_action_rows": int(len(selected_actions)),
        "failure_action_attribution_rows": int(len(failure_attribution)),
        "action_guard_rows": int(len(action_guard)),
    }
    (out / "water_research_readiness_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Water Research Readiness Report\n")
    lines.append(f"- Mode: `{args.mode}`")
    lines.append(f"- Gate passed: `{gate.get('passed', 'unknown')}`")
    if preflight:
        lines.append(f"- Preflight passed: `{preflight.get('passed', 'unknown')}`")
        if preflight.get("reasons"):
            lines.append(f"- Preflight blocking reasons: `{'; '.join(map(str, preflight.get('reasons', [])))}`")
    if gate:
        lines.append(
            "- Main gate: "
            f"PFV median reduction `{gate.get('PFV_median_reduction_pct', 'NA')}`, "
            f"PFV worse frac `{gate.get('PFV_worse_frac', 'NA')}`, "
            f"TFV mean reduction `{gate.get('TFV_mean_reduction_pct', 'NA')}`, "
            f"peak mean reduction `{gate.get('peak_TFV_rate_mean_reduction_pct', 'NA')}`."
        )
    if residual_overall:
        lines.append(
            "- Residual counterfactual data: "
            f"{residual_overall.get('rows', 0)} rows, "
            f"{residual_overall.get('events', 0)} events, "
            f"PFV improve frac `{residual_overall.get('pfv_improve_frac', 'NA')}`, "
            f"PFV improve + safe frac `{residual_overall.get('pfv_improve_safe_frac', 'NA')}`."
        )
    if not residual_tier.empty:
        lines.append("\n## Residual Delta-Tier Summary\n")
        lines.append(_table(residual_tier))
    if not policy.empty:
        lines.append("\n## Strategy Summary\n")
        lines.append(_table(policy))
    if not residual_template.empty:
        lines.append("\n## Residual Template Summary\n")
        lines.append(_table(residual_template, max_rows=12))
    if not residual_group.empty:
        lines.append("\n## Action-Value Group Diagnostics\n")
        lines.append(_table(residual_group, max_rows=16))
    if not selected_actions.empty:
        lines.append("\n## Selected Action Diagnostics\n")
        lines.append(_table(selected_actions, max_rows=16))
    if not failure_attribution.empty:
        lines.append("\n## Failure Action Attribution\n")
        lines.append(
            "This table links each failed closed-loop event to the residual action templates "
            "actually selected by the controller, enabling mechanism-level diagnosis rather "
            "than parameter-only tuning."
        )
        lines.append(_table(failure_attribution, max_rows=20))
    if not action_guard.empty:
        lines.append("\n## Empirical Action Guard\n")
        lines.append(_table(action_guard.sort_values("empirical_allow", ascending=False), max_rows=16))
    if not surrogate.empty:
        lines.append("\n## Surrogate Training Snapshot\n")
        lines.append(_table(surrogate.tail(5)))
    lines.append("\n## Recommended Evidence Package\n")
    lines.append(
        "1. Effectiveness: paired PFV/TFV/peak comparison against `internal_rules`, `auto_rbc`, "
        "`efd_static`, and `efd_storage_priority`."
    )
    lines.append(
        "2. Speed: report proposed wall-time per control step and compare with physical-model MPC or "
        "baseline simulation cost."
    )
    lines.append(
        "3. Accuracy: report GAT reconstruction and graph surrogate horizon-level direction accuracy."
    )
    lines.append(
        "4. Module contribution: ablate GAT, residual action-value shield, PFV-first objective, and "
        "native shield."
    )
    lines.append(
        "5. Robustness: test sparse sensor ratios, rainfall uncertainty, actuator cooldown, and safe fallback."
    )
    (out / "water_research_readiness_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"saved": str(out / "water_research_readiness_report.md"), "passed": gate.get("passed")}, indent=2))


if __name__ == "__main__":
    main()
