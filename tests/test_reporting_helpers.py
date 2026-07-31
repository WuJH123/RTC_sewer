from pathlib import Path
import importlib.util

import pandas as pd


def _load_compare_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "09_compare_baselines.py"
    spec = importlib.util.spec_from_file_location("compare_baselines", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_action_audit_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "18_audit_action_template_outcomes.py"
    spec = importlib.util.spec_from_file_location("audit_action_template", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_action_template_output_dir_is_run_tag_scoped(tmp_path):
    mod = _load_action_audit_module()
    diag = tmp_path / "diagnostics"

    scoped = mod._resolve_action_template_out_dir(diag, "formal", "run_a")
    legacy = mod._resolve_action_template_out_dir(diag, "debug", "")

    assert scoped == diag / "formal" / "run_a" / "action_template_outcomes"
    assert legacy == diag / "action_template_outcomes"


def test_selected_action_summary_parses_candidate_labels():
    mod = _load_compare_module()
    hist = pd.DataFrame(
        {
            "event_id": ["E1", "E1", "E1", "E2"],
            "phase": ["peak", "peak", "recession", "recession"],
            "fallback_to_nominal": [False, True, False, False],
            "selected_candidate_label": [
                "storage_inlet_restrict|scope=priority_corridor|d=-0.040|hold=2",
                "",
                "release_plus_pump_boost|scope=priority_upstream|d=0.080|hold=1",
                "release_plus_pump_boost|scope=priority_upstream|d=0.080|hold=1",
            ],
        }
    )

    summary = mod._selected_action_summary(hist)

    assert set(summary["template_name"]) == {"storage_inlet_restrict", "release_plus_pump_boost"}
    release = summary[summary["template_name"].eq("release_plus_pump_boost")].iloc[0]
    assert release["selected_count"] == 2
    assert release["events"] == 2
    assert release["phase_recession_count"] == 2


def test_failure_action_attribution_links_internal_failures_to_selected_actions():
    mod = _load_compare_module()
    fail = pd.DataFrame(
        {
            "baseline_policy": ["internal_rules", "auto_rbc"],
            "event_id": ["E1", "E1"],
            "failure_reason": ["PFV_worse;", "PFV_worse;"],
        }
    )
    hist = pd.DataFrame(
        {
            "event_id": ["E1", "E1", "E1"],
            "phase": ["peak", "recession", "recession"],
            "fallback_to_nominal": [False, False, True],
            "selected_candidate_label": [
                "pump_throttle|scope=priority_corridor|d=-0.040|hold=1",
                "release_plus_pump_boost|scope=priority_upstream|d=0.080|hold=1",
                "",
            ],
        }
    )

    attribution = mod._failure_action_attribution(fail, hist)

    internal = attribution[attribution["baseline_policy"].eq("internal_rules")]
    assert len(internal) == 2
    assert set(internal["template_name"]) == {"pump_throttle", "release_plus_pump_boost"}
    assert internal["selected_count"].sum() == 2
