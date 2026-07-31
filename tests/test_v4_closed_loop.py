from sewerrtc.v4.closed_loop import (
    CumulativeSafetyBudget,
    SURROGATE_ABLATIONS,
    audit_closed_loop_order,
    select_pfv_first_candidate,
    timing_budget_ok,
)


def test_surrogate_ablation_matrix_has_four_required_experiments() -> None:
    assert SURROGATE_ABLATIONS == {
        "A": ("true_state", "exact_evaluation"),
        "B": ("true_state", "v4_surrogate"),
        "C": ("gat_state", "exact_evaluation"),
        "D": ("gat_state", "v4_surrogate"),
    }


def test_exact_must_pass_before_surrogate_and_budget_is_under_10_minutes() -> None:
    assert audit_closed_loop_order(exact_exit_code=0)["status"] == "pass"
    assert audit_closed_loop_order(exact_exit_code=5)["status"] == "blocked"
    assert timing_budget_ok(599.0)
    assert not timing_budget_ok(600.0)


def test_pfv_first_selection_filters_unsafe_peak_and_terminal_candidates() -> None:
    import pandas as pd

    candidates = pd.DataFrame(
        {
            "candidate_id": ["pfv_bad", "peak_bad", "terminal_bad", "safe"],
            "delta_pfv": [1.0, 0.0, 0.0, 0.0],
            "delta_peak": [0.0, 1.0, 0.0, 0.0],
            "terminal_risk": [0.0, 0.0, 1.0, 0.0],
            "delta_tfv": [-100.0, -100.0, -100.0, -10.0],
            "action_cost": [0.0, 0.0, 0.0, 1.0],
            "switching_cost": [0.0, 0.0, 0.0, 0.0],
        }
    )
    selected = select_pfv_first_candidate(candidates)
    assert selected["candidate_id"] == "safe"

    budget = CumulativeSafetyBudget(initial_margin_m3=10.0)
    assert budget.consume(2.0)
    assert not budget.consume(9.0)


def test_selection_uses_upper_confidence_bounds_for_all_noninferiority_gates() -> None:
    import pandas as pd

    candidates = pd.DataFrame(
        {
            "candidate_id": ["mean_safe_but_ucb_unsafe", "tfv_bad", "safe"],
            "delta_pfv": [-1.0, -1.0, -1.0],
            "delta_peak": [-1.0, -1.0, -1.0],
            "delta_tfv": [-1.0, 1.0, -1.0],
            "delta_pfv_ucb": [0.1, -1.0, -1.0],
            "delta_peak_ucb": [-1.0, -1.0, -1.0],
            "delta_tfv_ucb": [-1.0, 1.0, -1.0],
            "terminal_risk": [0.0, 0.0, 0.0],
            "action_cost": [0.0, 0.0, 0.0],
            "switching_cost": [0.0, 0.0, 0.0],
        }
    )
    assert select_pfv_first_candidate(candidates)["candidate_id"] == "safe"
