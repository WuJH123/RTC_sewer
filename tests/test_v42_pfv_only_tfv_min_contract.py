from __future__ import annotations

import numpy as np

from sewerrtc.control.pfvfirst_mpc_v42 import (
    EngineeringStatus,
    FrozenFallback,
    MPCandidate,
    SafetyMargins,
    decide_pfvfirst_mpc,
)


def _fallback() -> FrozenFallback:
    return FrozenFallback("hold", np.zeros((3, 2)), "fallback")


def _candidate(
    name: str,
    *,
    pfv: float = 0.0,
    pfv_budget_metric: float | None = None,
    tfv: float = 0.0,
    peak: float = 0.0,
    priority: float = 0.0,
    uncertainty: bool = True,
    ood: bool = True,
    changed: int = 1,
    engineering: EngineeringStatus | None = None,
    executable: bool = True,
) -> MPCandidate:
    return MPCandidate(
        candidate_id=name,
        action_sequence=np.ones((3, 2)),
        pfv_delta_ucb_m3=pfv,
        peak_delta_ucb_m3s=peak,
        tfv_delta_di_m3=tfv,
        action_cost=999.0,
        terminal_cost=999.0,
        uncertainty_cost=999.0,
        changed_facilities=changed,
        engineering=engineering or EngineeringStatus(True, True, True, True, True),
        uncertainty_pass=uncertainty,
        ood_pass=ood,
        executable=executable,
        pfv_no_control_m3=1000.0,
        priority_depth_ucb_m=(priority,),
        priority_depth_limit_m=(1.0,),
        pfv_budget_metric_ucb_m3=pfv_budget_metric,
    )


def test_peak_cannot_beat_lower_tfv_candidate() -> None:
    d = decide_pfvfirst_mpc(
        candidates=[_candidate("a", tfv=-100.0), _candidate("b", tfv=-50.0, peak=-100.0)],
        fallback=_fallback(),
    )
    assert d.selected_id == "a"


def test_pfv_violation_cannot_be_compensated_by_tfv() -> None:
    d = decide_pfvfirst_mpc(
        candidates=[_candidate("bad", pfv=151.0, tfv=-10000.0), _candidate("safe", tfv=-1.0)],
        fallback=_fallback(),
    )
    assert d.selected_id == "safe"
    assert "PFV_budget_exceeded_vs_no_control" in d.audits[0].rejection_reasons


def test_complete_pfv_budget_metric_is_authoritative_when_present() -> None:
    # Legacy delta-UCB arithmetic would admit this because the allowance is
    # 100 + 5% * 1000 = 150 m3. The complete calibrated statistic says the
    # budget margin UCB is 100.01 m3, so Formal V2 must reject it.
    d = decide_pfvfirst_mpc(
        candidates=[
            _candidate(
                "unsafe_complete_budget",
                pfv=0.0,
                pfv_budget_metric=100.01,
                tfv=-10000.0,
            )
        ],
        fallback=_fallback(),
    )
    assert d.used_fallback
    assert d.audits[0].pfv_budget_metric_ucb_m3 == 100.01
    assert "PFV_budget_exceeded_vs_no_control" in d.audits[0].rejection_reasons


def test_complete_pfv_budget_metric_at_100_is_admitted() -> None:
    d = decide_pfvfirst_mpc(
        candidates=[_candidate("edge", pfv=999.0, pfv_budget_metric=100.0, tfv=-2.0)],
        fallback=_fallback(),
    )
    assert not d.used_fallback
    assert d.selected_id == "edge"


def test_priority_depth_is_not_an_admission_gate() -> None:
    d = decide_pfvfirst_mpc(
        candidates=[_candidate("deep", priority=99.0, tfv=-2.0)], fallback=_fallback()
    )
    assert d.selected_id == "deep"


def test_ood_diagnostic_does_not_reject_finite_pfv_tfv() -> None:
    d = decide_pfvfirst_mpc(
        candidates=[_candidate("ood", ood=False, tfv=-2.0)], fallback=_fallback()
    )
    assert d.selected_id == "ood"


def test_uncertainty_flag_does_not_duplicate_pfv_ucb_gate() -> None:
    d = decide_pfvfirst_mpc(
        candidates=[_candidate("uncertain", uncertainty=False, tfv=-2.0)], fallback=_fallback()
    )
    assert d.selected_id == "uncertain"


def test_engineering_dwell_violation_is_rejected() -> None:
    d = decide_pfvfirst_mpc(
        candidates=[
            _candidate(
                "dwell",
                engineering=EngineeringStatus(True, True, True, False, True),
                tfv=-100.0,
            ),
            _candidate("hold", tfv=0.0),
        ],
        fallback=_fallback(),
    )
    assert d.selected_id == "hold"
    assert "engineering_dwell_violation" in d.audits[0].rejection_reasons


def test_k_nine_is_rejected() -> None:
    d = decide_pfvfirst_mpc(
        candidates=[_candidate("k9", changed=9, tfv=-100.0), _candidate("ok")],
        fallback=_fallback(),
    )
    assert d.selected_id == "ok"
    assert "K_exceeded" in d.audits[0].rejection_reasons


def test_empty_pfv_safe_set_falls_back() -> None:
    d = decide_pfvfirst_mpc(
        candidates=[_candidate("bad", pfv=151.0)], fallback=_fallback()
    )
    assert d.used_fallback
    assert d.selected_id == "hold"
    assert d.metadata["fallback_is_engineering_fail_safe_not_pfv_certificate"] is True


def test_tfv_tie_uses_candidate_id() -> None:
    d = decide_pfvfirst_mpc(
        candidates=[_candidate("z", tfv=-1.0), _candidate("a", tfv=-1.0)],
        fallback=_fallback(),
    )
    assert d.selected_id == "a"
