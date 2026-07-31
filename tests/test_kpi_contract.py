import numpy as np

from sewerrtc.evaluation.kpi_contract import compute_kpis, integrate_volume_from_rates, peak_total_rate


def test_integrate_volume_from_rates_left_endpoint():
    rates = np.array([[1.0, 2.0], [0.0, 3.0]])
    assert integrate_volume_from_rates(rates, dt_sec=10.0) == 60.0


def test_peak_total_rate_sums_nodes_then_maxes_time():
    rates = np.array([[1.0, 2.0], [4.0, 1.0]])
    assert peak_total_rate(rates) == 5.0


def test_compute_kpis_uses_priority_and_all_scopes():
    all_rates = np.array([[1.0, 2.0], [0.0, 3.0]])
    priority_rates = np.array([[2.0], [3.0]])
    kpis = compute_kpis(all_flooding_rates=all_rates, priority_flooding_rates=priority_rates, dt_sec=10.0)
    assert kpis["TFV"] == 60.0
    assert kpis["PFV"] == 50.0
    assert kpis["peak_TFV_rate"] == 3.0
