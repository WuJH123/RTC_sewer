from sewerrtc.v4.evaluation import build_policy_lock


def test_policy_lock_contains_all_frozen_scientific_inputs() -> None:
    lock = build_policy_lock(
        model_sha="m",
        candidate_sha="c",
        threshold_sha="t",
        uncertainty_sha="u",
        fallback_sha="f",
        reference_sha="r",
        kpi_sha="k",
        event_split_sha="e",
        code_sha="g",
        config_sha="x",
    )

    assert lock["status"] == "locked"
    assert len(lock["components"]) == 10
