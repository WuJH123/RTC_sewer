"""Gate P3 feasibility map + plan audit: pure-function regression tests.

No SWMM run: every fixture is a small synthetic frame exercising the frozen
classification vocabulary, budgets, replay exemptions and recall math.
"""

import json

import pandas as pd

from sewerrtc.v4.pilot_feasibility_map import (
    FORBIDDEN_CLASS_TERMS,
    STATE_CLASSES,
    audit_feasibility_map,
    build_best_candidates,
    classify_feasibility_states,
    combine_state_samples,
    compute_generator_recall,
    evaluate_p3_gate,
    plan_feasibility_round_b_directives,
)
from sewerrtc.v4.pilot_feasibility_search import (
    FEASIBILITY_PHASE,
    P3_CONTRACT_VERSION,
    POSITIVE_CONTROL_ROLES,
    ROUND_A,
    ROUND_B,
    audit_feasibility_plan,
)
from sewerrtc.v4.pipeline_p3 import _round_b_directives_path
from sewerrtc.v4.pipeline import (
    ALL_STAGES,
    LONG_RUN_STAGES,
    PARTIAL_STAGE_RUN,
    PREFLIGHT_STAGE_RUN,
    PREREQUISITES,
    RUN_STAGE_GROUP_KEYS,
    RUN_STAGE_PLANS,
    STAGE_ARTIFACTS,
)

MARGIN = {"pfv_m3": 0.0, "tfv_m3": 0.0, "peak_m3s": 0.0}
BAND = {"pfv_m3": 25.0, "tfv_m3": 25.0, "peak_m3s": 0.01}


def _sample(
    sample_id: str,
    event: str,
    checkpoint: str,
    *,
    family: str = "famA",
    joint: bool = False,
    pfv_safe: bool = True,
    tfv_ok: bool = False,
    peak_ok: bool = True,
    delta_pfv: float = -5.0,
    delta_tfv: float = 30.0,
    delta_peak: float = 0.0,
    actual_sha: str = "",
    authentic: bool = True,
) -> dict:
    return {
        "sample_id": sample_id,
        "event_id": event,
        "checkpoint_id": checkpoint,
        "checkpoint_role": "responsive",
        "candidate_family": family,
        "pfv_safe": pfv_safe,
        "tfv_noninferior": tfv_ok,
        "peak_noninferior": peak_ok,
        "joint_noninferior": joint,
        "delta_pfv_h120_vs_no_control": delta_pfv,
        "delta_tfv_h120_vs_dynamic_internal": delta_tfv,
        "delta_peak_h120_vs_dynamic_internal": delta_peak,
        "actual_schedule_sha256": actual_sha or f"act_{sample_id}",
        "authentic_ok": authentic,
    }


def _catalog_row(
    event: str,
    checkpoint: str,
    *,
    positive: bool,
    split: str = "pilot_train",
) -> dict:
    return {
        "event_id": event,
        "checkpoint_id": checkpoint,
        "state_id": f"sha_{event}_{checkpoint}",
        "split": split,
        "checkpoint_min": 60.0,
        "positive_control_state": positive,
        "joint_missing_state": not positive,
        "oracle_revealed_flag_required": split != "pilot_train",
        "search_result_training_eligible": split == "pilot_train",
    }


def _six_state_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    catalog = pd.DataFrame(
        [
            _catalog_row("ev1", "ck1", positive=True),
            _catalog_row("ev2", "ck2", positive=False),
            _catalog_row("ev2", "ck3", positive=False),
            _catalog_row("ev3", "ck4", positive=False),
            _catalog_row("ev3", "ck5", positive=False),
            _catalog_row("ev4", "ck6", positive=False),
        ]
    )
    v2 = pd.DataFrame(
        [
            # S1: positive control with an online joint (famA).
            _sample(
                "a1", "ev1", "ck1", joint=True, tfv_ok=True, delta_tfv=-1.0
            ),
            # S2: tfv_always_degraded in v2.
            _sample("b1", "ev2", "ck2", delta_tfv=30.0),
            # S3: nothing safe, but all deltas inside margin + band.
            _sample(
                "c1",
                "ev2",
                "ck3",
                pfv_safe=False,
                delta_pfv=10.0,
                delta_tfv=5.0,
                delta_peak=0.005,
            ),
            # S4: never PFV-safe, far outside the band.
            _sample(
                "d1",
                "ev3",
                "ck4",
                pfv_safe=False,
                peak_ok=False,
                delta_pfv=100.0,
                delta_tfv=50.0,
                delta_peak=0.5,
            ),
            # S5: PFV-safe exists but TFV is hopeless (out of band).
            _sample("e1", "ev3", "ck5", delta_pfv=-2.0, delta_tfv=100.0),
            # S6: has evidence but one planned row stays unaccounted.
            _sample("g1", "ev4", "ck6", delta_tfv=40.0),
        ]
    )
    feas = pd.DataFrame(
        [
            # S1 gains a second joint family -> robust.
            _sample(
                "f1",
                "ev1",
                "ck1",
                family="famB",
                joint=True,
                tfv_ok=True,
                delta_tfv=-0.5,
            ),
            # S2: exact search finds the first joint (famC).
            _sample(
                "f2",
                "ev2",
                "ck2",
                family="famC",
                joint=True,
                tfv_ok=True,
                delta_tfv=-3.0,
            ),
            # S6: joint found, but the state stays unresolved fail-closed.
            _sample(
                "f3",
                "ev4",
                "ck6",
                family="famC",
                joint=True,
                tfv_ok=True,
                delta_tfv=-1.0,
            ),
        ]
    )
    return catalog, v2, feas


def _classified() -> pd.DataFrame:
    catalog, v2, feas = _six_state_fixture()
    samples = combine_state_samples(v2, feas)
    return classify_feasibility_states(
        catalog,
        samples,
        scientific_margin=MARGIN,
        boundary_band=BAND,
        unaccounted_by_state={("ev4", "ck6"): 1},
    )


def test_combine_keeps_only_responsive_v2_and_tags_source() -> None:
    _, v2, feas = _six_state_fixture()
    low = pd.DataFrame(
        [{**_sample("z1", "ev9", "ck9"), "checkpoint_role": "low_opportunity"}]
    )
    combined = combine_state_samples(
        pd.concat([v2, low], ignore_index=True), feas
    )
    assert "z1" not in set(combined["sample_id"])
    sources = combined.set_index("sample_id")["evidence_source"]
    assert sources["a1"] == "pilot_v2"
    assert sources["f2"] == FEASIBILITY_PHASE


def test_classification_priority_over_six_states() -> None:
    by_state = _classified().set_index(["event_id", "checkpoint_id"])
    assert (
        by_state.loc[("ev1", "ck1"), "state_feasibility_class"]
        == "joint_feasible_robust"
    )
    assert (
        by_state.loc[("ev2", "ck2"), "state_feasibility_class"]
        == "joint_feasible_found"
    )
    assert (
        by_state.loc[("ev2", "ck3"), "state_feasibility_class"]
        == "joint_boundary_found"
    )
    assert (
        by_state.loc[("ev3", "ck4"), "state_feasibility_class"]
        == "no_pfv_safe_found"
    )
    assert (
        by_state.loc[("ev3", "ck5"), "state_feasibility_class"]
        == "no_joint_found_under_budget"
    )
    # Unaccounted rows dominate even a found joint: fail-closed.
    assert (
        by_state.loc[("ev4", "ck6"), "state_feasibility_class"]
        == "execution_unresolved"
    )
    assert bool(by_state.loc[("ev2", "ck2"), "new_joint_from_search"])
    assert not bool(by_state.loc[("ev2", "ck2"), "online_joint_found"])
    assert set(by_state["state_feasibility_class"]).issubset(
        set(STATE_CLASSES)
    )
    for cls in STATE_CLASSES:
        for term in FORBIDDEN_CLASS_TERMS:
            assert term not in cls or cls == "no_joint_found_under_budget"
    # The frozen vocabulary itself must not smuggle forbidden physics claims.
    assert all(
        term not in cls
        for cls in STATE_CLASSES
        for term in FORBIDDEN_CLASS_TERMS
    )


def test_generator_recall_and_fallback_partition() -> None:
    report = compute_generator_recall(_classified())
    assert report["exact_joint_feasible_states"] == 2
    assert report["online_generator_joint_states"] == 1
    assert report["candidate_generator_state_recall"] == 0.5
    assert report["event_support"] == 2
    missed = {
        (row["event_id"], row["checkpoint_id"])
        for row in report["missed_feasible_states"]
    }
    assert missed == {("ev2", "ck2")}
    fallback = {
        (row["event_id"], row["checkpoint_id"])
        for row in report["fallback_only_states"]
    }
    assert fallback == {("ev3", "ck4"), ("ev3", "ck5")}


def test_best_candidates_prefer_joint_and_safe_minima() -> None:
    catalog, v2, feas = _six_state_fixture()
    best = build_best_candidates(combine_state_samples(v2, feas))
    by_state = best.set_index(["event_id", "checkpoint_id"])
    s1 = by_state.loc[("ev1", "ck1")]
    # famA has the lower joint delta TFV (-1.0 < -0.5).
    assert s1["best_joint_sample_id"] == "a1"
    assert s1["best_joint_value"] == -1.0
    s4 = by_state.loc[("ev3", "ck4")]
    assert s4["best_joint_sample_id"] == ""
    assert s4["best_pfv_safe_sample_id"] == ""
    assert s4["best_tfv_sample_id"] == "d1"


def test_round_b_directives_triggers_and_exclusions() -> None:
    catalog = pd.DataFrame(
        [
            _catalog_row("ev1", "ck1", positive=True),
            _catalog_row("ev2", "ck2", positive=False),  # near boundary
            _catalog_row("ev2", "ck3", positive=False),  # 2-family joint
            _catalog_row("ev3", "ck4", positive=False),  # budget exhausted
            _catalog_row("ev3", "ck5", positive=False),  # no trigger
        ]
    )
    samples = pd.DataFrame(
        [
            _sample("a1", "ev1", "ck1", joint=True, tfv_ok=True),
            _sample("b1", "ev2", "ck2", delta_tfv=10.0),
            _sample(
                "c1", "ev2", "ck3", family="famA", joint=True, tfv_ok=True
            ),
            _sample(
                "c2", "ev2", "ck3", family="famB", joint=True, tfv_ok=True
            ),
            _sample("d1", "ev3", "ck4", delta_tfv=10.0),
            _sample("e1", "ev3", "ck5", delta_pfv=-5.0, delta_tfv=100.0),
        ]
    )
    samples["evidence_source"] = "pilot_v2"
    plan = pd.DataFrame(
        [{"event_id": "ev2", "checkpoint_id": "ck2"}] * 16
        + [{"event_id": "ev3", "checkpoint_id": "ck4"}] * 32
        + [{"event_id": "ev3", "checkpoint_id": "ck5"}] * 16
    )
    directives = plan_feasibility_round_b_directives(
        catalog,
        samples,
        plan,
        scientific_margin=MARGIN,
        boundary_band=BAND,
    )
    assert len(directives) == 1
    row = directives.iloc[0]
    assert (row["event_id"], row["checkpoint_id"]) == ("ev2", "ck2")
    assert row["round_b_budget"] == 16
    assert "pfv_safe_tfv_near_boundary" in json.loads(row["triggers_json"])


def test_round_b_directives_frozen_once_round_b_planned(tmp_path) -> None:
    """Rebuilding after Round B must not overwrite the executed directives."""
    round_a_only = pd.DataFrame({"search_round": [ROUND_A, ROUND_A]})
    assert (
        _round_b_directives_path(tmp_path, round_a_only).name
        == "round_b_directives.csv"
    )
    with_round_b = pd.DataFrame({"search_round": [ROUND_A, ROUND_B]})
    assert (
        _round_b_directives_path(tmp_path, with_round_b).name
        == "round_b_directives_residual_diagnostic.csv"
    )
    no_column = pd.DataFrame({"sample_id": ["x"]})
    assert (
        _round_b_directives_path(tmp_path, no_column).name
        == "round_b_directives.csv"
    )


def test_p3_gate_recommendations() -> None:
    map_frame = _classified()
    report = compute_generator_recall(map_frame)
    gate = evaluate_p3_gate(report, map_frame)
    assert gate["recommendation"] == "fix_candidate_generator_before_train1600"
    assert not gate["gates"]["exact_joint_feasible_states_at_least_8"]
    assert not gate["gates"]["execution_unresolved_zero"]
    clean = map_frame[
        map_frame["state_feasibility_class"] != "execution_unresolved"
    ]
    perfect = {
        "exact_joint_feasible_states": 9,
        "online_generator_joint_states": 9,
        "candidate_generator_state_recall": 1.0,
        "event_support": 4,
    }
    assert (
        evaluate_p3_gate(perfect, clean)["recommendation"]
        == "generator_recall_perfect"
    )
    small = dict(perfect, exact_joint_feasible_states=5)
    small["online_generator_joint_states"] = 5
    assert (
        evaluate_p3_gate(small, clean)["recommendation"]
        == "exact_matches_online_original_30pct_gate_inappropriate"
    )


def _map_audit_fixture() -> dict:
    catalog = pd.DataFrame(
        [
            _catalog_row("ev1", "ck1", positive=True),
            _catalog_row("ev2", "ck2", positive=False),
        ]
    )
    v2 = pd.DataFrame(
        [
            _sample(
                "a1",
                "ev1",
                "ck1",
                joint=True,
                tfv_ok=True,
                actual_sha="SHA_REPLAY",
            ),
            _sample("b1", "ev2", "ck2", delta_tfv=30.0),
        ]
    )
    feas = pd.DataFrame(
        [
            _sample(
                "p_replay",
                "ev1",
                "ck1",
                joint=True,
                tfv_ok=True,
                actual_sha="SHA_REPLAY",
            ),
            _sample(
                "m1",
                "ev2",
                "ck2",
                family="famC",
                joint=True,
                tfv_ok=True,
                delta_tfv=-3.0,
            ),
        ]
    )
    plan = pd.DataFrame(
        [
            {
                "sample_id": "p_replay",
                "event_id": "ev1",
                "checkpoint_id": "ck1",
                "expected_replay_of": "a1",
                "expected_actual_sha": "SHA_REPLAY",
                "search_round": ROUND_A,
            },
            {
                "sample_id": "m1",
                "event_id": "ev2",
                "checkpoint_id": "ck2",
                "expected_replay_of": "",
                "expected_actual_sha": "",
                "search_round": ROUND_A,
            },
        ]
    )
    samples = combine_state_samples(v2, feas)
    map_frame = classify_feasibility_states(
        catalog, samples, scientific_margin=MARGIN, boundary_band=BAND
    )
    return {
        "catalog": catalog,
        "samples": samples,
        "map_frame": map_frame,
        "plan": plan,
    }


def test_map_audit_pass_and_replay_failure() -> None:
    fx = _map_audit_fixture()
    audit = audit_feasibility_map(
        fx["map_frame"],
        fx["samples"],
        {"accounting_closed": True, "missing": 0},
        catalog=fx["catalog"],
        candidate_plan=fx["plan"],
        hard_columns=("authentic_ok",),
        actual_duplicates=0,
    )
    assert audit["status"] == "pass"
    assert all(audit["checks"].values())
    assert audit["checks"]["replay_expected_nine"]
    assert audit["replay_success_rate"] == 1.0
    assert audit["p3_gate"]["recommendation"] == (
        "fix_candidate_generator_before_train1600"
    )
    # A replayed action that lands on a different actual SHA must block.
    broken = fx["samples"].copy()
    broken.loc[
        broken["sample_id"] == "p_replay", "actual_schedule_sha256"
    ] = "SHA_OTHER"
    bad = audit_feasibility_map(
        fx["map_frame"],
        broken,
        {"accounting_closed": True, "missing": 0},
        catalog=fx["catalog"],
        candidate_plan=fx["plan"],
        hard_columns=("authentic_ok",),
        actual_duplicates=0,
    )
    assert bad["status"] == "blocked"
    assert not bad["checks"]["replay_success_rate_100"]


def _plan_row(
    sample_id: str,
    event: str,
    checkpoint: str,
    *,
    family: str,
    role: str,
    round_tag: str,
    positive: bool,
    replay_of: str = "",
    expected_sha: str = "",
    req_sha: str = "",
) -> dict:
    return {
        "sample_id": sample_id,
        "event_id": event,
        "checkpoint_id": checkpoint,
        "family": family,
        "search_role": role,
        "search_round": round_tag,
        "positive_control_state": positive,
        "joint_missing_state": not positive,
        "search_result_training_eligible": True,
        "expected_replay_of": replay_of,
        "expected_actual_sha": expected_sha,
        "requested_schedule_sha256": req_sha or f"req_{sample_id}",
        "projected_schedule_sha256": f"proj_{sample_id}",
        "k_actual": 2,
        "binary_semantics_ok": True,
        "vsp_semantics_ok": True,
        "bounds_ok": True,
        "rate_limit_ok": True,
        "ramp_ok": True,
        "dwell_ok": True,
        "interlock_ok": True,
        "no_reversal_ok": True,
        "projection_valid": True,
        "source_contract_version": P3_CONTRACT_VERSION,
        "source_phase": FEASIBILITY_PHASE,
        "reference_reused": True,
        "network_sha256": "net",
        "contract_sha256": "con",
        "config_sha256": "cfg",
        "checkpoint_state_sha256": f"sha_{event}_{checkpoint}",
        "rainfall_sha256": f"rain_{event}",
    }


def _plan_audit_fixture() -> dict:
    catalog = pd.DataFrame(
        [
            _catalog_row("ev1", "ck1", positive=True),
            _catalog_row("ev2", "ck2", positive=False),
        ]
    )
    rows = [
        _plan_row(
            "p0",
            "ev1",
            "ck1",
            family=POSITIVE_CONTROL_ROLES[0],
            role=POSITIVE_CONTROL_ROLES[0],
            round_tag=ROUND_A,
            positive=True,
            replay_of="a1",
            expected_sha="SHA_REPLAY",
            # Replay is the only row allowed to collide with frozen SHAs.
            req_sha="FROZEN_V1_REQ",
        ),
    ]
    for index, family in enumerate(POSITIVE_CONTROL_ROLES[1:], start=1):
        rows.append(
            _plan_row(
                f"p{index}",
                "ev1",
                "ck1",
                family=family,
                role=family,
                round_tag=ROUND_A,
                positive=True,
            )
        )
    for index in range(2):
        rows.append(
            _plan_row(
                f"m{index}",
                "ev2",
                "ck2",
                family="toward_no_control",
                role="feasibility_candidate",
                round_tag=ROUND_A,
                positive=False,
            )
        )
    rows.append(
        _plan_row(
            "mb0",
            "ev2",
            "ck2",
            family="beam_sparse_k2_k4",
            role="feasibility_candidate",
            round_tag=ROUND_B,
            positive=False,
        )
    )
    plan = pd.DataFrame(rows)
    v1_plan = pd.DataFrame(
        [
            {
                "event_id": "ev1",
                "checkpoint_id": "ck1",
                "network_sha256": "net",
                "contract_sha256": "con",
                "config_sha256": "cfg",
                "checkpoint_state_sha256": "sha_ev1_ck1",
                "rainfall_sha256": "rain_ev1",
                "requested_schedule_sha256": "FROZEN_V1_REQ",
                "projected_schedule_sha256": "FROZEN_V1_PROJ",
            },
            {
                "event_id": "ev2",
                "checkpoint_id": "ck2",
                "network_sha256": "net",
                "contract_sha256": "con",
                "config_sha256": "cfg",
                "checkpoint_state_sha256": "sha_ev2_ck2",
                "rainfall_sha256": "rain_ev2",
                "requested_schedule_sha256": "FROZEN_V1_REQ2",
                "projected_schedule_sha256": "FROZEN_V1_PROJ2",
            },
        ]
    )
    v2_samples = pd.DataFrame(
        [
            {
                "event_id": "ev1",
                "checkpoint_id": "ck1",
                "actual_schedule_sha256": "SHA_REPLAY",
            }
        ]
    )
    branch_plan = pd.DataFrame({"branch": range(4 * len(plan))})
    return {
        "plan": plan,
        "branch_plan": branch_plan,
        "v1_plan": v1_plan,
        "v2_samples": v2_samples,
        "catalog": catalog,
    }


def test_plan_audit_pass_with_replay_exemption() -> None:
    fx = _plan_audit_fixture()
    audit = audit_feasibility_plan(
        fx["plan"],
        fx["branch_plan"],
        fx["v1_plan"],
        fx["v2_samples"],
        fx["catalog"],
    )
    assert audit["status"] == "pass", audit["checks"]
    assert audit["checks"]["non_replay_no_frozen_sha_overlap"]
    assert audit["checks"]["replay_present_all_positive_states"]
    assert audit["round_counts"] == {ROUND_A: 6, ROUND_B: 1}


def test_plan_audit_blocks_frozen_sha_reuse_outside_replay() -> None:
    fx = _plan_audit_fixture()
    plan = fx["plan"].copy()
    plan.loc[
        plan["sample_id"] == "m0", "requested_schedule_sha256"
    ] = "FROZEN_V1_REQ2"
    audit = audit_feasibility_plan(
        plan,
        fx["branch_plan"],
        fx["v1_plan"],
        fx["v2_samples"],
        fx["catalog"],
    )
    assert audit["status"] == "blocked"
    assert not audit["checks"]["non_replay_no_frozen_sha_overlap"]


def test_plan_audit_allows_cross_state_sha_collisions() -> None:
    # Actual-uniqueness is per state: the same schedule matrix applied at
    # two different checkpoints is two distinct actions and must pass.
    fx = _plan_audit_fixture()
    plan = fx["plan"].copy()
    plan.loc[plan["sample_id"] == "m0", "requested_schedule_sha256"] = (
        "req_p1"
    )
    plan.loc[plan["sample_id"] == "m0", "projected_schedule_sha256"] = (
        "proj_p1"
    )
    # Frozen SHA of a *different* state is equally legal for non-replay rows.
    plan.loc[plan["sample_id"] == "m1", "requested_schedule_sha256"] = (
        "FROZEN_V1_REQ"
    )
    audit = audit_feasibility_plan(
        plan,
        fx["branch_plan"],
        fx["v1_plan"],
        fx["v2_samples"],
        fx["catalog"],
    )
    assert audit["status"] == "pass", audit["checks"]


def test_plan_audit_blocks_round_b_on_positive_state() -> None:
    fx = _plan_audit_fixture()
    plan = fx["plan"].copy()
    plan.loc[plan["sample_id"] == "p1", "search_round"] = ROUND_B
    audit = audit_feasibility_plan(
        plan,
        fx["branch_plan"],
        fx["v1_plan"],
        fx["v2_samples"],
        fx["catalog"],
    )
    assert audit["status"] == "blocked"
    assert not audit["checks"]["round_b_only_missing_states"]
    assert not audit["checks"]["positive_round_a_within_budget"]


def test_feasibility_stage_wiring_constants() -> None:
    index = ALL_STAGES.index("PlanPilotFeasibilityMap")
    assert ALL_STAGES[index - 1] == "AuditLegacyOracleCompatibility"
    assert ALL_STAGES[index + 6] == "AuditPilotFeasibilityMap"
    # Dataset v3 chain sits between the feasibility map and Train1600.
    assert ALL_STAGES[index + 7] == "BuildPilotDatasetV3"
    assert ALL_STAGES[index + 11] == "FreezeP3Evidence"
    assert "RunPilotFeasibilityMap" in LONG_RUN_STAGES
    assert RUN_STAGE_PLANS["RunPilotFeasibilityMap"] == (
        "pilot_feasibility_p3/planning/feasibility_candidate_plan.csv"
    )
    assert RUN_STAGE_GROUP_KEYS["RunPilotFeasibilityMap"] == "sample_id"
    assert (
        PARTIAL_STAGE_RUN["BuildPilotFeasibilityPartial"]
        == "RunPilotFeasibilityMap"
    )
    assert (
        PARTIAL_STAGE_RUN["AuditPilotFeasibilityPartial"]
        == "RunPilotFeasibilityMap"
    )
    assert (
        PREFLIGHT_STAGE_RUN["AuditPilotFeasibilityPreflight"]
        == "RunPilotFeasibilityMap"
    )
    assert STAGE_ARTIFACTS["BuildPilotFeasibilityMap"] == (
        "pilot_feasibility_p3/map/pilot_state_feasibility_map.csv"
    )
    assert STAGE_ARTIFACTS["AuditPilotFeasibilityMap"] == (
        "pilot_feasibility_p3/map/pilot_feasibility_audit.json"
    )
    assert PREREQUISITES["PlanPilotFeasibilityMap"] == (
        "AuditLegacyOracleCompatibility",
    )
    assert PREREQUISITES["RunPilotFeasibilityMap"] == (
        "AuditPilotFeasibilityPreflight",
    )
    assert PREREQUISITES["BuildPilotFeasibilityMap"] == (
        "RunPilotFeasibilityMap",
    )
    assert PREREQUISITES["AuditPilotFeasibilityMap"] == (
        "BuildPilotFeasibilityMap",
    )
    # Train1600 entry stays gated on the (frozen-fail) pilot gate, untouched.
    assert PREREQUISITES["PlanTrain1600"] == ("EvaluatePilotGate",)
