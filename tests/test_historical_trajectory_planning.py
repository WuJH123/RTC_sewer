from __future__ import annotations

import pandas as pd


def _write_detail(path, event_id: str, policy_id: str, nodes: list[str], actions: list[str], rows: int = 4) -> None:
    data = {
        "event_id": [event_id] * rows,
        "policy_id": [policy_id] * rows,
        "elapsed_min": [5 * i for i in range(rows)],
        "rainfall_mm_h": [0.0, 10.0, 5.0, 0.0],
    }
    for i, node in enumerate(nodes):
        data[f"h:{node}"] = [float(i + j) for j in range(rows)]
        data[f"flood:{node}"] = [0.0, float(i), float(i + 1), 0.0]
    for i, aid in enumerate(actions):
        data[f"a:{aid}"] = [1.0, 1.0 - 0.1 * (i % 2), 0.9, 1.0]
    pd.DataFrame(data).to_csv(path, index=False)


def test_history_inventory_keeps_network_signatures_separate(tmp_path):
    from sewerrtc.data.historical_trajectory_planning import scan_trajectory_roots

    wuhan_nodes = ["J1", "J2", "J3"]
    beta_nodes = ["B1", "B2"]
    actions = [f"A{i:02d}" for i in range(36)]
    root = tmp_path / "Project6" / "outputs" / "bank" / "trajectories"
    beta = tmp_path / "Project5" / "outputs" / "pystorms_beta" / "trajectories"
    root.mkdir(parents=True)
    beta.mkdir(parents=True)
    _write_detail(root / "T20_D75_chicago_center__no_control_detail.csv", "T20_D75_chicago_center", "no_control", wuhan_nodes, actions)
    _write_detail(root / "T20_D75_chicago_center__proposed_detail.csv", "T20_D75_chicago_center", "proposed", wuhan_nodes, actions)
    _write_detail(beta / "beta_event__no_control_detail.csv", "beta_event", "no_control", beta_nodes, actions[:2])

    inventory = scan_trajectory_roots([root, beta], canonical_action_ids=actions)

    assert len(inventory) == 3
    assert inventory["node_signature"].nunique() == 2
    assert inventory.loc[inventory["event_id"].eq("beta_event"), "canonical_action_coverage"].iloc[0] < 1.0
    assert inventory.loc[inventory["event_id"].eq("T20_D75_chicago_center"), "can_train_gat"].all()


def test_gat_plan_allows_only_matching_networks_and_all_policies(tmp_path):
    from sewerrtc.data.historical_trajectory_planning import build_gat_mixing_plan, scan_trajectory_roots

    actions = [f"A{i:02d}" for i in range(36)]
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    _write_detail(root_a / "E1__no_control_detail.csv", "E1", "no_control", ["N1", "N2"], actions)
    _write_detail(root_a / "E1__auto_rbc_detail.csv", "E1", "auto_rbc", ["N1", "N2"], actions)
    _write_detail(root_b / "E2__no_control_detail.csv", "E2", "no_control", ["X1"], actions)

    inventory = scan_trajectory_roots([root_a, root_b], canonical_action_ids=actions)
    plan = build_gat_mixing_plan(inventory, base_node_signature=inventory.iloc[0]["node_signature"])

    used = plan[plan["gat_use"]]
    assert set(used["event_id"]) == {"E1"}
    assert set(used["policy_id"]) == {"no_control", "auto_rbc"}
    assert plan.loc[plan["event_id"].eq("E2"), "gat_exclusion_reason"].iloc[0] == "node_signature_mismatch"


def test_action_learning_plan_preserves_raw_temporal_action_contract(tmp_path):
    from sewerrtc.data.historical_trajectory_planning import build_action_learning_plan, scan_trajectory_roots

    actions = [f"A{i:02d}" for i in range(36)]
    root = tmp_path / "trajectories"
    root.mkdir()
    _write_detail(root / "E1__no_control_detail.csv", "E1", "no_control", ["N1", "N2"], actions)
    _write_detail(root / "E1__legacy_group_detail.csv", "E1", "legacy_group", ["N1", "N2"], actions)
    _write_detail(root / "E1__short_action_detail.csv", "E1", "short_action", ["N1", "N2"], actions[:10])

    inventory = scan_trajectory_roots([root], canonical_action_ids=actions)
    plan = build_action_learning_plan(inventory, canonical_action_ids=actions, horizon_steps=6)

    usable = plan[plan["action_learning_use"]]
    assert set(usable["policy_id"]) == {"no_control", "legacy_group"}
    assert usable["action_tensor_shape"].eq("[H,36]").all()
    assert plan.loc[plan["policy_id"].eq("legacy_group"), "effect_label_role"].iloc[0] == "observational_dynamics_pretraining"
    assert not plan.loc[plan["policy_id"].eq("short_action"), "action_learning_use"].iloc[0]
