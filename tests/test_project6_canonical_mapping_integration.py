from __future__ import annotations

from pathlib import Path
import numpy as np

from sewerrtc.control.actuator_scope import select_actuators_for_scope
from sewerrtc.control.canonical_action_order import CanonicalActionOrder
from sewerrtc.io.project_paths import load_config, cfg_path


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_project6_mapping_matches_training_and_online_order():
    cfg = load_config(ROOT / "configs/wuhan_project6_v8_storage_36.yaml")
    global_ids = np.load(ROOT / "outputs/cache_all109/transition_cache.npz", allow_pickle=True)["action_cols"].astype(str).tolist()
    old = [line.strip() for line in cfg_path(cfg, "network.control_enabled_actuator_ids_file").read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    order = CanonicalActionOrder.from_global_registry(global_ids, old)
    cache36 = np.load(ROOT / "outputs/cache_v8_storage_variablepump/transition_cache.npz", allow_pickle=True)["action_cols"].astype(str).tolist()
    audit = __import__("pandas").read_csv(cfg_path(cfg, "outputs.audit") / "actuator_table.csv")
    online = select_actuators_for_scope(audit, "control_enabled")["actuator_id"].astype(str).tolist()
    assert cache36 == [f"a:{x}" for x in order.canonical_ids]
    assert online == list(order.canonical_ids)
    synthetic = np.arange(109, dtype=np.float32).reshape(1, 1, 109)
    assert np.max(np.abs(order.project_global109(order.expand_to_global109(order.project_global109(synthetic))) - order.project_global109(synthetic))) == 0.0


def test_actual_pump_semantics_are_preserved_in_canonical_mapping():
    cfg = load_config(ROOT / "configs/wuhan_project6_v8_storage_36.yaml")
    global_ids = np.load(ROOT / "outputs/cache_all109/transition_cache.npz", allow_pickle=True)["action_cols"].astype(str).tolist()
    old = [line.strip() for line in cfg_path(cfg, "network.control_enabled_actuator_ids_file").read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    order = CanonicalActionOrder.from_global_registry(global_ids, old)
    values = order.align_action_dict({"ADD301.2": 0.0, "ADD301.3": 1.0, "add350.1": 0.37})
    mapping = {aid: values[i] for i, aid in enumerate(order.canonical_ids)}
    assert mapping["ADD301.2"] in {0.0, 1.0}
    assert mapping["ADD301.3"] in {0.0, 1.0}
    assert np.isclose(mapping["add350.1"], 0.37, atol=1.0e-7)
