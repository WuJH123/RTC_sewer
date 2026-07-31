from __future__ import annotations

import argparse
import json
import itertools
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config


def _effect_summary(data: pd.DataFrame) -> pd.DataFrame:
    effects = ["effect_PFV_H", "effect_TFV_H", "effect_peak_TFV_rate_H"]
    rows=[]
    for actuator, group in data.groupby("actuator_id"):
        values = {name: pd.to_numeric(group.get(name), errors="coerce") for name in effects}
        signs = pd.concat(values.values(), axis=0).dropna()
        consistency = float(max((signs < 0).mean(), (signs > 0).mean())) if len(signs) else np.nan
        n_events=int(group["event_id"].nunique()); level="very_low" if n_events < 2 else ("low" if n_events < 3 else "exploratory")
        rows.append({"actuator_id":actuator,"n_independent_events":n_events,"n_phases":int(group["phase"].nunique()),"PFV_effect_mean":values[effects[0]].mean(),"PFV_effect_median":values[effects[0]].median(),"TFV_effect_mean":values[effects[1]].mean(),"TFV_effect_median":values[effects[1]].median(),"peak_effect_mean":values[effects[2]].mean(),"peak_effect_median":values[effects[2]].median(),"direction_consistency":consistency,"effect_magnitude":float(np.nanmedian(np.abs(pd.concat(values.values(),axis=0)))) if len(signs) else np.nan,"evidence_level":level})
    return pd.DataFrame(rows).sort_values(["evidence_level","effect_magnitude"],ascending=[True,False])


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/wuhan_project6.yaml"); args=ap.parse_args()
    cfg=load_config(args.config); root=cfg_path(cfg,"project_root"); ab=root/"outputs/ablation_all109"; audits=root/"outputs/audits"
    data=pd.read_csv(ab/"exact_no_control_action_effect_dataset.csv")
    partial={"coverage_complete":False,"intended_cases":6540,"completed_cases":int(data["case_id"].nunique()),"complete_events":1,"partial_events":1,"delta":0.05,"hold_steps":2,"intended_use":"preliminary_local_screening_only","prohibited_use":"permanent actuator whitelist or formal reliability ranking"}
    (ab/"partial_dataset_manifest.json").write_text(json.dumps(partial,indent=2),encoding="utf-8")
    summary=_effect_summary(data); summary.to_csv(ab/"partial_local_effect_summary.csv",index=False)
    semantics=pd.read_csv(audits/"actuator_control_semantics.csv")
    existing=semantics[semantics["real_or_virtual"].eq("existing_asset")].copy()
    # The current retrofit configuration is the authoritative operational
    # registry.  Phase-1's old audit pre-dated the verified VFD declaration
    # and incorrectly exposed only one binary pump.
    controller_cfg = cfg.get("controller", {}) or {}
    vfd_pumps = [str(x) for x in controller_cfg.get("variable_speed_pump_ids", [])]
    enabled = {str(x) for x in controller_cfg.get("allowed_actuator_ids", [])}
    selected_ids=["add350.1", "ADD301.2", "cc006.1", "jichangheTank.2", "ADD424.1", "dwxh.2"]
    missing_ids = [aid for aid in selected_ids if aid not in set(semantics["actuator_id"].astype(str))]
    if missing_ids:
        raise ValueError(f"Targeted assets absent from actuator semantics audit: {missing_ids}")
    selected=semantics.set_index("actuator_id").loc[selected_ids].reset_index().copy()
    if enabled:
        not_enabled = [aid for aid in selected_ids if aid not in enabled]
        if not_enabled:
            raise ValueError(f"Targeted assets are not enabled in the current controller scope: {not_enabled}")
    selected.loc[selected["actuator_id"].isin(vfd_pumps), "control_semantics"] = "variable_speed_pump"
    selected.loc[selected["actuator_id"].isin(vfd_pumps), "continuous_or_binary"] = "continuous"
    selected.loc[selected["actuator_id"].isin(vfd_pumps), "semantic_evidence"] = "Current storage-retrofit config declares verified variable-speed operation."
    reasons={"add350.1":"verified variable-speed pump","ADD301.2":"verified variable-speed pump","cc006.1":"storage-adjacent inlet/orifice candidate","jichangheTank.2":"storage-adjacent outlet/orifice candidate","ADD424.1":"ordinary existing weir/regulator","dwxh.2":"downstream outlet corridor regulator"}
    selected["selection_reason"]=selected["actuator_id"].map(reasons)
    selected["selection_status"]="eligible_existing_asset"
    selected.loc[selected["real_or_virtual"].eq("virtual_or_planning_asset"), "selection_status"] = "eligible_retrofit_asset"
    missing={"required_role":"second_pump","available_verified_vfd_pumps":vfd_pumps,"status":"resolved"}
    events=["T50_D210_chicago_center","T50_D210_chicago_late","T50_D210_block","T50_D210_double_peak"]
    phases=["rising","peak","recession"]; deltas=[-0.20,-0.10,0.10,0.20]
    rows=[]
    for aid in selected["actuator_id"]:
        for event in events:
            for phase in phases:
                for delta in deltas:
                    rows.append({"experiment":"single","actuator_id":aid,"event_id":event,"phase":phase,"delta":delta,"hold_steps":2,"save_case_details":True,"detail_format":"parquet","launch_allowed":True,"status":"ready"})
    pair_assets=["add350.1","ADD301.2","cc006.1","jichangheTank.2"]
    for left, right in itertools.combinations(pair_assets, 2):
        for delta_left, delta_right in itertools.product((-0.10, 0.10), repeat=2):
            for event in events:
                for phase in phases:
                    rows.append({"experiment":"pairwise","actuator_id":f"{left}+{right}","actuator_a":left,"actuator_b":right,"event_id":event,"phase":phase,"delta_a":delta_left,"delta_b":delta_right,"hold_steps":2,"save_case_details":True,"detail_format":"parquet","launch_allowed":True,"status":"ready"})
    manifest=pd.DataFrame(rows); manifest.to_csv(root/"outputs/targeted_experiment_manifest.csv",index=False)
    resolved_ids=",".join(selected_ids)
    reliability = "outputs\\targeted_v8_storage_single\\actuator_dynamic_reliability.csv"
    report={"manifest":str(root/"outputs/targeted_experiment_manifest.csv"),"selected_existing_assets":selected.to_dict(orient="records"),"missing_requirement":missing,"intended_single_cases":288,"intended_pairwise_cases":288,"intended_total_cases":576,"launch_allowed":True,"expected_commands":[f"& $Py scripts\\76_generate_no_control_single_actuator_ablation.py --config configs\\wuhan_project6_v8_storage.yaml --event-ids T50_D210_chicago_center,T50_D210_chicago_late,T50_D210_block,T50_D210_double_peak --max-events 4 --actuator-ids {resolved_ids} --samples-per-phase 1 --delta-levels 0.10,0.20 --hold-steps 2 --workers 16 --keep-details --out-dir outputs\\targeted_v8_storage_single",f"& $Py scripts\\77_generate_no_control_joint_action_ablation.py --config configs\\wuhan_project6_v8_storage.yaml --event-ids T50_D210_chicago_center,T50_D210_chicago_late,T50_D210_block,T50_D210_double_peak --max-events 4 --actuator-ids ADD301.2,cc006.1,ADD424.1,dwxh.2 --samples-per-phase 1 --max-group-size 2 --max-combinations-per-phase 6 --max-action-amplitude 0.20 --hold-steps 2 --workers 16 --keep-details --allow-pilot-evidence --reliability {reliability} --out-dir outputs\\targeted_v8_storage_pairwise"],"reason":"Six current control-enabled assets cover two verified variable-speed pumps, storage inlet/outlet, a weir, and a downstream regulator."}
    (root/"outputs/targeted_experiment_plan.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
