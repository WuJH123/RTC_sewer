from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config
from sewerrtc.models.raw_joint_action_surrogate import RawJointActionSurrogate


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/wuhan_project6.yaml"); ap.add_argument("--dataset",required=True); ap.add_argument("--model",required=True); ap.add_argument("--out-dir",default="outputs/audits"); args=ap.parse_args()
    cfg=load_config(args.config); root=cfg_path(cfg,"project_root"); dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec=importlib.util.spec_from_file_location("trainer",root/"scripts/79_train_raw_joint_action_surrogate.py"); mod=importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(mod)
    ck=torch.load(args.model,map_location=dev,weights_only=False); d=np.load(args.dataset,allow_pickle=True); static,edge,amap,feat,priority,storage=mod._assets(cfg,ck["node_ids"],ck["action_ids"])
    model=RawJointActionSurrogate(n_nodes=len(ck["node_ids"]),n_actions=len(ck["action_ids"]),node_static_dim=static.shape[1],actuator_feature_dim=feat.shape[1],horizon_steps=int(ck["horizon_steps"]),hidden_dim=int(ck["hidden_dim"]),heads=4).to(dev); model.load_state_dict(ck["model"]); model.eval(); n=min(24,len(d["state"]))
    fixed={"state":torch.tensor(d["state"][:n],device=dev),"rain_seq":torch.tensor(d["rain_seq"][:n],device=dev),"actuator_mask":torch.ones(n,len(ck["action_ids"]),device=dev),"actuator_features":torch.tensor(feat,device=dev),"node_static":torch.tensor(static,device=dev),"edge_index":torch.tensor(edge,device=dev),"action_node_map":torch.tensor(amap,device=dev),"priority_indices":torch.tensor(priority,dtype=torch.long,device=dev),"storage_indices":torch.tensor(storage,dtype=torch.long,device=dev)}
    ref=torch.tensor(d["reference_action_seq"][:n],device=dev); cand=torch.tensor(d["candidate_action_seq"][:n],device=dev); a,b=0,1
    def effect(seq): return model(candidate_action_seq=seq,reference_action_seq=ref,**fixed)["delta_risk_rate_seq"].sum(1)
    with torch.no_grad():
        zero=effect(ref); base=effect(cand); swapped=cand.clone(); swapped[:,:,a],swapped[:,:,b]=cand[:,:,b],cand[:,:,a]
        ordered=cand.clone(); ordered[:,:,a]=torch.flip(cand[:,:,a],dims=[1]); identity=torch.abs(base-effect(swapped)).mean(0); temporal=torch.abs(base-effect(ordered)).mean(0)
    report={"model":str(args.model),"samples":n,"zero_action_relative_error":float(zero.abs().sum()/torch.clamp(torch.abs(ref).sum(),min=1.0)),"identity_swap_mean_abs_delta":{"PFV":float(identity[0]),"TFV":float(identity[1]),"peak":float(identity[2])},"temporal_order_mean_abs_delta":{"PFV":float(temporal[0]),"TFV":float(temporal[1]),"peak":float(temporal[2])},"old_model_identity_relative_max":0.0002366847454688128,"status":"development_only"}
    out=ensure_dir(root/args.out_dir); (out/"raw_joint_action_sequence_semantics_summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2))

if __name__=="__main__": main()
