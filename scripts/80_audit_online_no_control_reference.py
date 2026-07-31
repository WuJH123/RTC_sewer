from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from sewerrtc.control.no_control_reference_predictor import OnlineNoControlReferencePredictor
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config
from sewerrtc.models.raw_joint_action_surrogate import RawJointActionSurrogate


def _trainer_module(root: Path):
    path = root / "scripts" / "79_train_raw_joint_action_surrogate.py"
    spec = importlib.util.spec_from_file_location("raw_joint_trainer", path)
    module = importlib.util.module_from_spec(spec); assert spec and spec.loader; spec.loader.exec_module(module)
    return module


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/wuhan_project6.yaml"); ap.add_argument("--dataset", required=True); ap.add_argument("--model", required=True); ap.add_argument("--out-dir", default="outputs/audits")
    args=ap.parse_args(); cfg=load_config(args.config); root=cfg_path(cfg,"project_root"); dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt=torch.load(args.model,map_location=dev,weights_only=False); data=np.load(args.dataset,allow_pickle=True)
    trainer=_trainer_module(root); node_static,edge,amap,afeat,priority,storage=trainer._assets(cfg,ckpt["node_ids"],ckpt["action_ids"])
    model=RawJointActionSurrogate(n_nodes=len(ckpt["node_ids"]),n_actions=len(ckpt["action_ids"]),node_static_dim=node_static.shape[1],actuator_feature_dim=afeat.shape[1],horizon_steps=int(ckpt["horizon_steps"]),hidden_dim=int(ckpt["hidden_dim"]),heads=4).to(dev); model.load_state_dict(ckpt["model"]); model.eval()
    predictor=OnlineNoControlReferencePredictor(model); n=len(data["state"])
    with torch.no_grad():
        out=predictor.predict(state=torch.as_tensor(data["state"],device=dev),reference_action_seq=torch.as_tensor(data["reference_action_seq"],device=dev),rain_seq=torch.as_tensor(data["rain_seq"],device=dev),actuator_mask=torch.ones(n,len(ckpt["action_ids"]),device=dev),actuator_features=torch.as_tensor(afeat,device=dev),node_static=torch.as_tensor(node_static,device=dev),edge_index=torch.as_tensor(edge,device=dev),action_node_map=torch.as_tensor(amap,device=dev),priority_indices=torch.as_tensor(priority,dtype=torch.long,device=dev),storage_indices=torch.as_tensor(storage,dtype=torch.long,device=dev))
        pred=out["reference_risk_rate_seq"].cpu().numpy(); true=data["reference_risk_rate_seq"]
    report={"model":str(args.model),"dataset":str(args.dataset),"reference_source":out["reference_source"],"offline_consistency_mae_rate":{"PFV":float(np.abs(pred[:,:,0]-true[:,:,0]).mean()),"TFV":float(np.abs(pred[:,:,1]-true[:,:,1]).mean()),"peak_proxy":float(np.abs(pred[:,:,2]-true[:,:,2]).mean())},"oracle_inputs_used":False,"status":"development_only"}
    outdir=ensure_dir(root/args.out_dir); (outdir/"online_no_control_reference_audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2))

if __name__=="__main__": main()
