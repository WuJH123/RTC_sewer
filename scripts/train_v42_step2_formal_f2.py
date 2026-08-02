"""Formal F2 trainer for the four-reference hydraulic surrogate.

Input must already have raw four-reference admission, actual Engineering36
readback, causal 13-frame sparse-GAT history and no realised-future leakage.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,torch
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
from scripts.train_v42_step2_fast import _batch_indices,_evaluate,_forward,_hash_model,_slice,_targets,_tensorise
from sewerrtc.v4.formal_f2 import FORMAL_GENERATION_ID,read_table
from sewerrtc.v4.models_v42.hydraulic_multi_reference import MultiReferenceHydraulicSurrogate
from sewerrtc.v4.models_v42.hydraulic_trajectory_losses import HydraulicLossWeights,HydraulicTrajectoryLoss
from sewerrtc.v4.v42_priority_contract import get_pfv_core_node_indices
from sewerrtc.v4.v42_trajectory_builder import _load_graph_topology
def _rank(groups:list[str],seed:int)->list[str]:return sorted(groups,key=lambda g:(hashlib.sha256(f'formal-f2:{seed}:{g}'.encode()).hexdigest(),g))
def _split(f,seed,mintrain):
 groups=sorted(f.split_group_key.astype(str).unique())
 if len(groups)<mintrain+4:raise RuntimeError(f'formal Step2 needs >={mintrain+4} groups; got {len(groups)}')
 r=_rank(groups,seed);n=len(r);nv=max(2,round(.1*n));nc=max(2,round(.1*n));nv=int(nv);nc=int(nc)
 while n-nv-nc<mintrain and (nv>2 or nc>2):
  if nv>=nc and nv>2:nv-=1
  elif nc>2:nc-=1
 if n-nv-nc<mintrain:raise RuntimeError('formal Step2 cannot maintain minimum train groups')
 vg=r[:nv];cg=r[nv:nv+nc];tg=r[nv+nc:];return f[f.split_group_key.astype(str).isin(tg)].copy(),f[f.split_group_key.astype(str).isin(vg)].copy(),f[f.split_group_key.astype(str).isin(cg)].copy(),tg,vg,cg
def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--project-root',type=Path,default=PROJECT_ROOT);ap.add_argument('--manifest',type=Path,default=PROJECT_ROOT/'outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/step2/FORMAL_F2_STEP2_GAT_MANIFEST.parquet');ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--epochs',type=int,default=30);ap.add_argument('--patience',type=int,default=6);ap.add_argument('--batch-size',type=int,default=4);ap.add_argument('--hidden-dim',type=int,default=64);ap.add_argument('--gat-layers',type=int,default=3);ap.add_argument('--lr',type=float,default=5e-4);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--split-seed',type=int,default=42);ap.add_argument('--min-train-groups',type=int,default=65);a=ap.parse_args();f=read_table(a.manifest)
 if f.empty:raise ValueError('formal Step2 GAT manifest is empty')
 for c in ('training_admission_authorized','raw_independent_oracle_all_pass','actual_readback_verified'):
  if c not in f or not bool(f[c].astype(bool).all()):raise RuntimeError(f'formal Step2 requires all {c}=True')
 for c,e in {'state_source':'gat_sparse_reconstruction','history_input_contract':'gat_compatible_causal_state','reconstructor_contract':'formal_temporal_v42','reconstructed_history_contract':'PROJECT6_V42_CAUSAL_RECONSTRUCTED_HISTORY_V1'}.items():
  if c not in f or not bool(f[c].astype(str).eq(e).all()):raise RuntimeError(f'formal Step2 {c} contract mismatch')
 for c in ('current_frame_repetition_used','authoritative_swmm_history_used_as_online_input','realized_future_rainfall_used_online'):
  if c not in f or bool(f[c].astype(bool).any()):raise RuntimeError(f'formal Step2 leakage contract violated: {c}')
 trf,vaf,caf,tg,vg,cg=_split(f,a.split_seed,a.min_train_groups);tr=_tensorise(trf);va=_tensorise(vaf);ca=_tensorise(caf);torch.manual_seed(a.seed);np.random.seed(a.seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');g=_load_graph_topology(a.project_root);ei=torch.from_numpy(g['edge_index'].astype(np.int64)).to(device);ns=torch.from_numpy(g['node_static'].astype(np.float32)).to(device);am=torch.from_numpy(g['action_node_map'].astype(np.float32)).to(device);pri=torch.as_tensor(get_pfv_core_node_indices(list(g['node_ids'])),dtype=torch.long,device=device);model=MultiReferenceHydraulicSurrogate(n_nodes=int(g['n_nodes']),n_facilities=int(g['n_facilities']),state_feature_dim=1,static_feature_dim=int(g['node_static'].shape[1]),hidden_dim=a.hidden_dim,gat_heads=4,gat_layers=a.gat_layers,horizon=12).to(device);opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4);lossfn=HydraulicTrajectoryLoss(HydraulicLossWeights(depth=.5,node_flooding=2.,storage=0.,facility_flow=0.,outfall_flow=0.,kpi_consistency=.75),require_storage_targets=False,require_facility_flow_targets=False,require_outfall_flow_targets=False);a.output_dir.mkdir(parents=True,exist_ok=True);best=a.output_dir/'best_model.pt';history=[];bestloss=float('inf');stale=0
 for epoch in range(1,a.epochs+1):
  model.train();running=0.;seen=0
  for idx in _batch_indices(len(trf),a.batch_size,shuffle=True,seed=a.seed+epoch):
   b=_slice(tr,idx);opt.zero_grad(set_to_none=True);p=_forward(model,b,(ei,ns,am),pri,device);target=_targets(b,device);losses=lossfn(p,target);loss=lossfn.total(losses);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step();running+=float(loss.detach().item())*len(idx);seen+=len(idx)
  vr=_evaluate(model,va,(ei,ns,am),pri,device,a.batch_size,lossfn);row={'epoch':epoch,'train_loss':running/max(1,seen),'validation':vr};history.append(row);print(json.dumps(row,allow_nan=False),flush=True)
  if float(vr['loss'])<bestloss:bestloss=float(vr['loss']);stale=0;torch.save(model.state_dict(),best)
  else:
   stale+=1
   if stale>=a.patience:break
 model.load_state_dict(torch.load(best,map_location=device,weights_only=True));trainrep=_evaluate(model,tr,(ei,ns,am),pri,device,a.batch_size,lossfn);valrep=_evaluate(model,va,(ei,ns,am),pri,device,a.batch_size,lossfn);calrep=_evaluate(model,ca,(ei,ns,am),pri,device,a.batch_size,lossfn);report={'formal_generation_id':FORMAL_GENERATION_ID,'stage':'formal_f2_step2_single_seed','status':'pass','development_only':False,'formal_mainline_authorized':False,'formal_model':'MultiReferenceHydraulicSurrogate','four_reference_shared_model':True,'trajectory_first_kpi_derivation':True,'training_admission_authorized':True,'raw_independent_oracle_all_pass':True,'action_authority':'actual_readback_setting','history_input_contract':'gat_compatible_causal_state','rainfall_group_isolated_split':True,'formal_target_domain_only':True,'outfall_supervised':False,'storage_supervised':False,'facility_flow_supervised':False,'seed':a.seed,'split_seed':a.split_seed,'train_cases':len(trf),'validation_cases':len(vaf),'calibration_cases':len(caf),'train_rainfall_groups':tg,'validation_rainfall_groups':vg,'calibration_rainfall_groups':cg,'train_rainfall_group_count':len(tg),'validation_rainfall_group_count':len(vg),'calibration_rainfall_group_count':len(cg),'surrogate_model_sha256':_hash_model(model),'train':trainrep,'validation':valrep,'calibration':calrep,'history':history};(a.output_dir/'formal_step2_report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8');(a.output_dir/'split_groups.json').write_text(json.dumps({'train':tg,'validation':vg,'calibration':cg},indent=2),encoding='utf-8');return 0
if __name__=='__main__':raise SystemExit(main())
