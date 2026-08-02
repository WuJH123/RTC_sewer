"""Replace Formal F2 Step-2 history with causal sparse-GAT reconstructions.

A history source must match the rainfall/event/checkpoint state, cover t-120..t,
and produce all thirteen real Step1 calls at t-60..t. One failed checkpoint does
not discard other states from the same rainfall group.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from typing import Any
import numpy as np,pandas as pd,torch
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
from sewerrtc.models.temporal_sparse_gat_v42 import TemporalSparseGATReconstructorV42
from sewerrtc.v4.formal_f2 import FORMAL_GENERATION_ID,read_table
from sewerrtc.v4.v42_fast_e2e import make_causal_rainfall_forecast
from sewerrtc.v4.v42_step1_dataset import _build_usecols,_detail_extract_window,_sensor_layout,load_graph_assets

def _detail(path:Path,required:list[str])->pd.DataFrame:
 h=pd.read_csv(path,nrows=0);missing=[c for c in required if c not in set(map(str,h.columns))]
 if missing:raise KeyError(f'formal GAT history detail missing columns: {missing[:10]}')
 return pd.read_csv(path,usecols=required,low_memory=False).loc[:,required]
def _bounds(path:Path)->tuple[float,float]:
 x=pd.to_numeric(pd.read_csv(path,usecols=['elapsed_min']).elapsed_min,errors='coerce').dropna()
 if x.empty:raise ValueError('elapsed_min has no finite values')
 return float(x.min()),float(x.max())
def _signature(detail:pd.DataFrame,cp:float,graph):
 x=_detail_extract_window(detail,cp,graph.node_ids,graph.facility_ids)
 if x is None:raise ValueError('detail cannot reconstruct Step1 window at checkpoint')
 return x['depth_history'][-1].astype(np.float64),x['actions'][-1].astype(np.float64),x['rainfall'][-1:].astype(np.float64)
def _same(a,b)->bool:return all(x.shape==y.shape and np.allclose(x,y,atol=1e-6,rtol=0.) for x,y in zip(a,b))
def _reconstruct(detail:pd.DataFrame,cp:float,model,graph,mask:np.ndarray,device:torch.device):
 anchors=[cp-60.+5.*i for i in range(13)];ex=[]
 for anchor in anchors:
  item=_detail_extract_window(detail,anchor,graph.node_ids,graph.facility_ids)
  if item is None:raise ValueError(f'missing exact causal Step1 window at anchor={anchor:.6f}')
  ex.append(item)
 mh=np.broadcast_to(mask[None,:],(13,graph.n_nodes)).astype(np.float32,copy=True);sparse=np.stack([x['depth_history']*mh for x in ex]).astype(np.float32);masks=np.broadcast_to(mh[None,:,:],(13,13,graph.n_nodes)).copy().astype(np.float32);rain=np.stack([x['rainfall'] for x in ex]).astype(np.float32);actions=np.stack([x['actions'] for x in ex]).astype(np.float32)
 with torch.no_grad():p=model(sparse_depth_history=torch.from_numpy(sparse).to(device),sensor_mask_history=torch.from_numpy(masks).to(device),rainfall_history=torch.from_numpy(rain).to(device),historical_actions=torch.from_numpy(actions).to(device),node_static=torch.from_numpy(graph.node_static).to(device),link_static=torch.from_numpy(graph.link_static).to(device),edge_index=torch.from_numpy(graph.edge_index).to(device),action_node_map=torch.from_numpy(graph.action_node_map).to(device))
 hist=p.depth_mean.detach().cpu().numpy().astype(np.float32);std=p.depth_std.detach().cpu().numpy().astype(np.float32)
 if hist.shape!=(13,graph.n_nodes):raise RuntimeError(f'unexpected formal reconstructed history shape {hist.shape}')
 return hist,std,ex[-1]['rainfall'].astype(np.float32)
def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--project-root',type=Path,default=PROJECT_ROOT);ap.add_argument('--input-manifest',type=Path,default=PROJECT_ROOT/'outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/step2/FORMAL_F2_STEP2_RAW_MANIFEST.parquet');ap.add_argument('--step1-window-manifest',type=Path,default=PROJECT_ROOT/'outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/prepare/FORMAL_F2_STEP1_WINDOW_MANIFEST.parquet');ap.add_argument('--step1-model-dir',type=Path,required=True);ap.add_argument('--output-manifest',type=Path,default=PROJECT_ROOT/'outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/step2/FORMAL_F2_STEP2_GAT_MANIFEST.parquet');ap.add_argument('--min-rainfall-groups',type=int,default=65);ap.add_argument('--sensor-ratio',type=float,default=.1);ap.add_argument('--sensor-layout-seed',type=int,default=42);ap.add_argument('--hidden-dim',type=int,default=128);ap.add_argument('--heads',type=int,default=4);ap.add_argument('--gat-layers',type=int,default=3);a=ap.parse_args()
 frame=read_table(a.input_manifest);windows=read_table(a.step1_window_manifest)
 if frame.empty or windows.empty:raise ValueError('Formal F2 GAT materialisation input is empty')
 if 'training_admission_authorized' not in frame or not bool(frame.training_admission_authorized.astype(bool).all()):raise RuntimeError('formal GAT materialiser requires raw-authorized Step2 rows')
 graph=load_graph_assets(a.project_root);mask,indices,sensor_sha=_sensor_layout(graph.n_nodes,a.sensor_ratio,a.sensor_layout_seed);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=TemporalSparseGATReconstructorV42(n_nodes=graph.n_nodes,n_facilities=graph.n_facilities,node_static_dim=graph.node_static.shape[1],link_static_dim=graph.link_static.shape[1],hidden_dim=a.hidden_dim,heads=a.heads,gat_layers=a.gat_layers).to(device);model.load_state_dict(torch.load(a.step1_model_dir/'best_model.pt',map_location=device,weights_only=True));model.eval();required=_build_usecols(graph.node_ids,graph.facility_ids);cache={};bcache={};out=[];fail=[];success={}
 windows=windows[~windows.get('formal_split',pd.Series('',index=windows.index)).astype(str).isin(['formal_blind','challenge','locked_validation'])].copy()
 for state,grp in frame.groupby('state_key',sort=True):
  first=grp.iloc[0];rain=str(first.split_group_key);event=str(first.get('event_id',''));cp=float(first.checkpoint_min);candidate=Path(str(first.source_detail_path_candidate))
  try:
   ck=str(candidate.resolve());cache.setdefault(ck,_detail(candidate,required));csig=_signature(cache[ck],cp,graph);q=windows[windows.split_group_key.astype(str).eq(rain)].copy()
   if event and 'event_id' in q:
    qe=q[q.event_id.astype(str).eq(event)];q=qe if not qe.empty else q
   history_path=None;history_detail=None
   for raw in sorted(set(q.detail_path.astype(str))):
    p=Path(raw)
    if not p.exists():continue
    key=str(p.resolve())
    try:
     if key not in bcache:bcache[key]=_bounds(p)
     lo,hi=bcache[key]
     if lo>cp-120.+1e-6 or hi<cp-1e-6:continue
     if key not in cache:cache[key]=_detail(p,required)
     d=cache[key]
     if not _same(csig,_signature(d,cp,graph)):continue
     for anchor in [cp-60.+5.*i for i in range(13)]:
      if _detail_extract_window(d,anchor,graph.node_ids,graph.facility_ids) is None:raise ValueError(f'history source misses anchor={anchor}')
     history_path=p.resolve();history_detail=d;break
    except Exception:continue
   if history_path is None or history_detail is None:raise FileNotFoundError('no same-state same-rainfall detail covers checkpoint-120..checkpoint')
   hist,std,observed=_reconstruct(history_detail,cp,model,graph,mask,device);forecast=make_causal_rainfall_forecast(observed)
   for _,row in grp.iterrows():
    rec=row.copy();rec['history_source_detail_path']=str(history_path);rec['history_depth']=json.dumps(hist.tolist(),allow_nan=False);rec['gat_depth_std_history_mean']=float(std.mean());rec['gat_depth_std_current_mean']=float(std[-1].mean());rec['rainfall_forecast']=json.dumps(forecast.tolist(),allow_nan=False);rec['state_source']='gat_sparse_reconstruction';rec['history_input_contract']='gat_compatible_causal_state';rec['reconstructor_contract']='formal_temporal_v42';rec['reconstructed_history_contract']='PROJECT6_V42_CAUSAL_RECONSTRUCTED_HISTORY_V1';rec['current_frame_repetition_used']=False;rec['authoritative_swmm_history_used_as_online_input']=False;rec['realized_future_rainfall_used_online']=False;rec['future_SWMM_trajectories_supervision_only']=True;rec['sensor_layout_sha256']=sensor_sha;out.append(rec)
   success.setdefault(rain,set()).add(str(state))
  except Exception as exc:fail.append({'rainfall_group':rain,'state_key':str(state),'event_id':event,'checkpoint_min':cp,'candidate_detail':str(candidate),'required_start_min':cp-120.,'error':f'{type(exc).__name__}: {exc}'})
 result=pd.DataFrame(out);groups=int(result.split_group_key.astype(str).nunique()) if not result.empty else 0;a.output_manifest.parent.mkdir(parents=True,exist_ok=True)
 if not result.empty:result.to_parquet(a.output_manifest,index=False)
 inp=set(frame.split_group_key.astype(str));got=set(result.split_group_key.astype(str)) if not result.empty else set();audit={'formal_generation_id':FORMAL_GENERATION_ID,'stage':'formal_f2_step2_gat_history','status':'pass' if groups>=a.min_rainfall_groups else 'fail','development_only':False,'formal_mainline_authorized':False,'input_rows':len(frame),'output_rows':len(result),'input_states':int(frame.state_key.astype(str).nunique()),'output_states':int(result.state_key.astype(str).nunique()) if not result.empty else 0,'input_rainfall_groups':len(inp),'output_rainfall_groups':groups,'lost_rainfall_groups':sorted(inp-got),'minimum_rainfall_groups':a.min_rainfall_groups,'failed_states':len(fail),'failure_examples':fail[:200],'groups_with_alternate_successful_state':sum(1 for x in success.values() if x),'sensor_ratio':a.sensor_ratio,'sensor_count':len(indices),'sensor_layout_sha256':sensor_sha,'state_source':'gat_sparse_reconstruction','history_input_contract':'gat_compatible_causal_state','reconstructed_history_contract':'PROJECT6_V42_CAUSAL_RECONSTRUCTED_HISTORY_V1','current_frame_repetition_used':False,'authoritative_swmm_history_used_as_online_input':False,'realized_future_rainfall_used_online':False};(a.output_manifest.parent/'FORMAL_F2_STEP2_GAT_HISTORY_AUDIT.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');print(json.dumps(audit,indent=2,ensure_ascii=False,allow_nan=False),flush=True);return 0 if audit['status']=='pass' else 3
if __name__=='__main__':raise SystemExit(main())
