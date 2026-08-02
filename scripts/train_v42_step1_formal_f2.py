"""Formal F2 Temporal Sparse GAT trainer.

Consumes explicit F2 roles; never infers formal target status from domain_id.
One model seed per invocation. Historical training groups provide internal model
calibration; F2 Calibration/Locked/Challenge/Blind stay untouched.
"""
from __future__ import annotations
import argparse,hashlib,json,math,sys
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
from scripts.train_v42_step1_streaming import _graph_tensors,_run_epoch,_state_hash
from sewerrtc.models.temporal_sparse_gat_v42 import TemporalSparseGATReconstructorV42
from sewerrtc.v4.formal_f2 import FORMAL_GENERATION_ID,read_table
from sewerrtc.v4.v42_step1_streaming import Step1StreamingDataset
from sewerrtc.v4.v42_step1_training import Step1LossWeights

def _rank(groups:list[str],seed:int,salt:str)->list[str]:return sorted(groups,key=lambda g:(hashlib.sha256(f'{salt}:{seed}:{g}'.encode()).hexdigest(),g))
def _scores(model,dataset,graph,device,max_batches=200):
 loader=DataLoader(dataset,batch_size=16,shuffle=False,num_workers=0);ns,ls,ei,am=_graph_tensors(graph,device);ratios=[];scores=[]
 with torch.no_grad():
  for bi,b in enumerate(loader):
   if bi>=max_batches:break
   sdh=b['sparse_depth_history'].to(device);smh=b['sensor_mask_history'].to(device);rain=b['rainfall_history'].to(device);actions=b['historical_actions'].to(device);target=b['target_depth'].to(device);o=model(sparse_depth_history=sdh,sensor_mask_history=smh,rainfall_history=rain,historical_actions=actions,node_static=ns,link_static=ls,edge_index=ei,action_node_map=am);unobs=~(smh[:,-1,:]>=.5);err=(o.depth_mean-target).abs();std=torch.clamp(o.depth_std,min=1e-6);ratios.append((err[unobs]/std[unobs]).cpu().numpy());masked=std.masked_fill(~unobs,0.);scores.append((masked.sum(1)/unobs.sum(1).clamp(min=1)).cpu().numpy())
 return np.concatenate(ratios).astype(float) if ratios else np.zeros(0),np.concatenate(scores).astype(float) if scores else np.zeros(0)
def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--project-root',type=Path,default=PROJECT_ROOT);ap.add_argument('--manifest',type=Path,default=PROJECT_ROOT/'outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/prepare/FORMAL_F2_STEP1_WINDOW_MANIFEST.parquet');ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--model-seed',type=int,default=42);ap.add_argument('--split-seed',type=int,default=42);ap.add_argument('--sensor-layout-seed',type=int,default=42);ap.add_argument('--sensor-ratio',type=float,default=.1);ap.add_argument('--epochs',type=int,default=30);ap.add_argument('--patience',type=int,default=6);ap.add_argument('--batch-size',type=int,default=16);ap.add_argument('--hidden-dim',type=int,default=128);ap.add_argument('--heads',type=int,default=4);ap.add_argument('--gat-layers',type=int,default=3);ap.add_argument('--lr',type=float,default=3e-4);ap.add_argument('--priority-weight',type=float,default=1.);ap.add_argument('--wet-priority-weight',type=float,default=2.);ap.add_argument('--nll-weight',type=float,default=.1);ap.add_argument('--wet-threshold-m',type=float,default=.05);ap.add_argument('--aux-epochs',type=int,default=2);ap.add_argument('--aux-max-windows-per-group',type=int,default=32);ap.add_argument('--aux-max-windows-per-run',type=int,default=4);ap.add_argument('--min-train-groups',type=int,default=65);ap.add_argument('--internal-calibration-fraction',type=float,default=.08);a=ap.parse_args()
 f=read_table(a.manifest);req={'split_group_key','formal_split','step1_domain_role'}
 if not req.issubset(f):raise KeyError(f'formal Step1 manifest missing {sorted(req-set(f.columns))}')
 target=f[f.step1_domain_role.astype(str).eq('target_formal')&f.formal_split.astype(str).isin(['train','validation'])];train_candidates=sorted(target.loc[target.formal_split.eq('train'),'split_group_key'].astype(str).unique());val_groups=sorted(target.loc[target.formal_split.eq('validation'),'split_group_key'].astype(str).unique())
 if not val_groups:raise RuntimeError('formal Step1 has no explicit validation rainfall groups')
 ranked=_rank(train_candidates,a.split_seed,'step1-cal');ncal=max(2,int(round(a.internal_calibration_fraction*len(ranked))));ncal=min(ncal,max(0,len(ranked)-a.min_train_groups))
 if ncal<1:raise RuntimeError('formal Step1 cannot reserve calibration and keep >=min train groups')
 cal_groups=ranked[:ncal];train_groups=ranked[ncal:]
 if len(train_groups)<a.min_train_groups or set(train_groups)&set(val_groups) or set(cal_groups)&set(val_groups):raise RuntimeError('formal Step1 rainfall split/minimum violation')
 torch.manual_seed(a.model_seed);np.random.seed(a.model_seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(a.model_seed)
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');common=dict(project_root=a.project_root,manifest_path=a.manifest,sensor_ratio=a.sensor_ratio,sensor_layout_seed=a.sensor_layout_seed,domain_roles=('target_formal',));train_ds=Step1StreamingDataset(**common,allowed_groups=train_groups,shuffle_files=True,iteration_seed=a.model_seed);val_ds=Step1StreamingDataset(**common,allowed_groups=val_groups,shuffle_files=False,iteration_seed=a.model_seed);cal_ds=Step1StreamingDataset(**common,allowed_groups=cal_groups,shuffle_files=False,iteration_seed=a.model_seed)
 if len({train_ds.sensor_layout_sha256,val_ds.sensor_layout_sha256,cal_ds.sensor_layout_sha256})!=1:raise RuntimeError('sensor layout changed across formal Step1 splits')
 graph=train_ds.graph;model=TemporalSparseGATReconstructorV42(n_nodes=graph.n_nodes,n_facilities=graph.n_facilities,node_static_dim=graph.node_static.shape[1],link_static_dim=graph.link_static.shape[1],hidden_dim=a.hidden_dim,heads=a.heads,gat_layers=a.gat_layers).to(device);opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4);aux=f[f.step1_domain_role.astype(str).eq('auxiliary_pretrain')];aux_groups=sorted(aux.split_group_key.astype(str).unique());aux_hist=[]
 if aux_groups and a.aux_epochs>0:
  aux_ds=Step1StreamingDataset(project_root=a.project_root,manifest_path=a.manifest,sensor_ratio=a.sensor_ratio,sensor_layout_seed=a.sensor_layout_seed,domain_roles=('auxiliary_pretrain',),allowed_groups=aux_groups,max_windows_per_group=a.aux_max_windows_per_group,max_windows_per_physical_run=a.aux_max_windows_per_run,sampling_seed=a.model_seed,shuffle_files=True,iteration_seed=a.model_seed)
  for epoch in range(1,a.aux_epochs+1):aux_hist.append(_run_epoch(model=model,dataset=aux_ds,graph=graph,device=device,batch_size=a.batch_size,num_workers=0,optimizer=opt,weights=Step1LossWeights(priority_depth=0.,wet_priority_depth=0.,heteroscedastic_nll=0.),wet_threshold_m=a.wet_threshold_m,heartbeat_batches=10,epoch=epoch))
 a.output_dir.mkdir(parents=True,exist_ok=True);best=a.output_dir/'best_model.pt';last=a.output_dir/'last_model.pt';history=[];best_rmse=float('inf');stale=0
 for epoch in range(1,a.epochs+1):
  nw=a.nll_weight*min(1.,max(0.,(epoch-2)/5.));w=Step1LossWeights(global_depth=1.,priority_depth=a.priority_weight,wet_priority_depth=a.wet_priority_weight,heteroscedastic_nll=nw);tr=_run_epoch(model=model,dataset=train_ds,graph=graph,device=device,batch_size=a.batch_size,num_workers=0,optimizer=opt,weights=w,wet_threshold_m=a.wet_threshold_m,heartbeat_batches=10,epoch=epoch);va=_run_epoch(model=model,dataset=val_ds,graph=graph,device=device,batch_size=a.batch_size,num_workers=0,optimizer=None,weights=w,wet_threshold_m=a.wet_threshold_m,heartbeat_batches=20,epoch=epoch);rmse=va['overall_unobserved']['rmse'];row={'epoch':epoch,'nll_weight':nw,'train':tr,'validation':va};history.append(row);print(json.dumps(row,allow_nan=False),flush=True);torch.save(model.state_dict(),last)
  if rmse is not None and float(rmse)<best_rmse:best_rmse=float(rmse);stale=0;torch.save(model.state_dict(),best)
  else:
   stale+=1
   if stale>=a.patience:break
 model.load_state_dict(torch.load(best,map_location=device,weights_only=True));zero=Step1LossWeights();final_train=_run_epoch(model=model,dataset=train_ds,graph=graph,device=device,batch_size=a.batch_size,num_workers=0,optimizer=None,weights=zero,wet_threshold_m=a.wet_threshold_m,heartbeat_batches=0,epoch=1);final_val=_run_epoch(model=model,dataset=val_ds,graph=graph,device=device,batch_size=a.batch_size,num_workers=0,optimizer=None,weights=zero,wet_threshold_m=a.wet_threshold_m,heartbeat_batches=0,epoch=1);final_cal=_run_epoch(model=model,dataset=cal_ds,graph=graph,device=device,batch_size=a.batch_size,num_workers=0,optimizer=None,weights=zero,wet_threshold_m=a.wet_threshold_m,heartbeat_batches=0,epoch=1);ratio,scores=_scores(model,cal_ds,graph,device);us=float(np.quantile(ratio,.95)) if ratio.size else None;ood=float(np.quantile(scores,.99)) if scores.size else None
 report={'formal_generation_id':FORMAL_GENERATION_ID,'stage':'formal_f2_step1_single_seed','status':'pass','development_only':False,'formal_mainline_authorized':False,'formal_reconstructor':'TemporalSparseGATReconstructorV42','reconstructor_contract':'formal_temporal_v42','new_formal_training':True,'rainfall_group_isolated_split':True,'action_authority':'actual_readback_setting','uses_future_hydraulic_truth':False,'model_seed':a.model_seed,'split_seed':a.split_seed,'sensor_layout_seed':a.sensor_layout_seed,'sensor_layout_sha256':train_ds.sensor_layout_sha256,'train_rainfall_groups':train_groups,'validation_rainfall_groups':val_groups,'model_calibration_rainfall_groups':cal_groups,'train_rainfall_group_count':len(train_groups),'validation_rainfall_group_count':len(val_groups),'model_calibration_rainfall_group_count':len(cal_groups),'auxiliary_rainfall_group_count':len(aux_groups),'uncertainty_scale_95':us,'uncertainty_calibrated':bool(us is not None and math.isfinite(us)),'ood_score':'mean_predictive_std_unobserved','ood_threshold_99_target':ood,'ood_threshold_calibrated':bool(ood is not None and math.isfinite(ood)),'ood_calibrated':False,'ood_note':'Target threshold frozen. Formal OOD becomes true only after independent source/challenge detection audit.','gat_model_sha256':_state_hash(model),'auxiliary_history':aux_hist,'history':history,'final_train':final_train,'final_validation':final_val,'final_model_calibration':final_cal};(a.output_dir/'formal_step1_report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps(report,indent=2,allow_nan=False),flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
