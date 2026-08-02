"""Select GAT-history-compatible states for the fast core pipeline.

The original fast selector optimised candidate multiplicity but did not prove
that a selected checkpoint can supply nested t-120..t history required by
thirteen real Step1 calls. This metadata-only gate runs before materialisation.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
PROJECT_ROOT=Path(__file__).resolve().parents[1]
def _read(p:Path)->pd.DataFrame:return pd.read_parquet(p) if p.suffix.lower()=='.parquet' else pd.read_csv(p)
def _r(v:float)->float:return round(float(v),6)
def main()->int:
 ap=argparse.ArgumentParser(); ap.add_argument('--core-pool-dir',type=Path,default=PROJECT_ROOT/'outputs/project6_dual_reference_v4/final_v4/v42_paper/fast_e2e_64plus/core_pool'); ap.add_argument('--step1-window-manifest',type=Path,default=PROJECT_ROOT/'outputs/project6_dual_reference_v4/final_v4/v42_paper/step1_gat/dataset/step1_window_manifest.parquet'); ap.add_argument('--min-checkpoint-min',type=float,default=120.); ap.add_argument('--candidates-per-state',type=int,default=3); ap.add_argument('--seed',type=int,default=42); a=ap.parse_args()
 core=_read(a.core_pool_dir/'FAST_CORE_CASE_MANIFEST.parquet'); windows=_read(a.step1_window_manifest)
 req={'rainfall_group_key','counterfactual_state_key','checkpoint_min','candidate_action_signature','event_id'}
 if not req.issubset(core):raise KeyError(f'core manifest missing {sorted(req-set(core.columns))}')
 if not {'detail_path','anchor_min','split_group_key'}.issubset(windows):raise KeyError('Step1 window manifest lacks detail_path/anchor_min/split_group_key')
 windows=windows.copy(); windows['anchor_min']=pd.to_numeric(windows.anchor_min,errors='coerce')
 if 'rainfall_sha256' in windows:
  x=windows.rainfall_sha256.fillna('').astype(str).str.strip(); windows['_rain_group']=np.where(x.ne(''),x,windows.split_group_key.astype(str))
 else: windows['_rain_group']=windows.split_group_key.astype(str)
 windows['_event']=windows.event_id.fillna('').astype(str) if 'event_id' in windows else ''
 index={}
 for (rain,event,detail),grp in windows.groupby(['_rain_group','_event','detail_path'],dropna=False,sort=False):index.setdefault((str(rain),str(event)),{})[str(detail)]={_r(x) for x in grp.anchor_min.dropna().astype(float)}
 core=core.copy();core['checkpoint_min']=pd.to_numeric(core.checkpoint_min,errors='coerce');core=core[core.checkpoint_min.ge(a.min_checkpoint_min)].copy();counts=core.groupby(['rainfall_group_key','counterfactual_state_key']).candidate_action_signature.nunique(); states=counts[counts.ge(a.candidates_per_state)]
 pieces=[];ga=[]
 for rain in sorted(states.index.get_level_values(0).unique()):
  ss=states.loc[rain]; ids=list(ss.index) if isinstance(ss,pd.Series) else [ss.name]; candidates=[]
  for state in ids:
   sub=core[core.rainfall_group_key.astype(str).eq(str(rain))&core.counterfactual_state_key.astype(str).eq(str(state))].copy()
   if sub.empty:continue
   cp=float(sub.checkpoint_min.iloc[0]); event=str(sub.event_id.iloc[0]); required={_r(cp-60.+5.*i) for i in range(13)}; details=index.get((str(rain),event),{}) or index.get((str(rain),''),{}); hist=[p for p,anchors in details.items() if required.issubset(anchors)]
   if hist:candidates.append((-int(sub.candidate_action_signature.astype(str).nunique()),-cp,hashlib.sha256(f'{a.seed}:{rain}:{state}'.encode()).hexdigest(),str(state),sorted(hist)[0],sub))
  if not candidates:ga.append({'rainfall_group_key':str(rain),'gat_compatible':False,'selected_state':'','history_detail':''});continue
  candidates.sort(key=lambda x:x[:4]);_,_,_,state,hist,sub=candidates[0];sub=sub.sort_values(['source_priority','candidate_action_signature'],kind='mergesort').head(max(3,a.candidates_per_state));sub['history_source_detail_path_hint']=hist;pieces.append(sub);ga.append({'rainfall_group_key':str(rain),'gat_compatible':True,'selected_state':state,'history_detail':hist})
 selected=pd.concat(pieces,ignore_index=True) if pieces else core.iloc[0:0].copy();selected.to_parquet(a.core_pool_dir/'FAST_CORE_SELECTED_CASES.parquet',index=False);audit={'input_groups_with_candidate_choice':len(set(states.index.get_level_values(0))),'gat_compatible_groups':int(selected.rainfall_group_key.astype(str).nunique()) if not selected.empty else 0,'selected_states':int(selected.counterfactual_state_key.astype(str).nunique()) if not selected.empty else 0,'selected_rows':len(selected),'groups':ga};(a.core_pool_dir/'FAST_CORE_GAT_COMPATIBILITY_AUDIT.json').write_text(json.dumps(audit,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps(audit,indent=2,allow_nan=False),flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
