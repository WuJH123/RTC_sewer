"""Prepare Project6 V4.2 Formal F2 metadata and untouched evaluation ledger.

Event inventory is an evaluation-selection authority, never evidence that an
event was historically consumed. Step1 expansion is deliberately a following
stage because it resolves physical trajectories from structured metadata.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from typing import Any
import pandas as pd
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
from sewerrtc.v4.formal_f2 import DEFAULT_COUNTS,FORMAL_GENERATION_ID,FORMAL_TRAIN_MIN_GROUPS,assert_zero_split_overlap,build_event_ledger,canonical_rain_group,explicit_step1_roles,formal_step2_metadata_pool,load_registry,manifest_source_rows,pool_summary,read_table,resolve_source_files,split_overlap_matrix,text
def _write(f:pd.DataFrame,p:Path):p.parent.mkdir(parents=True,exist_ok=True);f.to_parquet(p,index=False) if p.suffix.lower()=='.parquet' else f.to_csv(p,index=False)
def _inventory(root:Path,reg:dict[str,Any]):
 for p in resolve_source_files(root,dict(reg.get('sources',{}).get('event_inventory',{}) or {})):
  if p.suffix.lower() not in {'.csv','.parquet'}:continue
  f=read_table(p);g={canonical_rain_group(r) for r in f.to_dict('records')};g.discard('')
  if g:return f,str(p)
 return pd.DataFrame(),''
def _reserved(root:Path,source:pd.DataFrame):
 adapter=root/'outputs/rainfall_library_v8_storage_variablepump/rainfall_event_table.formal_adapter.json';table=adapter.with_name('rainfall_event_table.csv');events=set();groups=set();audit={'adapter_path':str(adapter),'adapter_found':adapter.exists(),'rainfall_table_path':str(table),'rainfall_table_found':table.exists()}
 if adapter.exists():
  p=json.loads(adapter.read_text(encoding='utf-8'));split=text(p.get('split','')).casefold()
  if any(x in split for x in ('blind','reserved','challenge')):events.update(str(x) for x in p.get('event_ids',[]) if text(x))
  audit.update({'adapter_split':p.get('split'),'reserved_event_count':len(events)})
 if table.exists() and events:
  rain=pd.read_csv(table,low_memory=False)
  if 'event_id' in rain:
   for r in rain[rain.event_id.astype(str).isin(events)].to_dict('records'):
    g=canonical_rain_group(r)
    if g:groups.add(g)
 if events and not source.empty:groups.update(g for g in source.loc[source.event_id.astype(str).isin(events),'rainfall_group_key'].astype(str) if g)
 audit['reserved_rainfall_group_count']=len(groups);return events,groups,audit
def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--project-root',type=Path,default=PROJECT_ROOT);ap.add_argument('--registry',type=Path,default=PROJECT_ROOT/'configs/v42_formal_source_registry_f2.yaml');ap.add_argument('--step1-window-manifest',type=Path,default=PROJECT_ROOT/'outputs/project6_dual_reference_v4/final_v4/v42_paper/step1_gat/dataset/step1_window_manifest.parquet');ap.add_argument('--output-dir',type=Path,default=PROJECT_ROOT/'outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/prepare');ap.add_argument('--seed',type=int,default=42);ap.add_argument('--step1-validation-fraction',type=float,default=.15);ap.add_argument('--min-train-rainfall-groups',type=int,default=FORMAL_TRAIN_MIN_GROUPS);ap.add_argument('--calibration-groups',type=int,default=DEFAULT_COUNTS['calibration']);ap.add_argument('--locked-groups',type=int,default=DEFAULT_COUNTS['locked_validation']);ap.add_argument('--challenge-groups',type=int,default=DEFAULT_COUNTS['challenge']);ap.add_argument('--blind-groups',type=int,default=DEFAULT_COUNTS['formal_blind']);a=ap.parse_args()
 reg=load_registry(a.registry);source_all,audits=manifest_source_rows(a.project_root,reg)
 # Critical: event_inventory has historically_revealed=False and must not make all untouched events look historically consumed.
 historical=source_all[(source_all.historically_revealed.astype(bool)|source_all.formal_step1_allowed.astype(bool)|source_all.formal_step2_allowed.astype(bool))].copy() if not source_all.empty else source_all.copy();events,reserved,reserved_audit=_reserved(a.project_root,historical)
 if events and not historical.empty:historical=historical[~historical.event_id.astype(str).isin(events)].copy()
 inv,inv_path=_inventory(a.project_root,reg);ledger=build_event_ledger(historical,inventory=inv,historical_reserved_groups=sorted(reserved),seed=a.seed,evaluation_counts={'calibration':a.calibration_groups,'locked_validation':a.locked_groups,'challenge':a.challenge_groups,'formal_blind':a.blind_groups});assert_zero_split_overlap(ledger)
 if not a.step1_window_manifest.exists():raise FileNotFoundError(a.step1_window_manifest)
 base=read_table(a.step1_window_manifest)
 if 'rainfall_sha256' in base:
  x=base.rainfall_sha256.fillna('').astype(str).str.strip();use=x.ne('')
  if use.any():base['legacy_split_group_key']=base.split_group_key.astype(str);base.loc[use,'split_group_key']=x.loc[use]
 provisional=explicit_step1_roles(base,ledger,validation_fraction=a.step1_validation_fraction,split_seed=a.seed);step2=formal_step2_metadata_pool(historical,ledger);a.output_dir.mkdir(parents=True,exist_ok=True);_write(historical,a.output_dir/'FORMAL_F2_SOURCE_ROWS.parquet');_write(ledger,a.output_dir/'FORMAL_F2_EVENT_LEDGER.csv');_write(provisional,a.output_dir/'FORMAL_F2_STEP1_WINDOW_MANIFEST.parquet');_write(step2,a.output_dir/'FORMAL_F2_STEP2_METADATA_POOL.parquet');pd.DataFrame(audits).to_csv(a.output_dir/'FORMAL_F2_SOURCE_AUDIT.csv',index=False)
 s=pool_summary(provisional,step2,ledger);s.update({'status':'pass','development_only':False,'formal_mainline_authorized':False,'registry_path':str(a.registry),'event_inventory_path':inv_path,'reserved_audit':reserved_audit,'source_count':len(reg.get('sources',{})),'resolved_manifest_count':sum(1 for x in audits if x.get('status')=='read'),'required_min_train_rainfall_groups':a.min_train_rainfall_groups,'raw_readmission_pending_rows':int(step2.get('raw_readmission_pending',pd.Series(dtype=bool)).astype(bool).sum()) if not step2.empty else 0,'step1_provisional_only':True});reasons=[];warnings=[]
 if s['formal_train_ledger_groups']<a.min_train_rainfall_groups:reasons.append('formal_train_ledger_groups_below_minimum')
 if s['step1_target_train_groups']<a.min_train_rainfall_groups:warnings.append('provisional_step1_target_groups_below_minimum_expand_from_structured_physical_sources')
 if s['step2_train_rainfall_groups']<a.min_train_rainfall_groups:reasons.append('step2_metadata_groups_below_minimum_before_raw_readmission')
 if any(int(v) for v in split_overlap_matrix(ledger).values()):reasons.append('rainfall_split_overlap')
 for role,required in [('calibration',a.calibration_groups),('locked_validation',a.locked_groups),('challenge',a.challenge_groups),('formal_blind',a.blind_groups)]:
  actual=int(s['evaluation_group_counts'].get(role,0))
  if actual<required:reasons.append(f'{role}_untouched_group_shortfall:{actual}<{required}')
 if reasons:s['status']='fail'
 s['reasons']=reasons;s['warnings']=warnings;(a.output_dir/'FORMAL_F2_PREPARE_AUDIT.json').write_text(json.dumps(s,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
 for role in ('train','calibration','locked_validation','challenge','formal_blind'):(a.output_dir/f'{role}_groups.json').write_text(json.dumps({'formal_generation_id':FORMAL_GENERATION_ID,'groups':sorted(ledger.loc[ledger.formal_f2_role.eq(role),'rainfall_group_key'].astype(str))},indent=2),encoding='utf-8')
 print(json.dumps(s,indent=2,ensure_ascii=False,allow_nan=False),flush=True);return 0 if not reasons else 3
if __name__=='__main__':raise SystemExit(main())
