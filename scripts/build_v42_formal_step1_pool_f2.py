"""Expand Formal F2 Step1 with all physically compatible historical trajectories.

Uses structured F2 source rows and completion metadata only. Step1 does not
require four-reference labels; compatible physical detail files are SHA-
deduplicated and timeline-spread to avoid overweighting duplicate cases.
"""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np,pandas as pd
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
from sewerrtc.v4.formal_f2 import FORMAL_GENERATION_ID,explicit_step1_roles,read_table,sha256_file,text
from sewerrtc.v4.v42_step1_dataset import _build_usecols,load_graph_assets
ROLE_ALIASES=('candidate','no_control','dynamic_internal','dynamic_internal_rules','hold_previous')
def _index(root:Path):
 try:r=subprocess.run(['rg','--files','-uu','-g','completion.json',str(root)],capture_output=True,text=True,check=False);paths=[Path(x) for x in r.stdout.splitlines() if x.strip()] if r.returncode in (0,1) else []
 except FileNotFoundError:paths=list(root.rglob('completion.json'))
 out={}
 for p in paths:
  try:q=json.loads(p.read_text(encoding='utf-8'))
  except Exception:continue
  cid=text(q.get('case_id',''))
  if cid:out.setdefault(cid,[]).append(p)
 return out
def _detail(completion:Path):
 try:p=json.loads(completion.read_text(encoding='utf-8'))
 except Exception:return None
 b=p.get('branches',{})
 if not isinstance(b,dict):return None
 for role in ROLE_ALIASES:
  v=b.get(role)
  if v is None:continue
  raw=v if isinstance(v,str) else text(v.get('detail_path') or v.get('path') or v.get('detail')) if isinstance(v,dict) else ''
  if not raw:continue
  q=Path(raw)
  for c in (q,completion.parent/q,completion.parent/q.name):
   if c.exists():return c.resolve()
 return None
def _anchors(path:Path,limit:int):
 x=pd.to_numeric(pd.read_csv(path,usecols=['elapsed_min']).elapsed_min,errors='coerce').dropna().to_numpy(float);times={round(float(v),6) for v in x};valid=[a for a in sorted(times) if {round(a-60+5*i,6) for i in range(13)}.issubset(times)]
 if len(valid)<=limit:return valid
 idx=np.linspace(0,len(valid)-1,limit,dtype=int);return [valid[i] for i in sorted(set(idx))]
def _prefix(a:Path,b:Path)->int:
 n=0
 for x,y in zip(a.resolve().parts,b.resolve().parts):
  if x.casefold()!=y.casefold():break
  n+=1
 return n
def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--project-root',type=Path,default=PROJECT_ROOT);ap.add_argument('--source-rows',type=Path,default=PROJECT_ROOT/'outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/prepare/FORMAL_F2_SOURCE_ROWS.parquet');ap.add_argument('--ledger',type=Path,default=PROJECT_ROOT/'outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/prepare/FORMAL_F2_EVENT_LEDGER.csv');ap.add_argument('--base-step1-manifest',type=Path,default=PROJECT_ROOT/'outputs/project6_dual_reference_v4/final_v4/v42_paper/step1_gat/dataset/step1_window_manifest.parquet');ap.add_argument('--output-manifest',type=Path,default=PROJECT_ROOT/'outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/prepare/FORMAL_F2_STEP1_WINDOW_MANIFEST.parquet');ap.add_argument('--output-root',type=Path,default=PROJECT_ROOT/'outputs/project6_dual_reference_v4');ap.add_argument('--max-windows-per-physical-run',type=int,default=4);ap.add_argument('--validation-fraction',type=float,default=.15);ap.add_argument('--split-seed',type=int,default=42);ap.add_argument('--min-target-train-groups',type=int,default=65);a=ap.parse_args()
 src=read_table(a.source_rows);ledger=read_table(a.ledger);graph=load_graph_assets(a.project_root);required=set(_build_usecols(graph.node_ids,graph.facility_ids));index=_index(a.output_root);records=[];fail=[];cache={}
 if a.base_step1_manifest.exists():
  base=read_table(a.base_step1_manifest)
  for _,r in base.iterrows():d=r.to_dict();d['source_dataset']=text(d.get('source_dataset','legacy_step1_manifest'));records.append(d)
 eligible=src[src.get('formal_step1_allowed',pd.Series(False,index=src.index)).astype(bool)].copy()
 for _,r in eligible.iterrows():
  group=text(r.get('rainfall_group_key',''));cid=text(r.get('case_id',''));raw=text(r.get('detail_path',''));path=Path(raw) if raw else None
  try:
   if path is None or not path.exists():
    options=index.get(cid,[])
    if not options:raise FileNotFoundError(f'no completion for case_id={cid!r}')
    sm=Path(text(r.get('source_manifest','')));options=sorted(options,key=lambda p:(-_prefix(p,sm),str(p).casefold()));path=next((q for q in (_detail(x) for x in options) if q is not None),None)
   if path is None or not path.exists():raise FileNotFoundError('no physical detail')
   header=set(map(str,pd.read_csv(path,nrows=0).columns));missing=sorted(required-header)
   if missing:raise KeyError(f'missing Step1 columns {missing[:8]}')
   key=str(path.resolve());cache.setdefault(key,sha256_file(path));psha=cache[key]
   for anchor in _anchors(path,a.max_windows_per_physical_run):records.append({'detail_path':key,'anchor_min':float(anchor),'split_group_key':group,'rainfall_sha256':group,'physical_identity_sha256':psha,'source_dataset':text(r.get('source_id','historical')),'formal_generation_id':FORMAL_GENERATION_ID})
  except Exception as exc:fail.append({'source_id':text(r.get('source_id','')),'case_id':cid,'rainfall_group':group,'error':f'{type(exc).__name__}: {exc}'})
 out=pd.DataFrame(records)
 if out.empty:raise RuntimeError('no Formal F2 Step1 windows')
 if 'rainfall_sha256' in out:
  x=out.rainfall_sha256.fillna('').astype(str).str.strip();use=x.ne('');out.loc[use,'split_group_key']=x.loc[use]
 out=out.drop_duplicates(['physical_identity_sha256','anchor_min','split_group_key']).reset_index(drop=True);out=explicit_step1_roles(out,ledger,validation_fraction=a.validation_fraction,split_seed=a.split_seed);train=int(out.loc[out.step1_domain_role.eq('target_formal')&out.formal_split.eq('train'),'split_group_key'].astype(str).nunique());val=int(out.loc[out.step1_domain_role.eq('target_formal')&out.formal_split.eq('validation'),'split_group_key'].astype(str).nunique());aux=int(out.loc[out.step1_domain_role.eq('auxiliary_pretrain'),'split_group_key'].astype(str).nunique());a.output_manifest.parent.mkdir(parents=True,exist_ok=True);out.to_parquet(a.output_manifest,index=False);audit={'formal_generation_id':FORMAL_GENERATION_ID,'stage':'formal_f2_step1_pool','status':'pass' if train>=a.min_target_train_groups else 'fail','rows':len(out),'physical_runs':int(out.physical_identity_sha256.astype(str).nunique()),'target_train_rainfall_groups':train,'target_validation_rainfall_groups':val,'auxiliary_rainfall_groups':aux,'minimum_target_train_groups':a.min_target_train_groups,'failed_source_rows':len(fail),'failure_examples':fail[:200]};(a.output_manifest.parent/'FORMAL_F2_STEP1_POOL_AUDIT.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');print(json.dumps(audit,indent=2,ensure_ascii=False,allow_nan=False),flush=True);return 0 if audit['status']=='pass' else 3
if __name__=='__main__':raise SystemExit(main())
