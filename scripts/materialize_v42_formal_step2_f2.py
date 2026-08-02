"""Materialise and re-admit the Formal F2 Step-2 four-reference dataset.

Every admitted row is independently rechecked against the current raw SWMM
contract. Candidate/No-control/Dynamic-Internal/Hold-Previous must share forcing
and checkpoint state, expose actual Engineering36 readback, and contain H120.
Large Formal pools use bounded LRU caches so thousands of wide SWMM CSVs cannot
accumulate in RAM.
"""
from __future__ import annotations
import argparse,hashlib,json,math,subprocess,sys
from collections import OrderedDict
from pathlib import Path
from typing import Any,Mapping
import numpy as np,pandas as pd
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
from sewerrtc.prompt3.gate5r_pipeline import branch_state_hashes
from sewerrtc.simulation.pyswmm_runner import physical_network_sha256
from sewerrtc.v4.formal_f2 import FORMAL_GENERATION_ID,canonical_rain_group,checkpoint_of,read_table,sha256_file,text,yes
from sewerrtc.v4.v42_fast_e2e import make_causal_rainfall_forecast
from sewerrtc.v4.v42_fast_feasibility import _cols,_kpis,_load_graph_topology,_rain,_read_core_detail,_select
from sewerrtc.v4.v42_priority_contract import get_pfv_core_node_indices
from sewerrtc.v4.v42_trajectory_builder import N_FACILITIES,N_HISTORY_FRAMES,N_HORIZON_STEPS
ROLES=('candidate','no_control','dynamic_internal','hold_previous');ROLE_ALIASES={'candidate':('candidate',),'no_control':('no_control',),'dynamic_internal':('dynamic_internal','dynamic_internal_rules'),'hold_previous':('hold_previous',)};BINARY_PUMPS={'add301.2','add301.3'};ATOL=1e-7

def _completion_paths(root:Path)->list[Path]:
 try:
  r=subprocess.run(['rg','--files','-uu','-g','completion.json',str(root)],capture_output=True,text=True,check=False)
  if r.returncode in (0,1):return [Path(x) for x in r.stdout.splitlines() if x.strip()]
 except FileNotFoundError:pass
 return list(root.rglob('completion.json'))
def _completion_index(root:Path)->dict[str,list[Path]]:
 out={}
 for p in _completion_paths(root):
  try:q=json.loads(p.read_text(encoding='utf-8'))
  except Exception:continue
  cid=text(q.get('case_id',''))
  if cid:out.setdefault(cid,[]).append(p)
 return {k:sorted(v,key=lambda p:str(p).casefold()) for k,v in out.items()}
def _prefix(a:Path,b:Path)->int:
 aa=[x.casefold() for x in a.resolve().parts];bb=[x.casefold() for x in b.resolve().parts];n=0
 for x,y in zip(aa,bb):
  if x!=y:break
  n+=1
 return n
def _choose(paths:list[Path],manifest:Path)->Path:
 if not paths:raise FileNotFoundError('no completion candidates')
 return sorted(paths,key=lambda p:(-_prefix(p,manifest),len(p.parts),str(p).casefold()))[0]
def _detail_path(completion:Path,payload:Mapping[str,Any])->Path|None:
 raw=payload.get('detail_path')
 if not raw and isinstance(payload.get('result'),Mapping):raw=payload['result'].get('detail_file')
 if not raw:return None
 p=Path(str(raw))
 for q in (p,completion.parent/p,completion.parent/p.name):
  if q.exists():return q.resolve()
 return None
def _resolve(completion:Path,payload:Mapping[str,Any],role:str)->Path:
 branches=payload.get('branches',{})
 if isinstance(branches,Mapping):
  for alias in ROLE_ALIASES[role]:
   value=branches.get(alias)
   if value is None:continue
   raw=value if isinstance(value,str) else text(value.get('detail_path') or value.get('path') or value.get('detail')) if isinstance(value,Mapping) else ''
   if not raw:continue
   p=Path(raw)
   for q in (p,completion.parent/p,completion.parent/p.name):
    if q.exists():return q.resolve()
 if role=='candidate':
  own=_detail_path(completion,payload)
  if own is not None:return own
 cid=text(payload.get('case_id',completion.parent.name));parts=cid.rsplit('__',1)
 base=parts[0] if len(parts)==2 and parts[1].casefold() in {x.casefold() for xs in ROLE_ALIASES.values() for x in xs} else cid
 for alias in ROLE_ALIASES[role]:
  sibling=completion.parent.parent/f'{base}__{alias}'/'completion.json'
  if not sibling.exists():continue
  try:sibling_payload=json.loads(sibling.read_text(encoding='utf-8'))
  except Exception:continue
  detail=_detail_path(sibling,sibling_payload)
  if detail is not None:return detail
 raise FileNotFoundError(f'{completion}: no strict detail for role={role}')
def _times(cp:float)->tuple[list[float],list[float]]:return ([cp-(N_HISTORY_FRAMES-1-i)*5. for i in range(N_HISTORY_FRAMES)],[cp+(i+1)*10. for i in range(N_HORIZON_STEPS)])
def _state_arrays(detail:pd.DataFrame,cp:float,nodes:list[str],fac:list[str])->tuple[np.ndarray,np.ndarray,np.ndarray]:
 row=_select(detail,[cp]);return _cols(row,'h:',nodes)[0],_cols(row,'flood:',nodes)[0],_cols(row,'setting:',fac)[0]
def _same_state(details:Mapping[str,pd.DataFrame],cp:float,nodes:list[str],fac:list[str])->bool:
 hashes={r:branch_state_hashes(d,checkpoint_min=cp,facility_ids=fac) for r,d in details.items()}
 return all(len({h.get(k,'') for h in hashes.values()})==1 and bool(next(iter({h.get(k,'') for h in hashes.values()}))) for k in ('prefix_history_sha256','checkpoint_pre_action_sha256'))
def _state_sha(detail:pd.DataFrame,cp:float,nodes:list[str],fac:list[str])->str:
 return str(branch_state_hashes(detail,checkpoint_min=cp,facility_ids=fac).get('checkpoint_pre_action_sha256') or '')
def _action_ok(a:np.ndarray,fac:list[str])->bool:
 if a.ndim!=2 or a.shape[1]!=N_FACILITIES or not np.isfinite(a).all() or np.nanmin(a)<-1e-8 or np.nanmax(a)>1+1e-8:return False
 for i,f in enumerate(fac):
  if f.casefold() in BINARY_PUMPS and not bool((np.isclose(a[:,i],0,atol=1e-8)|np.isclose(a[:,i],1,atol=1e-8)).all()):return False
 return True
def _find_bool(payload:Any,names:set[str])->bool|None:
 if isinstance(payload,Mapping):
  for k,v in payload.items():
   if str(k).casefold() in names:return yes(v)
  for v in payload.values():
   x=_find_bool(v,names)
   if x is not None:return x
 elif isinstance(payload,list):
  for v in payload:
   x=_find_bool(v,names)
   if x is not None:return x
 return None
def _engineering(payload:Mapping[str,Any],source:Mapping[str,Any],raw:bool)->bool:
 if not raw and 'actuator_semantics_ok' in source:return yes(source.get('actuator_semantics_ok'))
 for names in ({'bounds','bounds_ok','bounds_pass','binary_semantics_ok'},{'rate','rate_ok','rate_limit','rate_limit_ok','rate_pass'},{'ramp','ramp_ok','ramp_pass','rate_limit_ok'},{'dwell','dwell_ok','dwell_pass'},{'interlock','interlock_ok','interlock_pass'}):
  if _find_bool(payload,names) is not True:return False
 return True
def _no_hotstart(payload:Mapping[str,Any],source:Mapping[str,Any])->bool:
 if 'no_hotstart' in source:return yes(source.get('no_hotstart'))
 for k in ('hotstart_used','use_hotstart','hot_start_used','hotstart'):
  if k in payload:return not yes(payload.get(k))
 return False
def _physical(payload:Mapping[str,Any],source:Mapping[str,Any],frozen:str,raw:bool,*,physical_sha:str='',expected_network:str='',expected_physical:str='',network_path:Path|None=None)->bool:
 if not raw and 'physical_sha_ok' in source:return yes(source.get('physical_sha_ok'))
 if expected_physical and physical_sha!=expected_physical:return False
 physical_vals={text(source.get(k,'')) for k in ('physical_network_sha256','physical_sha256','physical_inp_sha256')}|{text(payload.get(k,'')) for k in ('physical_network_sha256','physical_sha256','physical_inp_sha256')};physical_vals.discard('')
 if physical_vals:return physical_sha in physical_vals
 network_vals={text(source.get(k,'')) for k in ('network_sha256','inp_sha256')}|{text(payload.get(k,'')) for k in ('network_sha256','inp_sha256')};network_vals.discard('')
 if expected_network and network_vals:return expected_network in network_vals
 if network_path is not None:
  for obj in (source,payload):
   raw_kwargs=obj.get('runner_kwargs') or obj.get('source_runner_kwargs')
   try:kw=json.loads(raw_kwargs) if isinstance(raw_kwargs,str) else raw_kwargs
   except Exception:kw=None
   if isinstance(kw,Mapping) and kw.get('inp_path'):
    try:
     if Path(str(kw['inp_path'])).resolve()==network_path.resolve():return True
    except OSError:pass
 return bool(frozen and frozen in (network_vals|physical_vals))
def _gate(source:Mapping[str,Any],name:str)->bool:return name in source and yes(source.get(name))
def _h3sha(a:np.ndarray)->str:return hashlib.sha256(np.ascontiguousarray(a[:3],dtype=np.float64).tobytes(order='C')).hexdigest()

def _cached_detail(cache:OrderedDict[str,pd.DataFrame],path:Path,nodes:list[str],fac:list[str],max_items:int)->pd.DataFrame:
 key=str(path.resolve())
 if key in cache:
  value=cache.pop(key);cache[key]=value;return value
 value=_read_core_detail(path,nodes,fac);cache[key]=value
 while len(cache)>max_items:cache.popitem(last=False)
 return value

def _source(meta:pd.Series,cache:OrderedDict[str,pd.DataFrame],max_items:int=8)->dict[str,Any]:
 path=Path(str(meta['source_manifest'])).resolve();key=str(path)
 if key in cache:value=cache.pop(key);cache[key]=value
 else:
  value=read_table(path);cache[key]=value
  while len(cache)>max_items:cache.popitem(last=False)
 p=int(meta['source_row_number'])
 if p<0 or p>=len(value):raise IndexError('source_row_number outside source manifest')
 return value.iloc[p].to_dict()
def _uid(g:str,s:str,a:str)->str:return hashlib.sha256(f'{g}|{s}|{a}'.encode()).hexdigest()
def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--project-root',type=Path,default=PROJECT_ROOT);ap.add_argument('--metadata-pool',type=Path,default=PROJECT_ROOT/'outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/prepare/FORMAL_F2_STEP2_METADATA_POOL.parquet');ap.add_argument('--output-root',type=Path,default=PROJECT_ROOT/'outputs/project6_dual_reference_v4');ap.add_argument('--output-manifest',type=Path,default=PROJECT_ROOT/'outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/step2/FORMAL_F2_STEP2_RAW_MANIFEST.parquet');ap.add_argument('--min-rainfall-groups',type=int,default=65);ap.add_argument('--detail-cache-items',type=int,default=24);ap.add_argument('--source-cache-items',type=int,default=8);a=ap.parse_args()
 if a.detail_cache_items<4 or a.source_cache_items<1:raise ValueError('cache limits too small')
 meta=read_table(a.metadata_pool)
 if meta.empty:raise ValueError('Formal F2 Step2 metadata pool is empty')
 graph=_load_graph_topology(a.project_root);network_path=a.project_root/'data/wuhan_v8_storage_retrofit.inp';frozen=sha256_file(network_path);physical_sha=physical_network_sha256(network_path);contract_path=a.project_root/'docs/contracts/PROJECT6_V4_FINAL_PIPELINE_CONTRACT.json';contract=json.loads(contract_path.read_text(encoding='utf-8')) if contract_path.exists() else {};expected_network=text(contract.get('network_sha256',''));expected_physical=text(contract.get('physical_network_sha256',''));nodes=list(map(str,graph['node_ids']));fac=list(map(str,graph['facility_ids']));priority=get_pfv_core_node_indices(nodes);index=_completion_index(a.output_root);detail_cache:OrderedDict[str,pd.DataFrame]=OrderedDict();source_cache:OrderedDict[str,pd.DataFrame]=OrderedDict();records=[];failures=[];peak_detail_cache=0
 for row_i,(_,m) in enumerate(meta.iterrows(),start=1):
  cid=text(m.get('case_id',''));sid=text(m.get('source_id',''))
  try:
   src=_source(m,source_cache,a.source_cache_items);group=canonical_rain_group(src) or text(m.get('rainfall_group_key',''));cp=checkpoint_of(src);cp=float(m['checkpoint_min']) if not np.isfinite(cp) else cp
   if not group or not np.isfinite(cp):raise ValueError('missing rainfall group or checkpoint')
   if not cid:cid=text(src.get('case_id',src.get('candidate_id','')))
   if not cid or cid not in index:raise FileNotFoundError(f'completion not found for case_id={cid!r}')
   source_manifest=Path(str(m['source_manifest']));completion=_choose(index[cid],source_manifest);payload=json.loads(completion.read_text(encoding='utf-8'));paths={r:_resolve(completion,payload,r) for r in ROLES};details={r:_cached_detail(detail_cache,p,nodes,fac,a.detail_cache_items) for r,p in paths.items()};peak_detail_cache=max(peak_detail_cache,len(detail_cache))
   ht,ft=_times(float(cp));history=_select(details['candidate'],ht);history_depth=_cols(history,'h:',nodes);history_actions=_cols(history,'setting:',fac);history_rain=_rain(history);branches={}
   for role in ROLES:
    future=_select(details[role],ft);branches[role]={'depth':_cols(future,'h:',nodes),'flood':_cols(future,'flood:',nodes),'action':_cols(future,'setting:',fac),'rainfall':_rain(future)}
   future_rain=branches['candidate']['rainfall'];forcing=all(np.allclose(future_rain,branches[r]['rainfall'],atol=ATOL,rtol=0.) for r in ('no_control','dynamic_internal','hold_previous'));hashes={r:branch_state_hashes(d,checkpoint_min=float(cp),facility_ids=fac) for r,d in details.items()};same=_same_state(details,float(cp),nodes,fac);prefix_ok=bool(len({h.get('prefix_history_sha256','') for h in hashes.values()})==1 and next(iter({h.get('prefix_history_sha256','') for h in hashes.values()}),''));state_sha=_state_sha(details['candidate'],float(cp),nodes,fac);ca=branches['candidate']['action'];ha=branches['hold_previous']['action'];changed=int(np.max(np.sum(np.abs(ca[:3]-ha[:3])>1e-8,axis=1)));semantics=all(_action_ok(branches[r]['action'],fac) for r in ROLES);authorized=bool(m.get('step2_accepted_from_manifest',False));raw=bool(m.get('raw_readmission_pending',False));engineering=_engineering(payload,src,raw);readback=all(branches[r]['action'].shape==(N_HORIZON_STEPS,N_FACILITIES) and np.isfinite(branches[r]['action']).all() for r in ROLES);h120=bool(len(history)==N_HISTORY_FRAMES and all(len(branches[r]['depth'])==N_HORIZON_STEPS for r in ROLES));pfv,tfv,peak=_kpis(branches,priority);kpi=all(math.isfinite(float(x)) for x in (pfv,tfv,peak))
   gates={'h120_eligible':h120,'label_validity_pfv':math.isfinite(float(pfv)),'label_validity_tfv':math.isfinite(float(tfv)),'label_validity_peak':math.isfinite(float(peak)),'same_state_ok':same,'physical_sha_ok':_physical(payload,src,frozen,raw,physical_sha=physical_sha,expected_network=expected_network,expected_physical=expected_physical,network_path=network_path),'rainfall_sha_ok':bool(forcing and group),'prefix_sha_ok':bool(prefix_ok and state_sha),'readback_ok':readback,'no_hotstart':_no_hotstart(payload,src) if raw else _gate(src,'no_hotstart'),'k_le_8':changed<=8,'actuator_semantics_ok':bool(semantics and engineering),'h120_window_complete':h120,'kpi_recompute_ok':kpi}
   if authorized:
    for k in ('physical_sha_ok','no_hotstart'):gates[k]=gates[k] or _gate(src,k)
   if not all(gates.values()):raise RuntimeError(f'raw formal admission failed: {[k for k,v in gates.items() if not v]}')
   action_sha=_h3sha(ca);forecast=make_causal_rainfall_forecast(history_rain);rec={'formal_generation_id':FORMAL_GENERATION_ID,'development_only':False,'formal_mainline_authorized':False,'source_dataset':sid,'source_manifest':str(source_manifest),'source_manifest_sha256':text(m.get('source_manifest_sha256','')),'source_row_number':int(m['source_row_number']),'case_id':cid,'case_uid':_uid(group,state_sha,action_sha),'event_id':text(src.get('event_id',m.get('event_id',''))),'rainfall_sha256':group,'split_group_key':group,'checkpoint_min':float(cp),'state_key':state_sha,'candidate_action_sha256':action_sha,'actual_k':changed,'history_depth_swmm_truth_diagnostic':json.dumps(history_depth.tolist(),allow_nan=False),'history_depth':json.dumps(history_depth.tolist(),allow_nan=False),'history_actions_readback':json.dumps(history_actions.tolist(),allow_nan=False),'rainfall_realized_future_diagnostic':json.dumps(future_rain.tolist(),allow_nan=False),'rainfall_forecast':json.dumps(forecast.tolist(),allow_nan=False),'rainfall_input_authority':'causal_persistence_decay_from_observed_history','pfv_delta':float(pfv),'tfv_delta':float(tfv),'peak_delta':float(peak),'training_admission_authorized':True,'raw_independent_oracle_all_pass':True,'same_state_raw_verified':True,'same_forcing_raw_verified':True,'actual_readback_verified':True,'future_SWMM_trajectories_supervision_only':True};rec.update(gates)
   for role in ROLES:rec[f'action_{role}_readback']=json.dumps(branches[role]['action'].tolist(),allow_nan=False);rec[f'trajectory_depth_{role}']=json.dumps(branches[role]['depth'].tolist(),allow_nan=False);rec[f'trajectory_flood_{role}']=json.dumps(branches[role]['flood'].tolist(),allow_nan=False);rec[f'source_detail_path_{role}']=str(paths[role])
   records.append(rec)
  except Exception as exc:failures.append({'source_dataset':sid,'case_id':cid,'source_manifest':text(m.get('source_manifest','')),'source_row_number':int(m.get('source_row_number',-1)),'error':f'{type(exc).__name__}: {exc}'})
  if row_i%100==0:print(json.dumps({'stage':'formal_f2_step2_raw_readmission','processed':row_i,'total':len(meta),'accepted':len(records),'failed':len(failures),'detail_cache_items':len(detail_cache)},allow_nan=False),flush=True)
 out=pd.DataFrame(records);a.output_manifest.parent.mkdir(parents=True,exist_ok=True)
 if not out.empty:out=out.sort_values(['rainfall_sha256','state_key','candidate_action_sha256'],kind='mergesort').drop_duplicates(['rainfall_sha256','state_key','candidate_action_sha256']);out.to_parquet(a.output_manifest,index=False)
 groups=int(out.rainfall_sha256.astype(str).nunique()) if not out.empty else 0;source_summary={str(s):{'rows':len(g),'rainfall_groups':int(g.rainfall_sha256.astype(str).nunique()),'states':int(g.state_key.astype(str).nunique())} for s,g in out.groupby('source_dataset')} if not out.empty else {};audit={'formal_generation_id':FORMAL_GENERATION_ID,'stage':'formal_f2_step2_raw_readmission','development_only':False,'formal_mainline_authorized':False,'input_metadata_rows':len(meta),'accepted_rows':len(out),'accepted_rainfall_groups':groups,'minimum_rainfall_groups':a.min_rainfall_groups,'accepted_states':int(out.state_key.astype(str).nunique()) if not out.empty else 0,'actual_k_distribution':out.actual_k.value_counts().sort_index().to_dict() if not out.empty else {},'source_summary':source_summary,'failed_rows':len(failures),'failure_examples':failures[:200],'frozen_inp_sha256':frozen,'physical_network_sha256':physical_sha,'formal_network_sha256':expected_network,'formal_physical_network_sha256':expected_physical,'four_reference_shared_model_ready':not out.empty,'trajectory_first_kpi_derivation':True,'action_authority':'actual_readback_setting','raw_independent_oracle_all_pass':not out.empty,'detail_cache_limit':a.detail_cache_items,'peak_detail_cache_items':peak_detail_cache,'source_manifest_cache_limit':a.source_cache_items,'status':'pass' if groups>=a.min_rainfall_groups else 'fail'}
 if audit['status']=='fail':audit['reason']='accepted formal Step2 rainfall groups below minimum'
 (a.output_manifest.parent/'FORMAL_F2_STEP2_RAW_ADMISSION_AUDIT.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');print(json.dumps(audit,indent=2,ensure_ascii=False,allow_nan=False),flush=True);return 0 if audit['status']=='pass' else 3
if __name__=='__main__':raise SystemExit(main())
