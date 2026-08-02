"""Visible orchestration for Project6 V4.2 Formal F2.

Runs restartable metadata/readmission, three-seed Step1, causal GAT materialising,
three-seed Step2, and structural audit. It never fabricates downstream paper
evidence: OOD/safety calibration and authoritative closed-loop/lock/blind remain
explicit required stages.
"""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
from typing import Any
PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
from sewerrtc.v4.formal_f2 import FORMAL_GENERATION_ID
def _run(cmd:list[str],root:Path)->None:print('\nRUN:',' '.join(cmd),flush=True);subprocess.run(cmd,cwd=str(root),check=True)
def _json(p:Path)->dict[str,Any]:return json.loads(p.read_text(encoding='utf-8'))
def _status(root:Path)->dict[str,Any]:
 formal=root/'outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2';prep=formal/'prepare/FORMAL_F2_PREPARE_AUDIT.json';raw=formal/'step2/FORMAL_F2_STEP2_RAW_ADMISSION_AUDIT.json';gat=formal/'step2/FORMAL_F2_STEP2_GAT_HISTORY_AUDIT.json';s1=sorted((formal/'step1').glob('seed_*/formal_step1_report.json'));s2=sorted((formal/'step2/models').glob('seed_*/formal_step2_report.json'));p={'formal_generation_id':FORMAL_GENERATION_ID,'prepare':_json(prep) if prep.exists() else None,'raw_step2':_json(raw) if raw.exists() else None,'gat_step2':_json(gat) if gat.exists() else None,'step1_seed_reports':[str(x) for x in s1],'step2_seed_reports':[str(x) for x in s2],'formal_mainline_authorized':False};r=[]
 if not p['prepare'] or p['prepare'].get('status')!='pass':r.append('formal_prepare_not_pass')
 if not p['raw_step2'] or p['raw_step2'].get('status')!='pass':r.append('formal_step2_raw_admission_not_pass')
 if len(s1)<3:r.append('formal_step1_requires_three_model_seeds')
 if not p['gat_step2'] or p['gat_step2'].get('status')!='pass':r.append('formal_step2_gat_history_not_pass')
 if len(s2)<3:r.append('formal_step2_requires_three_model_seeds')
 a=[_json(x) for x in s1]
 if a:
  if len({tuple(x.get('train_rainfall_groups',[])) for x in a})!=1:r.append('step1_split_changes_with_model_seed')
  if any(int(x.get('train_rainfall_group_count',0))<65 for x in a):r.append('step1_train_rainfall_groups_below_65')
  if any(x.get('uses_future_hydraulic_truth') is not False for x in a):r.append('step1_future_truth_contract_violation')
 b=[_json(x) for x in s2]
 if b:
  if len({tuple(x.get('train_rainfall_groups',[])) for x in b})!=1:r.append('step2_split_changes_with_model_seed')
  if any(int(x.get('train_rainfall_group_count',0))<65 for x in b):r.append('step2_train_rainfall_groups_below_65')
  if any(x.get('raw_independent_oracle_all_pass') is not True for x in b):r.append('step2_raw_oracle_not_all_pass')
 p['structural_training_chain_pass']=not r;p['reasons']=r;p['next_required_stages']=['Step1 independent OOD calibration and evidence.json','Step2 ensemble/conformal PFV+Peak safety calibration','true_state_offline_validation','authoritative exact SWMM closed loop','surrogate closed loop','GAT-integrated closed loop','policy lock','challenge','one-shot locked validation','formal blind >=24 new rainfall SHA with all authoritative baselines'];formal.mkdir(parents=True,exist_ok=True);(formal/'FORMAL_F2_STATUS.json').write_text(json.dumps(p,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');return p
def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--project-root',type=Path,default=PROJECT_ROOT);ap.add_argument('--stage',choices=('prepare','step1','step2','audit','all'),default='prepare');ap.add_argument('--seeds',type=int,nargs='+',default=[17,42,73]);ap.add_argument('--primary-step1-seed',type=int,default=42);ap.add_argument('--split-seed',type=int,default=42);ap.add_argument('--sensor-layout-seed',type=int,default=42);a=ap.parse_args();root=a.project_root;py=str(Path(sys.executable));formal=root/'outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2';prep=formal/'prepare';s1=formal/'step1';s2=formal/'step2'
 def prepare():
  _run([py,'-u',str(root/'scripts/prepare_v42_formal_f2.py'),'--project-root',str(root),'--seed',str(a.split_seed)],root);_run([py,'-u',str(root/'scripts/materialize_v42_formal_step2_f2.py'),'--project-root',str(root),'--metadata-pool',str(prep/'FORMAL_F2_STEP2_METADATA_POOL.parquet'),'--output-manifest',str(s2/'FORMAL_F2_STEP2_RAW_MANIFEST.parquet'),'--min-rainfall-groups','65'],root)
 def step1():
  for seed in a.seeds:_run([py,'-u',str(root/'scripts/train_v42_step1_formal_f2.py'),'--project-root',str(root),'--manifest',str(prep/'FORMAL_F2_STEP1_WINDOW_MANIFEST.parquet'),'--output-dir',str(s1/f'seed_{seed}'),'--model-seed',str(seed),'--split-seed',str(a.split_seed),'--sensor-layout-seed',str(a.sensor_layout_seed),'--min-train-groups','65'],root)
 def step2():
  primary=s1/f'seed_{a.primary_step1_seed}'
  if not (primary/'best_model.pt').exists():raise FileNotFoundError(primary/'best_model.pt')
  _run([py,'-u',str(root/'scripts/materialize_v42_formal_gat_history_f2.py'),'--project-root',str(root),'--input-manifest',str(s2/'FORMAL_F2_STEP2_RAW_MANIFEST.parquet'),'--step1-window-manifest',str(prep/'FORMAL_F2_STEP1_WINDOW_MANIFEST.parquet'),'--step1-model-dir',str(primary),'--output-manifest',str(s2/'FORMAL_F2_STEP2_GAT_MANIFEST.parquet'),'--min-rainfall-groups','69','--sensor-layout-seed',str(a.sensor_layout_seed)],root)
  for seed in a.seeds:_run([py,'-u',str(root/'scripts/train_v42_step2_formal_f2.py'),'--project-root',str(root),'--manifest',str(s2/'FORMAL_F2_STEP2_GAT_MANIFEST.parquet'),'--output-dir',str(s2/'models'/f'seed_{seed}'),'--seed',str(seed),'--split-seed',str(a.split_seed),'--min-train-groups','65'],root)
 if a.stage=='prepare':prepare()
 elif a.stage=='step1':step1()
 elif a.stage=='step2':step2()
 elif a.stage=='all':prepare();step1();step2()
 p=_status(root);print(json.dumps(p,indent=2,ensure_ascii=False,allow_nan=False),flush=True);return 0 if (a.stage=='audit' and p['structural_training_chain_pass']) or a.stage!='audit' else 3
if __name__=='__main__':raise SystemExit(main())
