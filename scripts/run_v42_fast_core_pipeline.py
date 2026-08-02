"""Orchestrate the bounded existing-data fast scientific potential line.

Smoke and Potential must never write the same artifacts concurrently. This
runner owns an exclusive lock and archives the previous run before starting a
new one. It also preselects states with complete t-120..t GAT history coverage.
"""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from contextlib import contextmanager
from datetime import datetime,timezone
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[1]
def run(cmd:list[str])->None:print('\nRUN:',' '.join(cmd),flush=True);subprocess.run(cmd,cwd=str(PROJECT_ROOT),check=True)
@contextmanager
def _lock(path:Path,payload:dict):
 path.parent.mkdir(parents=True,exist_ok=True)
 try:fd=os.open(str(path),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
 except FileExistsError:raise RuntimeError(f'another Fast Core run owns shared artifacts: {path} {path.read_text(encoding="utf-8",errors="replace") if path.exists() else ""}')
 try:
  with os.fdopen(fd,'w',encoding='utf-8') as f:json.dump(payload,f,indent=2);f.flush();os.fsync(f.fileno())
  yield
 finally:
  try:path.unlink(missing_ok=True)
  except Exception:pass
def _archive(fast:Path,root:Path)->str|None:
 if not fast.exists() or not any(fast.iterdir()):fast.mkdir(parents=True,exist_ok=True);return None
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');mode='unknown';meta=fast/'FAST_RUN_METADATA.json'
 if meta.exists():
  try:mode=str(json.loads(meta.read_text(encoding='utf-8')).get('mode','unknown'))
  except Exception:pass
 root.mkdir(parents=True,exist_ok=True);dest=root/f'{stamp}_{mode}';i=1
 while dest.exists():dest=root/f'{stamp}_{mode}_{i}';i+=1
 fast.rename(dest);fast.mkdir(parents=True,exist_ok=True);return str(dest)
def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=('smoke','potential'),default='potential');ap.add_argument('--project-root',type=Path,default=PROJECT_ROOT);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--reuse-step1-model-dir',type=Path,default=None);a=ap.parse_args()
 root=a.project_root;py=str(Path(sys.executable));v4=root/'outputs/project6_dual_reference_v4';final=v4/'final_v4';paper=final/'v42_paper';fast=paper/'fast_e2e_64plus';run_id=f'{a.mode}_seed{a.seed}_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}'
 payload={'pid':os.getpid(),'mode':a.mode,'seed':a.seed,'run_id':run_id,'start_utc':datetime.now(timezone.utc).isoformat(),'shared_run_dir':str(fast)}
 with _lock(paper/'.fast_core_pipeline.lock',payload):
  archived=_archive(fast,paper/'fast_runs');core=fast/'core_pool';raw=fast/'step2_fast_e2e_core_manifest.parquet';meta={**payload,'archived_previous_run':archived,'status':'running'};(fast/'FAST_RUN_METADATA.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
  if a.mode=='smoke':target,matmin,runmin,s1e,s2e,mintrain=8,8,8,1,1,0
  else:target,matmin,runmin,s1e,s2e,mintrain=88,81,64,3,4,65
  try:
   run([py,'-u',str(root/'scripts/build_v42_fast_core_pool.py'),'--project-root',str(root),'--v4-root',str(v4),'--output-dir',str(core),'--min-checkpoint-min','120','--candidates-per-state','3','--seed',str(a.seed)])
   run([py,'-u',str(root/'scripts/select_v42_fast_gat_compatible_core.py'),'--core-pool-dir',str(core),'--step1-window-manifest',str(final/'v42_paper/step1_gat/dataset/step1_window_manifest.parquet'),'--min-checkpoint-min','120','--candidates-per-state','3','--seed',str(a.seed)])
   run([py,'-u',str(root/'scripts/materialize_v42_fast_core_train1600.py'),'--project-root',str(root),'--output-root',str(v4),'--core-pool-dir',str(core),'--output-manifest',str(raw),'--target-groups',str(target),'--min-groups',str(matmin),'--candidates-per-state','3','--seed',str(a.seed)])
   cmd=[py,'-u',str(root/'scripts/run_v42_fast_e2e_64plus.py'),'--project-root',str(root),'--target-rainfall-groups',str(target),'--min-rainfall-groups',str(runmin),'--min-step2-train-groups',str(mintrain),'--candidates-per-state','3','--min-checkpoint-min','120','--step1-epochs',str(s1e),'--step2-epochs',str(s2e),'--seed',str(a.seed),'--prebuilt-step2-manifest',str(raw)]
   if a.mode=='smoke':cmd.append('--smoke')
   if a.reuse_step1_model_dir is not None:cmd.extend(['--reuse-step1-model-dir',str(a.reuse_step1_model_dir)])
   run(cmd);meta['status']='complete'
  except Exception as exc:meta['status']='failed';meta['error']=f'{type(exc).__name__}: {exc}';raise
  finally:meta['end_utc']=datetime.now(timezone.utc).isoformat();(fast/'FAST_RUN_METADATA.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
 return 0
if __name__=='__main__':raise SystemExit(main())
