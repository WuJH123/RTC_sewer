"""Background monitor: wait for Aug1 generate to finish, then run Build/Train/Gate.

Polls every 60s. When PID 23896 exits AND manifest appears, proceeds to:
  1. build_v4_augmented_dataset
  2. train_v4_aug1
  3. evaluate_v4_aug1_model_gate
"""
import os, sys, time, json, traceback
sys.path.insert(0, r'E:\RTC_sewer\Project6')
os.chdir(r'E:\RTC_sewer\Project6')

import psutil
from sewerrtc.prompt3 import action_effect_v4_aug1 as v4a

CONFIG = r'E:\RTC_sewer\Project6\configs\wuhan_project6_dual_reference_v4.yaml'
BASE = r'E:\RTC_sewer\Project6\outputs\project6_dual_reference_v4\dual_reference_aug1'
LOG = os.path.join(BASE, 'monitor.log')
TARGET_PID = 23896
POLL_SEC = 60

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        pass


def pid_alive(pid):
    try:
        p = psutil.Process(pid)
        return p.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)
    except psutil.NoSuchProcess:
        return False


def wait_for_generation():
    log(f"waiting for PID {TARGET_PID} to exit")
    while pid_alive(TARGET_PID):
        time.sleep(POLL_SEC)
    log(f"PID {TARGET_PID} exited. waiting for manifest")
    manifest = os.path.join(BASE, 'v4_aug1_generation_manifest.csv')
    # give up to 10 min for manifest to appear
    deadline = time.time() + 600
    while not os.path.exists(manifest) and time.time() < deadline:
        time.sleep(15)
    if not os.path.exists(manifest):
        log("ERROR: manifest did not appear within 10 min")
        return False
    log(f"manifest appeared: {manifest}")
    return True


def run_stage(name, fn, *args, **kwargs):
    log(f"=== {name} START ===")
    t0 = time.time()
    try:
        code, outputs = fn(*args, **kwargs)
    except Exception:
        log(f"=== {name} EXCEPTION ===\n{traceback.format_exc()}")
        return None
    dt = time.time() - t0
    log(f"=== {name} END code={code} dt={dt:.1f}s outputs={list(outputs.keys()) if isinstance(outputs,dict) else outputs}")
    return code


def main():
    log("monitor started")
    if not wait_for_generation():
        sys.exit(2)
    # Stage 1: Build
    c1 = run_stage("Build", v4a.build_v4_augmented_dataset, CONFIG, smoke=False)
    if c1 != 0:
        log(f"Build failed with code={c1}; aborting downstream stages")
        sys.exit(c1 or 3)
    # Stage 2: Train
    c2 = run_stage("Train", v4a.train_v4_aug1, CONFIG, smoke=False)
    if c2 != 0:
        log(f"Train failed with code={c2}; aborting downstream stages")
        sys.exit(c2 or 3)
    # Stage 3: Gate
    c3 = run_stage("Gate", v4a.evaluate_v4_aug1_model_gate, CONFIG, smoke=False)
    log(f"Gate returned code={c3}")
    log("monitor pipeline complete")


if __name__ == '__main__':
    main()
