"""Gate 0: Preflight read-only audit for Aug1 manifest recovery V2.

Checks 12 items from the specification:
  1. Aug1 generation process running
  2. .writer.lock existence & content
  3. Lock PID alive?
  4. Cases dir recent mtime changes
  5. Heartbeat file
  6. Current manifest existence
  7. Manifest & audit SHA256
  8. Cases sub-directories
  9. CSV flat vs nested
 10. Temp / zero-byte / partial files
 11. Duplicate filenames
 12. Unparseable filenames
"""
import os, sys, json, hashlib, time, re, subprocess
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone

PROJECT_ROOT = Path(r"E:\RTC_sewer\Project6")
AUG1_DIR = PROJECT_ROOT / "outputs" / "project6_dual_reference_v4" / "dual_reference_aug1"
CASES_DIR = AUG1_DIR / "cases"
LOCK_FILE = AUG1_DIR / ".writer.lock"
MANIFEST = AUG1_DIR / "v4_aug1_generation_manifest.csv"
AUDIT_JSON = AUG1_DIR / "v4_aug1_generation_audit.json"
OUT_DIR = AUG1_DIR / "recovery_v2"

BRANCH_TOKENS = {"__c_": "candidate", "__no_con": "no_control",
                 "__hold_p": "hold_previous", "__intern": "internal_current_action",
                 "__passiv": "passive_anchor"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "ERROR"


def check_pid_alive(pid: int) -> bool:
    try:
        r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                           capture_output=True, text=True, timeout=10)
        return str(pid) in r.stdout
    except Exception:
        return False


def find_python_pids() -> list[dict]:
    """Find python processes that might be Aug1 generators."""
    procs = []
    try:
        r = subprocess.run(
            ["wmic", "process", "where", "name like '%python%'",
             "get", "ProcessId,CommandLine", "/format:csv"],
            capture_output=True, text=True, timeout=15)
        for line in r.stdout.strip().split("\n")[2:]:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                pid_str = parts[-2].strip()
                cmd = parts[-1].strip() if len(parts) > 2 else ""
                if pid_str.isdigit():
                    procs.append({"pid": int(pid_str), "cmd": cmd[:200]})
    except Exception:
        pass
    return procs


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    audit = {"gate": 0, "started_at": datetime.now(timezone.utc).isoformat()}
    issues = []

    # ── 1. Aug1 generation process running ─────────────────────────────
    python_procs = find_python_pids()
    aug1_like = [p for p in python_procs
                 if "aug1" in p["cmd"].lower() or "action_effect" in p["cmd"].lower()
                 or "recover_manifest" in p["cmd"].lower()]
    audit["active_aug1_processes"] = aug1_like
    if aug1_like:
        issues.append("ACTIVE_AUG1_PROCESS")

    # ── 2. .writer.lock ────────────────────────────────────────────────
    lock_info = {"exists": LOCK_FILE.exists()}
    if LOCK_FILE.exists():
        lock_stat = LOCK_FILE.stat()
        lock_info["size_bytes"] = lock_stat.st_size
        lock_info["mtime"] = datetime.fromtimestamp(lock_stat.st_mtime).isoformat()
        lock_info["age_minutes"] = round((time.time() - lock_stat.st_mtime) / 60, 1)
        try:
            lock_content = LOCK_FILE.read_text(encoding="utf-8")
            lock_info["content"] = lock_content[:500]
            # Try to extract PID
            m = re.search(r'"pid"\s*:\s*(\d+)', lock_content)
            if m:
                lock_pid = int(m.group(1))
                lock_info["pid"] = lock_pid
            else:
                m2 = re.search(r'PID\s*[:=]\s*(\d+)', lock_content, re.I)
                if m2:
                    lock_pid = int(m2.group(1))
                    lock_info["pid"] = lock_pid
        except Exception as e:
            lock_info["read_error"] = str(e)
    audit["lock"] = lock_info

    # ── 3. Lock PID alive? ─────────────────────────────────────────────
    if "pid" in lock_info:
        alive = check_pid_alive(lock_info["pid"])
        lock_info["pid_alive"] = alive
        if alive:
            issues.append("LOCK_PID_STILL_ALIVE")

    # ── 4. Cases dir recent mtime ──────────────────────────────────────
    if CASES_DIR.exists():
        csv_files = sorted(CASES_DIR.glob("*.csv"))
        mtimes = [f.stat().st_mtime for f in csv_files]
        newest = max(mtimes) if mtimes else 0
        oldest = min(mtimes) if mtimes else 0
        now = time.time()
        audit["cases_dir"] = {
            "exists": True,
            "total_csv": len(csv_files),
            "newest_mtime": datetime.fromtimestamp(newest).isoformat() if newest else None,
            "newest_age_minutes": round((now - newest) / 60, 1) if newest else None,
            "oldest_mtime": datetime.fromtimestamp(oldest).isoformat() if oldest else None,
            "all_same_age": (newest - oldest) < 60 if mtimes else None,
        }
        # Check if files are still being modified (within last 2 min)
        recently_modified = sum(1 for m in mtimes if (now - m) < 120)
        audit["cases_dir"]["recently_modified_count"] = recently_modified
        if recently_modified > 0:
            issues.append("CASES_DIR_RECENTLY_MODIFIED")
    else:
        audit["cases_dir"] = {"exists": False}
        issues.append("CASES_DIR_MISSING")

    # ── 5. Heartbeat ───────────────────────────────────────────────────
    hb_files = list(AUG1_DIR.glob("*heartbeat*")) + list(AUG1_DIR.glob("*progress*"))
    audit["heartbeat_files"] = []
    for hf in hb_files:
        st = hf.stat()
        audit["heartbeat_files"].append({
            "path": str(hf), "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(),
            "age_minutes": round((time.time() - st.st_mtime) / 60, 1),
        })

    # ── 6. Current manifest ────────────────────────────────────────────
    audit["manifest"] = {"exists": MANIFEST.exists()}
    if MANIFEST.exists():
        st = MANIFEST.stat()
        audit["manifest"]["size_bytes"] = st.st_size
        audit["manifest"]["mtime"] = datetime.fromtimestamp(st.st_mtime).isoformat()
        # Count rows
        try:
            import csv
            with open(MANIFEST, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.reader(f))
            audit["manifest"]["row_count"] = len(rows) - 1  # minus header
        except Exception as e:
            audit["manifest"]["read_error"] = str(e)

    # ── 7. SHA256 of manifest & audit ──────────────────────────────────
    audit["sha256"] = {}
    for name, path in [("manifest", MANIFEST), ("audit_json", AUDIT_JSON)]:
        if path.exists():
            audit["sha256"][name] = sha256_file(path)
        else:
            audit["sha256"][name] = "FILE_NOT_FOUND"

    # ── 8-10. File inventory ───────────────────────────────────────────
    all_csvs = list(CASES_DIR.rglob("*.csv")) if CASES_DIR.exists() else []
    subdirs = [d for d in CASES_DIR.iterdir() if d.is_dir()] if CASES_DIR.exists() else []
    temp_files = list(CASES_DIR.rglob("*.tmp")) + list(CASES_DIR.rglob("*.partial"))
    zero_byte = [f for f in all_csvs if f.stat().st_size == 0]

    # Parse all CSVs into stems
    stem_branches = defaultdict(lambda: defaultdict(list))
    unparseable = []
    for f in all_csvs:
        base = f.stem
        matched = False
        for tok, branch in BRANCH_TOKENS.items():
            if tok in base:
                stem = base.split(tok)[0]
                stem_branches[stem][branch].append(f)
                matched = True
                break
        if not matched:
            unparseable.append(str(f.name))

    # Duplicate detection
    dup_identical = 0
    dup_conflict = 0
    for stem, branches in stem_branches.items():
        for branch, files in branches.items():
            if len(files) > 1:
                shas = [sha256_file(f) for f in files]
                if len(set(shas)) == 1:
                    dup_identical += len(files) - 1
                else:
                    dup_conflict += 1

    audit["file_inventory"] = {
        "total_csv": len(all_csvs),
        "subdirectories": [str(d.name) for d in subdirs],
        "subdirectory_count": len(subdirs),
        "temp_files": [str(f) for f in temp_files],
        "zero_byte_files": [str(f.name) for f in zero_byte],
        "unparseable_files": unparseable[:50],
        "unparseable_count": len(unparseable),
        "unique_stems": len(stem_branches),
        "duplicate_identical": dup_identical,
        "duplicate_conflict": dup_conflict,
    }

    # Branch completeness per stem
    complete_groups = 0
    partial_groups = 0
    for stem, branches in stem_branches.items():
        has_all_ref = all(
            any(b in branches for b in ["no_control", "passive_anchor",
                                         "internal_current_action", "hold_previous"])
            for _ in [1]
        )
        ref_present = {b for b in ["no_control", "passive_anchor",
                                    "internal_current_action", "hold_previous"]
                       if b in branches}
        if len(ref_present) == 4 and "candidate" in branches:
            complete_groups += 1
        else:
            partial_groups += 1

    audit["file_inventory"]["complete_groups"] = complete_groups
    audit["file_inventory"]["partial_groups"] = partial_groups

    # Candidate file count
    total_cand = sum(len(files) for stem, branches in stem_branches.items()
                     for branch, files in branches.items() if branch == "candidate")
    audit["file_inventory"]["total_candidate_files"] = total_cand

    # ── Verdict ────────────────────────────────────────────────────────
    active_writer = bool(aug1_like) or audit.get("cases_dir", {}).get("recently_modified_count", 0) > 0
    if active_writer:
        audit["verdict"] = "BLOCKED_ACTIVE_WRITER"
    elif dup_conflict > 0:
        audit["verdict"] = "BLOCKED_DUPLICATE_CONFLICT"
    else:
        audit["verdict"] = "READY_FOR_RECOVERY"

    audit["issues"] = issues
    audit["elapsed_sec"] = round(time.time() - t0, 1)
    audit["completed_at"] = datetime.now(timezone.utc).isoformat()

    # ── Write outputs ──────────────────────────────────────────────────
    preflight_path = OUT_DIR / "preflight_process_audit.json"
    with open(preflight_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False, default=str)

    # File inventory CSV
    inv_path = OUT_DIR / "preflight_file_inventory.csv"
    inv_rows = []
    for stem, branches in sorted(stem_branches.items()):
        for branch, files in sorted(branches.items()):
            for fp in files:
                inv_rows.append({
                    "stem": stem, "branch": branch, "filename": fp.name,
                    "size_bytes": fp.stat().st_size,
                    "sha256": sha256_file(fp),
                })
    import csv
    with open(inv_path, "w", encoding="utf-8", newline="") as f:
        if inv_rows:
            w = csv.DictWriter(f, fieldnames=list(inv_rows[0].keys()))
            w.writeheader()
            w.writerows(inv_rows)

    # Lock audit
    lock_audit_path = OUT_DIR / "preflight_lock_audit.json"
    with open(lock_audit_path, "w", encoding="utf-8") as f:
        json.dump(lock_info, f, indent=2, ensure_ascii=False, default=str)

    # Existing manifest backup hash
    backup_path = OUT_DIR / "existing_manifest_backup_manifest.json"
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump({
            "original_manifest": str(MANIFEST),
            "exists": MANIFEST.exists(),
            "sha256": audit["sha256"].get("manifest", "N/A"),
            "row_count": audit.get("manifest", {}).get("row_count", 0),
        }, f, indent=2)

    # ── Print summary ──────────────────────────────────────────────────
    print("=" * 60)
    print("GATE 0: PREFLIGHT AUDIT")
    print("=" * 60)
    print(f"Verdict: {audit['verdict']}")
    print(f"Issues: {issues}")
    print(f"Active Aug1 processes: {len(aug1_like)}")
    print(f"Lock exists: {lock_info['exists']}" +
          (f" (PID {lock_info.get('pid','?')}, alive={lock_info.get('pid_alive','?')})" if lock_info['exists'] else ""))
    print(f"Cases CSV: {len(all_csvs)} files, {len(stem_branches)} stems")
    print(f"  Complete groups: {complete_groups}")
    print(f"  Partial groups:  {partial_groups}")
    print(f"  Candidate files: {total_cand}")
    print(f"  Subdirectories:  {len(subdirs)}")
    print(f"  Temp files:      {len(temp_files)}")
    print(f"  Zero-byte files: {len(zero_byte)}")
    print(f"  Unparseable:     {len(unparseable)}")
    print(f"  Dup identical:   {dup_identical}")
    print(f"  Dup conflict:    {dup_conflict}")
    print(f"Manifest exists: {MANIFEST.exists()}" +
          (f" ({audit.get('manifest',{}).get('row_count',0)} rows)" if MANIFEST.exists() else ""))
    print(f"Elapsed: {audit['elapsed_sec']}s")
    print(f"\nOutputs in: {OUT_DIR}")
    for p in [preflight_path, inv_path, lock_audit_path, backup_path]:
        print(f"  {p}")

    return 5 if active_writer else 0


if __name__ == "__main__":
    sys.exit(main())
