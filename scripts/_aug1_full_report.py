"""Corrected Aug1 case report - fast version."""
import os, csv, re, time, subprocess
from collections import defaultdict, Counter

CASES_DIR = r'E:\RTC_sewer\Project6\outputs\project6_dual_reference_v4\dual_reference_aug1\cases'
PLAN_CSV  = r'E:\RTC_sewer\Project6\outputs\project6_dual_reference_v4\dual_reference_aug1\v4_aug1_case_plan.csv'
AUG1_DIR  = r'E:\RTC_sewer\Project6\outputs\project6_dual_reference_v4\dual_reference_aug1'

BRANCH_TOKENS = {
    '__c_':       'candidate',
    '__no_con':   'no_control',
    '__hold_p':   'hold_previous',
    '__intern':   'internal_current_action',
    '__passiv':   'passive_anchor',
}
NEED_REF = {'__no_con', '__hold_p', '__intern', '__passiv'}

# ── Scan all CSV files ─────────────────────────────────────────────────
files = [f for f in os.listdir(CASES_DIR) if f.endswith('.csv')]
print('=' * 60)
print('1. TOTAL CSV FILES: %d' % len(files))

# Parse each file: stem + branch token
file_list = []
for f in files:
    base = f[:-4]
    for tok in BRANCH_TOKENS:
        if tok in base:
            stem = base.split(tok)[0]
            file_list.append((f, stem, tok))
            break

# Branch totals
branch_totals = Counter()
for _, _, tok in file_list:
    branch_totals[BRANCH_TOKENS.get(tok, tok)] += 1

print()
print('   Branch file totals:')
for b in ['candidate', 'no_control', 'passive_anchor', 'internal_current_action', 'hold_previous']:
    print('     %-25s: %d' % (b, branch_totals.get(b, 0)))

# ── Stems and their branches ───────────────────────────────────────────
stem_branches = defaultdict(set)
stem_cand_files = defaultdict(list)
for f, stem, tok in file_list:
    stem_branches[stem].add(tok)
    if tok == '__c_':
        stem_cand_files[stem].append(f)

print()
print('   Unique stems (event+checkpoint): %d' % len(stem_branches))

# ── Parse stem -> event_id + checkpoint ────────────────────────────────
stem_info = {}
for stem in stem_branches:
    m = re.match(r'^(.+?)_(\d+)$', stem)
    if m:
        stem_info[stem] = (m.group(1), int(m.group(2)))
    else:
        stem_info[stem] = (stem, None)

all_event_ids = sorted(set(eid for eid, _ in stem_info.values()))
print()
print('=' * 60)
print('2. UNIQUE EVENTS: %d' % len(all_event_ids))
for eid in all_event_ids:
    print('   - %s' % eid)

# ── Full cases ─────────────────────────────────────────────────────────
full_stems = []
partial_stems = []
events_with_full = set()

for stem, toks in stem_branches.items():
    has_all_ref = NEED_REF.issubset(toks)
    has_cand = '__c_' in toks
    if has_all_ref and has_cand:
        full_stems.append(stem)
        events_with_full.add(stem_info[stem][0])
    else:
        partial_stems.append((stem, toks))

print()
print('=' * 60)
print('3. FULL CASES (all 5 branches present): %d / %d stems' % (len(full_stems), len(stem_branches)))
print('   Unique events with full cases: %d / %d' % (len(events_with_full), len(all_event_ids)))

# Candidate files per stem
cand_counts = [len(stem_cand_files[s]) for s in full_stems]
total_cand_full = sum(cand_counts)
print('   Total candidate files (full-case stems): %d' % total_cand_full)
if cand_counts:
    print('   Candidates per stem: min=%d max=%d median=%d' % (
        min(cand_counts), max(cand_counts), sorted(cand_counts)[len(cand_counts)//2]))

# Checkpoint distribution
ckpt_counter = Counter()
for stem in full_stems:
    ckpt = stem_info[stem][1]
    ckpt_counter[ckpt] += 1
print('   Checkpoint distribution (full-case stems):')
for ck in sorted(ckpt_counter.keys()):
    print('     %d min: %d stems' % (ck, ckpt_counter[ck]))

# Event x checkpoint matrix
event_ckpt = defaultdict(set)
for stem in full_stems:
    eid, ck = stem_info[stem]
    event_ckpt[eid].add(ck)
print()
print('   Events x checkpoints (full cases):')
for eid in sorted(event_ckpt.keys()):
    cks = sorted(event_ckpt[eid])
    nc = sum(len(stem_cand_files[s]) for s in full_stems if stem_info[s][0] == eid)
    print('     %-35s  ckpts=%s  cand_files=%d' % (eid, cks, nc))

# ── Partial stems ──────────────────────────────────────────────────────
print()
print('=' * 60)
print('4. PARTIAL STEMS: %d' % len(partial_stems))
for stem, toks in sorted(partial_stems):
    missing_ref = NEED_REF - toks
    has_cand = '__c_' in toks
    missing_names = sorted([BRANCH_TOKENS.get(m, m) for m in missing_ref])
    cand_status = '%d cand' % len(stem_cand_files.get(stem, [])) if has_cand else 'NO cand'
    eid, ck = stem_info[stem]
    print('   %-35s event=%-30s ckpt=%s | %s | missing=%s' % (
        stem, eid, ck, cand_status, missing_names))

# ── Action schedule diversity (sample first 50 candidate files) ────────
print()
print('=' * 60)
print('5. ACTION SCHEDULE DIVERSITY (sampling up to 50 files)')

schedule_hashes = set()
schedule_by_event = defaultdict(set)
total_cand_checked = 0
MAX_SAMPLE = 50

all_cand = []
for stem in sorted(stem_cand_files.keys()):
    for cf in stem_cand_files[stem]:
        all_cand.append((stem, cf))

# Sample evenly across stems
sample = all_cand[:MAX_SAMPLE] if len(all_cand) <= MAX_SAMPLE else []
if len(all_cand) > MAX_SAMPLE:
    step = max(1, len(all_cand) // MAX_SAMPLE)
    sample = all_cand[::step][:MAX_SAMPLE]

for stem, cf in sample:
    fpath = os.path.join(CASES_DIR, cf)
    try:
        with open(fpath, 'r', encoding='utf-8', newline='') as fh:
            reader = csv.DictReader(fh)
            action_cols = sorted([h for h in reader.fieldnames if h.startswith('a:')])
            rows = []
            for i, row in enumerate(reader):
                if i >= 3:
                    break
                rows.append(row)
            # Build fingerprint from first 3 timesteps
            fp_parts = []
            for i, row in enumerate(rows):
                for col in action_cols[:10]:  # first 10 action cols
                    fp_parts.append('%d:%s=%s' % (i, col, row.get(col, '')))
            sched_fp = '|'.join(fp_parts)
            schedule_hashes.add(sched_fp)
            eid = stem_info[stem][0]
            schedule_by_event[eid].add(sched_fp)
            total_cand_checked += 1
    except Exception:
        pass

print('   Candidate files sampled: %d / %d total' % (total_cand_checked, len(all_cand)))
print('   Unique action schedules (3-step, 10-col fingerprint): %d' % len(schedule_hashes))
print()
print('   Per-event unique schedules (sampled):')
for eid in sorted(schedule_by_event.keys()):
    print('     %-35s %d unique' % (eid, len(schedule_by_event[eid])))

# ── Plan CSV analysis ──────────────────────────────────────────────────
print()
print('=' * 60)
print('6. PLAN CSV ANALYSIS')
plan_sigs = set()
matched = set()
if os.path.isfile(PLAN_CSV):
    with open(PLAN_CSV, 'r', encoding='utf-8', newline='') as fh:
        plan_rows = list(csv.DictReader(fh))
    print('   Total plan entries: %d' % len(plan_rows))
    plan_events = set(r['event_id'] for r in plan_rows)
    print('   Unique events in plan: %d' % len(plan_events))
    plan_sigs = set(r['case_signature'] for r in plan_rows)
    print('   Unique case_signatures: %d' % len(plan_sigs))
    reserve_count = sum(1 for r in plan_rows if r.get('reserve','').lower() == 'true')
    print('   Reserve / non-reserve: %d / %d' % (reserve_count, len(plan_rows) - reserve_count))
    
    # Action type/direction/failure_target distribution
    action_types = Counter(r['action_type'] for r in plan_rows)
    print('   Action types: %s' % dict(action_types))
    failure_targets = Counter(r['failure_target'] for r in plan_rows)
    print('   Failure targets:')
    for ft, cnt in failure_targets.most_common():
        print('     %-40s %d' % (ft, cnt))
    
    # Check which plan signatures have been generated
    generated_sigs = set()
    for stem in stem_cand_files:
        for cf in stem_cand_files[stem]:
            h = cf.replace(stem + '__c_', '').replace('.csv', '')
            generated_sigs.add(h)
    
    matched = generated_sigs & plan_sigs
    print()
    print('   Generated sigs matching plan: %d / %d (%.1f%%)' % (
        len(matched), len(plan_sigs), 100.0 * len(matched) / len(plan_sigs)))
    print('   Generated sigs NOT in plan: %d' % len(generated_sigs - plan_sigs))
    
    # Duration distribution
    dur_counter = Counter(r['duration_min'] for r in plan_rows)
    print('   Duration distribution:')
    for dur in sorted(dur_counter.keys(), key=lambda x: int(x)):
        print('     %s min: %d cases' % (dur, dur_counter[dur]))
else:
    print('   Plan CSV not found!')

# ── Manifest & lock ────────────────────────────────────────────────────
print()
print('=' * 60)
print('7. MANIFEST & LOCK STATUS')
manifest = os.path.join(AUG1_DIR, 'v4_aug1_case_manifest.csv')
lock = os.path.join(AUG1_DIR, '.writer.lock')
print('   Manifest exists: %s' % os.path.isfile(manifest))
print('   Lock file exists: %s' % os.path.isfile(lock))
if os.path.isfile(lock):
    mtime = os.path.getmtime(lock)
    age_min = (time.time() - mtime) / 60
    print('   Lock age: %.1f min' % age_min)

try:
    result = subprocess.run(['tasklist', '/FI', 'PID eq 23896'], capture_output=True, text=True)
    running = '23896' in result.stdout
    print('   PID 23896 running: %s' % running)
except:
    print('   PID 23896 check: failed')

monitor_log = os.path.join(AUG1_DIR, 'monitor.log')
if os.path.isfile(monitor_log):
    with open(monitor_log, 'r') as f:
        lines = f.readlines()
    print('   Monitor log: %d lines' % len(lines))
    for l in lines[-5:]:
        print('     %s' % l.rstrip())

# ── GRAND SUMMARY ──────────────────────────────────────────────────────
print()
print('=' * 60)
print('GRAND SUMMARY')
print('  Total CSV files:              %d' % len(files))
print('  Total candidate files:        %d' % branch_totals.get('candidate', 0))
print('  Unique stems:                 %d' % len(stem_branches))
print('  Full 5-branch case stems:     %d' % len(full_stems))
print('  Partial stems:                %d' % len(partial_stems))
print('  Unique events (all):          %d' % len(all_event_ids))
print('  Events with full cases:       %d' % len(events_with_full))
print('  Unique action schedules:      %d (sampled)' % len(schedule_hashes))
if os.path.isfile(PLAN_CSV):
    print('  Plan total:                   %d' % len(plan_rows))
    print('  Plan sigs matched:            %d / %d' % (len(matched), len(plan_sigs)))
print('  Manifest generated:           %s' % ('YES' if os.path.isfile(manifest) else 'NO'))
