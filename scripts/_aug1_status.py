"""One-shot Aug1 status reporter."""
import os, time, collections, csv, psutil, yaml

base = r"outputs\project6_dual_reference_v4\dual_reference_aug1"
cases = os.path.join(base, "cases")

csvs = [f for f in os.listdir(cases) if f.endswith(".csv")]
inps = [f for f in os.listdir(cases) if f.endswith(".inp")]
outs = [f for f in os.listdir(cases) if f.endswith(".out")]

groups = collections.defaultdict(lambda: {"c": 0, "refs": set()})
for f in csvs:
    stem, br = f[:-4].split("__", 1)
    tok = br.split("_")[0]
    if tok == "c":
        groups[stem]["c"] += 1
    else:
        groups[stem]["refs"].add(tok)

need = {"hold", "no", "intern", "passiv"}
complete = [s for s, g in groups.items() if need.issubset(g["refs"])]
incomplete = [s for s, g in groups.items() if not need.issubset(g["refs"])]
valid_cand = sum(groups[s]["c"] for s in complete)
total_cand = sum(g["c"] for g in groups.values())


def eid(stem):
    return stem.rsplit("_", 1)[0]


ev_any = set(eid(s) for s in groups)
ev_complete = set(eid(s) for s in complete)

cfg = yaml.safe_load(open(r"configs\wuhan_project6_dual_reference_v4.yaml"))
aug = ((cfg.get("v4", {}) or {}).get("aug1", {}) or {})

plan = os.path.join(base, "v4_aug1_case_plan.csv")
plan_rows = list(csv.DictReader(open(plan))) if os.path.exists(plan) else []
plan_events = len(set(r.get("event_id", "") for r in plan_rows))

manifest = os.path.join(base, "v4_aug1_generation_manifest.csv")
manifest_exists = os.path.exists(manifest)
manifest_rows = 0
if manifest_exists:
    manifest_rows = len(list(csv.DictReader(open(manifest))))

lock = os.path.join(base, ".writer.lock")
lock_age = None
if os.path.exists(lock):
    lock_age = (time.time() - os.path.getmtime(lock)) / 60

alive_pids = []
for p in psutil.process_iter(["pid", "name", "create_time", "cmdline"]):
    try:
        cl = " ".join(p.info.get("cmdline") or [])
    except Exception:
        cl = ""
    if "python" in (p.info.get("name") or "").lower() and (
        "aug1" in cl or "dual_reference" in cl or "action_effect" in cl
    ):
        age = (time.time() - p.info["create_time"]) / 60
        alive_pids.append((p.info["pid"], age, cl[:140]))

now = time.time()
all_files = [(f, now - os.path.getmtime(os.path.join(cases, f))) for f in csvs]
all_files.sort(key=lambda x: x[1])
newest = all_files[0] if all_files else None

print("=== 1. 文件统计 ===")
print("CSV 总数:", len(csvs))
print("  - Candidate (c_*) CSV:", total_cand)
print("  - 参考分支 CSV:", len(csvs) - total_cand)
print("INP 文件:", len(inps))
print("OUT 文件:", len(outs))
print("组 (event+checkpoint) 总数:", len(groups))
print("  - 引用分支齐全的组:", len(complete))
print("  - 引用分支不全的组:", len(incomplete))
for s in incomplete[:6]:
    print("    *", s, " refs=", groups[s]["refs"], " cand=", groups[s]["c"])
print("有效候选案例数（引用齐全的组内 c_* CSV）:", valid_cand)
print("覆盖事件数（齐全组）:", len(ev_complete))
print("覆盖事件数（所有组）:", len(ev_any))
print("最新 CSV 文件:", newest[0], "(%.1f 秒前)" % newest[1])

print()
print("=== 2. 配置目标 ===")
print("effective_target:", aug.get("effective_target"))
print("reserve:", aug.get("reserve"))
print("计划总预算:", aug.get("effective_target", 0) + aug.get("reserve", 0))
print("minimum_events:", aug.get("minimum_events"))
print("minimum_validation_events:", aug.get("minimum_validation_events"))
print("Plan 行数:", len(plan_rows))
print("Plan 事件数:", plan_events)

print()
print("=== 3. 进度分析 ===")
target = aug.get("effective_target", 1600)
min_ev = aug.get("minimum_events", 24)
print("有效候选 %d / 目标 %d  ->  达成率 %.1f%%" % (valid_cand, target, valid_cand / target * 100))
ev_ok = len(ev_complete) >= min_ev
print("事件数 %d / 最低 %d  ->  %s" % (len(ev_complete), min_ev, "达标" if ev_ok else "未达标"))
if valid_cand >= target and ev_ok:
    print("结论：硬门（>=1600 案例、>=24 事件）均已达标")
else:
    print(
        "缺口：还需 %d 个案例 / 还需 %d 个事件"
        % (max(0, target - valid_cand), max(0, min_ev - len(ev_complete)))
    )

print()
print("=== 4. 进程状态 ===")
print("Manifest 已生成:", manifest_exists, " (行数 %d)" % manifest_rows)
if lock_age is not None:
    print("Lock 文件: 存在, 年龄 %.1f 分钟" % lock_age)
else:
    print("Lock 文件: 不存在")
if alive_pids:
    for pid, age, cmd in alive_pids:
        print("  PID %d  年龄 %.1f 分钟  cmd=%s" % (pid, age, cmd))
else:
    print("未发现 Aug1 相关 Python 进程")

print()
print("=== 5. 后续步骤条件 ===")
print("- Aug1 Build: 需 manifest 存在且行数 >=1600，事件数 >=24")
print("- Aug1 Train: 需 Build 产出 v4_aug1_dataset_manifest.csv 且样本数达标")
print("- Aug1 Gate:  需 Train 产出模型报告")
print("- 3 事件 Smoke: 需 Gate 通过（六输出头准确率 >=0.7）")
