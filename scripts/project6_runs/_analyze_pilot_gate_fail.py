"""Read-only diagnostics for the two failing AuditPilotDataset checks."""
from pathlib import Path

import pandas as pd

PROJECT = Path(r"e:\RTC_sewer\Project6")
DS = (
    PROJECT
    / "outputs"
    / "project6_dual_reference_v4"
    / "final_v4"
    / "pilot"
    / "dataset"
    / "pilot_sample_manifest.csv"
)


def main() -> None:
    df = pd.read_csv(DS)
    print("total rows:", len(df))
    print("columns:", list(df.columns))
    key = ["event_id", "checkpoint_id"]

    # ---- flat fraction ----
    flat = df["confirmed_flat"].astype(bool)
    print("\n== confirmed_flat ==")
    print("count=", int(flat.sum()), "frac=%.4f" % flat.mean())

    # ---- joint on responsive checkpoints ----
    r = df[df["checkpoint_role"] == "responsive"].copy()
    cj = r.groupby(key)["joint_noninferior"].any()
    print("\n== joint on responsive checkpoints ==")
    print("responsive checkpoints=", len(cj))
    print("with joint=", int(cj.sum()), "frac=%.4f" % cj.mean())

    # ---- per responsive state accepted informative (reserve rule) ----
    print("\n== per-responsive-state accepted count (reserve rule) ==")
    noop_col = "is_noop" if "is_noop" in r.columns else None
    inf = r[~r[noop_col].astype(bool)] if noop_col else r
    g = inf.groupby(key).size()
    print("responsive states=", len(g))
    print("min=", int(g.min()), "max=", int(g.max()))
    print("states below 6:", int((g < 6).sum()))
    print("smallest:", {str(k): int(v) for k, v in g.sort_values().head(12).items()})

    # candidate budget currently used per responsive state
    tot = r.groupby(key).size()
    print("\n== per-responsive-state total candidates ==")
    print("min=", int(tot.min()), "max=", int(tot.max()))

    # ---- where do the confirmed_flat samples live ----
    print("\n== confirmed_flat by checkpoint_role ==")
    print(df.groupby("checkpoint_role")["confirmed_flat"].agg(["sum", "size"]))
    low = df[df["checkpoint_role"] != "responsive"]
    print("low-opportunity rows=", len(low))
    print(
        "low-opportunity locally_responsive rate=%.3f"
        % low["locally_responsive"].astype(bool).mean()
    )
    print(
        "low-opportunity flat_state sum=",
        int(low["flat_state"].astype(bool).sum()),
        "confirmed_flat sum=",
        int(low["confirmed_flat"].astype(bool).sum()),
    )

    # ---- which responsive checkpoints lack joint ----
    cj2 = r.groupby(key)["joint_noninferior"].any()
    lacking = cj2[~cj2]
    print("\n== responsive checkpoints WITHOUT joint (%d) ==" % len(lacking))
    for k in lacking.index:
        sub = r[(r["event_id"] == k[0]) & (r["checkpoint_id"] == k[1])]
        print(
            "  %s | pfv_safe=%d tfv_noninf=%d peak_noninf=%d"
            % (
                k[1],
                int(sub["pfv_safe"].astype(bool).sum()),
                int(sub["tfv_noninferior"].astype(bool).sum()),
                int(sub["peak_noninferior"].astype(bool).sum()),
            )
        )
    print("\n== joint checkpoints by event ==")
    got = cj2[cj2]
    print(sorted({k[0] for k in got.index}))


if __name__ == "__main__":
    main()
