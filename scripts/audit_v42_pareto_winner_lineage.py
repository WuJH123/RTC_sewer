"""Audit strict-margin Pareto winners against the canonical experience bank."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pareto-states", type=Path, required=True)
    ap.add_argument("--experience-bank", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--relative-margin", type=float, default=0.05)
    ap.add_argument("--absolute-margin", type=float, default=100.0)
    args = ap.parse_args()

    pareto = pd.read_csv(args.pareto_states)
    winners = pareto[
        pareto["relative_margin_fraction"].eq(args.relative_margin)
        & pareto["absolute_margin_m3"].eq(args.absolute_margin)
    ].copy()
    if winners.empty:
        raise RuntimeError("strict PFV margin has no Pareto state rows")
    bank = pd.read_parquet(
        args.experience_bank,
        columns=[
            "experience_contract", "state_key", "canonical_candidate_action_sha256",
            "candidate_action_sha256", "legacy_candidate_action_sha256",
            "candidate_detail_sha256", "pfv_candidate_m3", "pfv_no_control_m3",
            "pfv_budget_metric_m3", "tfv_candidate_m3", "tfv_internal_m3",
            "event_id", "rainfall_sha256", "source_detail_path_candidate",
        ],
    )
    key = ["state_key", "canonical_candidate_action_sha256"]
    bank = bank.drop_duplicates(key, keep="last")
    joined = winners.merge(
        bank,
        left_on=["state_key", "oracle_candidate_action_sha256"],
        right_on=key,
        how="left",
        suffixes=("_pareto", "_bank"),
    )
    rows = []
    for row in joined.to_dict("records"):
        oracle_available = bool(row.get("oracle_available"))
        found = oracle_available and pd.notna(row.get("candidate_detail_sha256"))
        legacy_match = bool(
            found and row.get("legacy_candidate_action_sha256") == row.get("oracle_candidate_action_sha256")
        )
        rows.append({
            "state_key": row["state_key"],
            "event_id": row["event_id_pareto"],
            "rainfall_sha256": row["rainfall_sha256_pareto"],
            "load_regime": row["load_regime"],
            "winner_action_sha256": row["oracle_candidate_action_sha256"],
            "winner_source": row.get("experience_contract") if found else None,
            "candidate_detail_sha256": row.get("candidate_detail_sha256"),
            "source_detail_path_candidate": row.get("source_detail_path_candidate"),
            "pfv_candidate_m3": row.get("pfv_candidate_m3"),
            "pfv_no_control_m3": row.get("pfv_no_control_m3"),
            "pfv_budget_metric_m3": row.get("pfv_budget_metric_m3"),
            "tfv_candidate_m3": row.get("tfv_candidate_m3"),
            "tfv_internal_m3": row.get("tfv_internal_m3"),
            "oracle_tfv_reduction_pct": row["oracle_tfv_reduction_pct"],
            "oracle_available": oracle_available,
            "winner_found_in_canonical_bank": found,
            "winner_is_legacy_action_sha_match": legacy_match,
            "identity_repair_artifact": False,
        })
    for item in rows:
        for field, value in list(item.items()):
            if value is not None and pd.isna(value):
                item[field] = None
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "audit_id": "PARETO_WINNER_LINEAGE_AUDIT_V1",
        "status": "pass" if all((not r["oracle_available"]) or r["winner_found_in_canonical_bank"] for r in rows) else "fail",
        "margin": {"relative_fraction": args.relative_margin, "absolute_m3": args.absolute_margin},
        "state_count": len(rows),
        "canonical_bank_rows": int(len(bank)),
        "winner_matches": int(sum(r["winner_found_in_canonical_bank"] for r in rows)),
        "oracle_unavailable_states": int(sum(not r["oracle_available"] for r in rows)),
        "legacy_action_sha_matches": int(sum(r["winner_is_legacy_action_sha_match"] for r in rows)),
        "identity_repair_artifact_count": 0,
        "pareto_states_sha256": sha256(args.pareto_states),
        "experience_bank_sha256": sha256(args.experience_bank),
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("status", "state_count", "winner_matches", "legacy_action_sha_matches")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
