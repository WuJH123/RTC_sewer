"""Section I: freeze the Peak Boundary data role in the event usage ledger.

``ClassifyExistingGate5R`` already stamps the 16 policy-tuned Peak Boundary
events with ``used_peak_boundary=True`` and ``policy_tuned_on_event=True`` and
they inherit ``formal_eligible=False`` from ``opportunity_scanned=True``. The
only remaining Section I bookkeeping flag is ``oracle_revealed=True`` -- these
events had their oracle KPIs revealed during Peak Boundary candidate search, so
the ledger must record that fact.

This lives outside ``sewerrtc/v4`` on purpose: it must not perturb
``working_code_sha`` and therefore must not force a re-stamp of the frozen Peak
Boundary prerequisite chain. It only sets flags that have no functional effect
on split isolation (those events are already excluded from Formal via
``formal_eligible=False``); it makes the ledger honest and Section-I compliant.

The operation is idempotent and fail-closed: it only touches rows that are
already both ``used_peak_boundary`` and ``policy_tuned_on_event`` true, never
sets ``formal_eligible=True``, and re-validates the ledger before writing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sewerrtc.v4.event_splits import validate_ledger  # noqa: E402

DEFAULT_LEDGER = (
    _PROJECT_ROOT
    / "outputs"
    / "project6_dual_reference_v4"
    / "final_v4"
    / "inventory"
    / "event_usage_ledger.csv"
)


def freeze_peak_boundary_role(ledger: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with Peak Boundary events frozen as oracle-revealed.

    Selects rows that ``ClassifyExistingGate5R`` marked as tuned Peak Boundary
    development events (``used_peak_boundary`` and ``policy_tuned_on_event`` both
    true) and sets ``oracle_revealed=True`` / ``formal_eligible=False``. Raises
    if the ledger schema is invalid after the update.
    """
    required = {
        "used_peak_boundary",
        "policy_tuned_on_event",
        "oracle_revealed",
        "formal_eligible",
    }
    missing = required - set(ledger.columns)
    if missing:
        raise ValueError(f"ledger missing columns: {sorted(missing)}")
    result = ledger.copy()
    mask = (
        result["used_peak_boundary"].astype(bool)
        & result["policy_tuned_on_event"].astype(bool)
    )
    result.loc[mask, "oracle_revealed"] = True
    # Defensive: policy-tuned oracle-revealed events can never be formal.
    result.loc[mask, "formal_eligible"] = False
    validate_ledger(result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=DEFAULT_LEDGER,
        help="Path to event_usage_ledger.csv",
    )
    args = parser.parse_args(argv)
    if not args.ledger.exists():
        print(f"LEDGER_MISSING path={args.ledger}")
        return 1
    ledger = pd.read_csv(args.ledger)
    before = int(
        (
            ledger["used_peak_boundary"].astype(bool)
            & ledger["policy_tuned_on_event"].astype(bool)
            & ledger["oracle_revealed"].astype(bool)
        ).sum()
    )
    frozen = freeze_peak_boundary_role(ledger)
    tuned = int(
        (
            frozen["used_peak_boundary"].astype(bool)
            & frozen["policy_tuned_on_event"].astype(bool)
        ).sum()
    )
    after = int(
        (
            frozen["used_peak_boundary"].astype(bool)
            & frozen["policy_tuned_on_event"].astype(bool)
            & frozen["oracle_revealed"].astype(bool)
        ).sum()
    )
    frozen.to_csv(args.ledger, index=False)
    formal_true = int(frozen["formal_eligible"].astype(bool).sum())
    print(
        "FREEZE_PEAK_BOUNDARY_ROLE "
        f"tuned_peak_events={tuned} "
        f"oracle_revealed_before={before} oracle_revealed_after={after} "
        f"formal_eligible_true_total={formal_true}"
    )
    if after != tuned:
        print("FREEZE_INCOMPLETE oracle_revealed not set on all tuned events")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
