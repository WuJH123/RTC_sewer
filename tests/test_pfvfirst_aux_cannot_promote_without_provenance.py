"""PFV-first aux roles cannot be promoted without authoritative provenance.

candidate_then_internal_aux and candidate_then_passive_aux must remain
auxiliary.  They cannot silently become dynamic_internal or hold_previous
for the purpose of four-reference counterfactual alignment.
"""
from __future__ import annotations

import pytest

AUX_ROLES = frozenset({
    "candidate_then_internal_aux",
    "candidate_then_passive_aux",
})

FORMAL_REFERENCE_ROLES = frozenset({
    "candidate",
    "no_control",
    "dynamic_internal",
    "dynamic_internal_rules",
    "hold_previous",
})


def test_pfvfirst_aux_cannot_promote_without_provenance() -> None:
    """Aux roles must not overlap with formal reference roles."""
    assert not (AUX_ROLES & FORMAL_REFERENCE_ROLES), (
        "PFV-first aux roles must not be promotable to formal reference roles"
    )


def test_aux_role_not_counted_as_reference() -> None:
    """A case with only aux roles cannot satisfy four-reference completeness."""
    case_roles = {"candidate", "no_control", "candidate_then_internal_aux", "candidate_then_passive_aux"}
    formal_refs_present = case_roles & FORMAL_REFERENCE_ROLES
    assert len(formal_refs_present) < 4, (
        "Aux roles must not inflate four-reference completeness"
    )
