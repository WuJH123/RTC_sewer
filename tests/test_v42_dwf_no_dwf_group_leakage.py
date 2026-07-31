"""Tests for V4.2 DWF / no-DWF group leakage (v42_dwf_audit).

Verifies:
- Same rainfall fingerprint groups DWF and no-DWF together
- base_rainfall_fingerprint is consistent
"""

from __future__ import annotations

import pytest

from sewerrtc.v4.v42_dwf_audit import _base_rainfall_fingerprint


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBaseRainfallFingerprint:
    """base_rainfall_fingerprint is consistent."""

    def test_same_event_same_fingerprint(self):
        """Same event_id always produces the same fingerprint."""
        fp1 = _base_rainfall_fingerprint("EVT_001")
        fp2 = _base_rainfall_fingerprint("EVT_001")
        assert fp1 == fp2

    def test_different_event_different_fingerprint(self):
        """Different event_ids produce different fingerprints."""
        fp_a = _base_rainfall_fingerprint("EVT_A")
        fp_b = _base_rainfall_fingerprint("EVT_B")
        assert fp_a != fp_b

    def test_fingerprint_is_hex_string(self):
        """Fingerprint should be a hex SHA-256 string."""
        fp = _base_rainfall_fingerprint("EVT_TEST")
        assert len(fp) == 64  # SHA-256 hex length
        assert all(c in "0123456789abcdef" for c in fp)


class TestDWFNoDWFGrouping:
    """Same rainfall fingerprint groups DWF and no-DWF together."""

    def test_dwf_and_no_dwf_same_event_share_fingerprint(self):
        """DWF and no-DWF samples from same event share base_rainfall_fingerprint."""
        event_id = "EVT_SHARED"
        fp_dwf = _base_rainfall_fingerprint(event_id)
        fp_no_dwf = _base_rainfall_fingerprint(event_id)
        assert fp_dwf == fp_no_dwf

    def test_grouping_by_fingerprint(self):
        """Samples with same fingerprint are in the same group."""
        # Simulate: two samples (one DWF, one no-DWF) from same event
        event_a = "EVT_A"
        event_b = "EVT_B"

        samples = [
            {"event_id": event_a, "dwf": True, "fp": _base_rainfall_fingerprint(event_a)},
            {"event_id": event_a, "dwf": False, "fp": _base_rainfall_fingerprint(event_a)},
            {"event_id": event_b, "dwf": True, "fp": _base_rainfall_fingerprint(event_b)},
        ]

        # Group by fingerprint
        groups = {}
        for s in samples:
            groups.setdefault(s["fp"], []).append(s)

        # EVT_A samples (DWF + no-DWF) should be in same group
        fp_a = _base_rainfall_fingerprint(event_a)
        group_a = groups[fp_a]
        assert len(group_a) == 2
        dwf_values = {s["dwf"] for s in group_a}
        assert dwf_values == {True, False}  # both DWF and no-DWF present
