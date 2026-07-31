"""Frozen Project6 V4 final-pipeline interfaces.

This package contains orchestration, contracts, planning and audit logic.  The
authoritative hydraulic implementation remains in
``sewerrtc.simulation.pyswmm_runner``.
"""

from .runtime import EXIT_BLOCKED, EXIT_INCOMPLETE, EXIT_PASS, EXIT_RUNTIME_ERROR

__all__ = [
    "EXIT_PASS",
    "EXIT_BLOCKED",
    "EXIT_INCOMPLETE",
    "EXIT_RUNTIME_ERROR",
]
