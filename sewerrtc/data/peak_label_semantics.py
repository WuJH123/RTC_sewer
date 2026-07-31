from __future__ import annotations

import numpy as np


RISK_LABEL_CHANNELS = (
    "PFV_rate",
    "TFV_rate",
    "running_peak_TFV_rate",
)


def _validate_risk_sequence(risk_rate_seq: np.ndarray) -> np.ndarray:
    values = np.asarray(risk_rate_seq)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("risk_rate_seq must have shape [N,H,3]")
    if not np.isfinite(values).all():
        raise ValueError("risk_rate_seq contains NaN or Inf")
    return values


def repair_observational_risk_rate_seq(risk_rate_seq: np.ndarray) -> np.ndarray:
    """Return PFV rate, TFV rate, and the running TFV peak for each horizon."""
    values = _validate_risk_sequence(risk_rate_seq)
    repaired = values.copy()
    repaired[:, :, 2] = np.maximum.accumulate(values[:, :, 1], axis=1)
    return repaired


def repair_paired_risk_rate_sequences(
    reference_risk_rate_seq: np.ndarray,
    delta_risk_rate_seq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Repair same-state labels while preserving candidate-minus-reference effects."""
    reference = _validate_risk_sequence(reference_risk_rate_seq)
    delta = _validate_risk_sequence(delta_risk_rate_seq)
    if reference.shape != delta.shape:
        raise ValueError("reference and delta risk sequences must have identical shapes")
    candidate = reference + delta
    repaired_reference = repair_observational_risk_rate_seq(reference)
    repaired_candidate = repair_observational_risk_rate_seq(candidate)
    return repaired_reference, repaired_candidate - repaired_reference


def peak_label_semantics_valid(risk_rate_seq: np.ndarray, *, atol: float = 1.0e-5) -> bool:
    values = _validate_risk_sequence(risk_rate_seq)
    expected = np.maximum.accumulate(values[:, :, 1], axis=1)
    return bool(np.allclose(values[:, :, 2], expected, rtol=0.0, atol=float(atol)))
