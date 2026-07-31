"""Forecast contract validation for Project6 V3."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping
import json


def load_forecast_contract(path: str | Path) -> dict[str, object]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def validate_forecast_record(record: Mapping[str, object], contract: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    for field in contract.get("required_fields", []):
        if field not in record:
            errors.append(f"missing:{field}")
    if record.get("truth_available_to_controller") is not False:
        errors.append("truth_available_to_controller_must_be_false")
    min_horizon = contract.get("minimum_required_horizon_min")
    if min_horizon is not None and "forecast_horizon" in record:
        try:
            if float(record["forecast_horizon"]) < float(min_horizon):
                errors.append("insufficient_forecast_horizon")
        except (TypeError, ValueError):
            errors.append("forecast_horizon_not_numeric")
    expected_units = contract.get("rainfall_units")
    if expected_units and record.get("rainfall_units") != expected_units:
        errors.append("rainfall_units_mismatch")
    expected_tz = contract.get("timezone")
    if expected_tz and record.get("timezone") != expected_tz:
        errors.append("timezone_mismatch")
    return errors
