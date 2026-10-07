"""Compatibility exports for the shared engineering-admission authority."""

from sglib.core.infra.engineering_admission import (
    AUTHORITY_SCHEMA,
    COUNTRY_SCHEMA,
    EngineeringAdmission,
    EngineeringAdmissionError,
    chunked_nearest_assignment,
    evaluate_region,
    load_engineering_admission,
    validate_pre_submission,
)

__all__ = [
    "AUTHORITY_SCHEMA",
    "COUNTRY_SCHEMA",
    "EngineeringAdmission",
    "EngineeringAdmissionError",
    "chunked_nearest_assignment",
    "evaluate_region",
    "load_engineering_admission",
    "validate_pre_submission",
]
