"""Errors shared by learned task preparation and path validation."""

from __future__ import annotations




class TaskMatrixError(ValueError):
    """Raised when task expansion or frozen reuse identity is invalid."""



class TrainingTaskError(RuntimeError):
    """Raised for an invalid, failed, or incomplete prepared task."""
