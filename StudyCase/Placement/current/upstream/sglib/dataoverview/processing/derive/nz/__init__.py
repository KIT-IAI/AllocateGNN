"""New Zealand DataOverview product bindings."""

from __future__ import annotations

from typing import Any

from .core9 import (
    build_core9_dataoverview_from_fresh,
    build_size_gate,
)


def _core9(context, force: bool = False) -> Any:
    return build_core9_dataoverview_from_fresh(
        context.repo_root, context.derived_root, force=force
    )


def _admission(context, force: bool = False) -> Any:
    return build_size_gate(context.repo_root, context.derived_root, force=force)


# The shared dispatcher discovers this mapping dynamically.  All country
# branching therefore remains in the protocol adapter rather than core logic.
PRODUCT_RUNNERS = {
    "station_ledger_2024": _core9,
    "station_sites": _core9,
    "analysis_regions": _core9,
    "admission_gate": _admission,
}


__all__ = [
    "PRODUCT_RUNNERS",
    "build_core9_dataoverview_from_fresh",
    "build_size_gate",
]
