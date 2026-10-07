"""Australia product-runner registrations."""

from __future__ import annotations

from typing import Any, Callable

from ...config import CountryPipelineContext


def _stations(context: CountryPipelineContext, *, force: bool = False):
    from .stations import run

    return run(context, force=force)


def _regions(context: CountryPipelineContext, *, force: bool = False):
    from .regions import run

    return run(context, force=force)


def _ledger(context: CountryPipelineContext, *, force: bool = False):
    from .ledger import run

    return run(context, force=force)


def _fy2024(context: CountryPipelineContext, *, force: bool = False):
    from .fy2024 import run

    # The publication runner is already idempotent through its input/output
    # receipt; it accepts the resolved country context rather than a force flag.
    return run(context)


PRODUCT_RUNNERS: dict[str, Callable[..., Any]] = {
    "stations": _stations,
    "regions": _regions,
    "ledger": _ledger,
    "fy2024": _fy2024,
}

__all__ = ["PRODUCT_RUNNERS"]
