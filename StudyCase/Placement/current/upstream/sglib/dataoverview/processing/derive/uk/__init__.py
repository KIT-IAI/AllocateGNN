"""United Kingdom product-runner registrations."""

from __future__ import annotations

from typing import Any, Callable

from ...config import CountryPipelineContext


def _regions(context: CountryPipelineContext, *, force: bool = False):
    from .regions import run

    return run(context, force=force)


def _substations(context: CountryPipelineContext, *, force: bool = False):
    from .substations import run

    return run(context, force=force)


PRODUCT_RUNNERS: dict[str, Callable[..., Any]] = {
    "regions": _regions,
    "substations": _substations,
}

__all__ = ["PRODUCT_RUNNERS"]
