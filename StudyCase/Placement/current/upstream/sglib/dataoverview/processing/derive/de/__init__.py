"""Germany product-runner registrations."""

from __future__ import annotations

from typing import Any, Callable

from ...config import CountryPipelineContext


def _products(context: CountryPipelineContext, *, force: bool = False):
    from .products import run

    return run(context, force=force)


PRODUCT_RUNNERS: dict[str, Callable[..., Any]] = {
    "regions": _products,
    "substations": _products,
    "gva": _products,
}

__all__ = ["PRODUCT_RUNNERS"]
