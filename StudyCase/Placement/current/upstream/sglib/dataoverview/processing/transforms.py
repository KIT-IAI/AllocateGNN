"""Bind product identifiers to country/protocol behavior."""

from __future__ import annotations

from importlib import import_module
from typing import Callable

from .config import build_context
from .unit import Unit, UnitContext


def _country(context: UnitContext, unit: Unit):
    if unit.country is None:
        raise ValueError(f"{unit.id} requires a country")
    return build_context(context.repo_root, context.config)


def _country_product(context: UnitContext, unit: Unit) -> None:
    country_context = _country(context, unit)
    product = unit.id.split(".")[1]
    module = import_module(
        f"sglib.dataoverview.processing.derive.{unit.country}"
    )
    runners = getattr(module, "PRODUCT_RUNNERS", None)
    if not isinstance(runners, dict) or product not in runners:
        raise ValueError(
            f"no {product!r} transform registered for country {unit.country!r}"
        )
    runners[product](country_context, force=context.refresh)


def _grid(context: UnitContext, unit: Unit) -> None:
    from .pipeline import run_grid

    run_grid(_country(context, unit), force=context.refresh)


def _features(context: UnitContext, unit: Unit) -> None:
    from .pipeline import run_features

    run_features(_country(context, unit), force=context.refresh)


def _inventory(context: UnitContext, unit: Unit) -> None:
    from sglib.dataoverview.overview.inventory import write_inventory

    write_inventory(_country(context, unit))


def _matrix(context: UnitContext, unit: Unit) -> None:
    from sglib.dataoverview.overview.matrix import write_matrix

    write_matrix(context.repo_root)


_RUNNERS: dict[str, Callable[[UnitContext, Unit], None]] = {
    "grid": _grid,
    "features": _features,
    "inventory": _inventory,
    "matrix": _matrix,
}


def runner_for(product: str):
    return _RUNNERS.get(product, _country_product)
