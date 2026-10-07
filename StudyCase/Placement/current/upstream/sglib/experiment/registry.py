"""Experiment DAG units and their completion contracts (plan 006a §4).

Units mirror the sealed 006 products: three preflight units, one observation
unit per region (seven observation families inside), one planning unit per
region (field × seed solver coordinates inside), country-level bounds, defense
units, and a final audit. Completion is read from content-chain
receipts whose committed scientific parameters must equal the registered ones.
"""

from __future__ import annotations

from functools import partial
from sglib.core.chain.stage import topological_order as _chain_topological_order

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from sglib.core.infra.content_chain import ContentChainError, verify_chain

from .config import LoadedExperimentConfig, registered_values_match


RECEIPT_SCHEMA = "sg_content_chain_receipt_v1"
AUDIT_SCHEMA = "sg_experiment_audit_v1"
PREPARED_SCHEMA = "sg_prepared_experiment_unit_v1"
PREPARED_MARKER = "_prepared.json"
UNIT_TYPES = {
    ("preflight", "connection"): "preflight_connection",
    ("preflight", "planning_pool"): "preflight_planning_pool",
    ("observe", None): "observe",
    ("planning", None): "planning",
    ("bounds", "c6"): "bounds",
    ("defense", "support"): "defense",
}


@dataclass(frozen=True)
class ExperimentUnit:
    id: str
    step: str
    country: str
    member: str
    depends_on: tuple[str, ...]
    coordinates: tuple[tuple[str, int | None], ...] = ()


class ExperimentRegistryError(ValueError):
    pass


def unit_type(unit: ExperimentUnit) -> str | None:
    return UNIT_TYPES.get((unit.step, unit.member)) or UNIT_TYPES.get((unit.step, None))


def siting_fields(config: LoadedExperimentConfig, candidate_registry: Mapping[str, Any]) -> list[tuple[str, tuple[int | None, ...]]]:
    """Materialized, non-QA candidates eligible for the planning task, with their seeds."""

    eligibility = str(config.values["planning"]["eligibility"])
    fields = []
    for item in candidate_registry["candidates"]:
        if item["materialize"] and not item["qa_only"] and eligibility in item["task_eligibility"]:
            seeds = tuple(item["seed_policy"]["seeds"]) or (None,)
            fields.append((str(item["label"]), seeds))
    if not fields:
        raise ExperimentRegistryError("candidate registry has no planning-eligible candidates")
    return fields


def build_registry(country: str, config: LoadedExperimentConfig, candidate_registry: Mapping[str, Any]) -> dict[str, ExperimentUnit]:
    prefix = str(country)
    units: dict[str, ExperimentUnit] = {}

    def add(step: str, member: str, dependencies: Iterable[str], coordinates=()) -> str:
        key = f"{prefix}.{step}.{member}"
        if key in units:
            raise ExperimentRegistryError(f"duplicate Experiment unit: {key}")
        units[key] = ExperimentUnit(key, step, prefix, member, tuple(dependencies), tuple(coordinates))
        return key

    connection = add("preflight", "connection", ())
    pool = add("preflight", "planning_pool", ())
    fields = siting_fields(config, candidate_registry)
    coordinates = tuple((label, seed) for label, seeds in fields for seed in seeds)
    observe_units, planning_units = [], []
    for region in config.values["regions"]:
        observe_units.append(add("observe", region, (connection,)))
        planning_units.append(add("planning", region, (pool,), coordinates))
    bounds = add("bounds", "c6", observe_units)
    defense = add("defense", "support", (*observe_units, *planning_units))
    add("audit", "audit", (bounds, defense, *observe_units, *planning_units))
    topological_order(units)
    return units


topological_order = partial(_chain_topological_order, label='Experiment', method='breadth', error=ExperimentRegistryError)


def seed_directory(config: LoadedExperimentConfig, seed: int | None) -> str:
    return str(config.values["planning"]["static_seed_directory"]) if seed is None else f"seed_{int(seed)}"


def coordinate_root(unit: ExperimentUnit, root: Path, config: LoadedExperimentConfig, field: str, seed: int | None) -> Path:
    return root / "planning" / unit.member / field / seed_directory(config, seed)


def expected_node_id(unit: ExperimentUnit, config: LoadedExperimentConfig, field: str | None = None, seed: int | None = None) -> str:
    templates = config.values["node_ids"]
    kind = unit_type(unit)
    if kind is None:
        raise ExperimentRegistryError(f"{unit.id}: no node id template")
    return str(templates[kind]).format(country=unit.country, region=unit.member, field=field or "",
                                       seed="0" if seed is None else str(int(seed)))


def unit_output_path(unit: ExperimentUnit, root: Path) -> Path:
    """The single completion marker of a unit (planning: its region directory)."""

    if unit.step == "preflight":
        return root / "preflight" / unit.member / "receipt.json"
    if unit.step == "observe":
        return root / "observations" / unit.member / "receipt.json"
    if unit.step == "planning":
        return root / "planning" / unit.member
    if unit.step == "bounds":
        return root / "bounds/receipt.json"
    if unit.step == "defense":
        return root / "defense/receipt.json"
    if unit.step == "audit":
        return root / "audit.json"
    raise ExperimentRegistryError(f"unknown Experiment step: {unit.step}")


def _read(path: Path) -> dict[str, Any] | None:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def receipt_status(path: Path, node_id: str, kind: str, config: LoadedExperimentConfig) -> str:
    """DONE only for a verified receipt of the expected node whose registered values match."""

    if not path.exists():
        return "PENDING"
    document = _read(path)
    if document is None or document.get("schema_version") != RECEIPT_SCHEMA:
        return "INVALID"
    try:
        verify_chain(document)
    except (ContentChainError, ValueError, KeyError, TypeError):
        return "INVALID"
    if document.get("node_id") != node_id:
        return "INVALID"
    parameters = document.get("commitment", {}).get("scientific_parameters")
    if not isinstance(parameters, Mapping) or registered_values_match(kind, config.registrations, parameters):
        return "INVALID"
    outputs = document.get("outputs")
    if not isinstance(outputs, Mapping) or not outputs:
        return "INVALID"
    for name in outputs:
        candidates = (path.parent / name, path.parent / f"{name}.npz", path.parent / f"{name}.csv")
        if not any(candidate.is_file() for candidate in candidates):
            return "INVALID"
    return "DONE"


def unit_status(unit: ExperimentUnit, root: Path, config: LoadedExperimentConfig) -> str:
    marker = unit_output_path(unit, root)
    if unit.step == "audit":
        if not marker.exists():
            return "PENDING"
        document = _read(marker)
        return "DONE" if document and document.get("schema_version") == AUDIT_SCHEMA and document.get("status") == "PASS" else "INVALID"
    kind = unit_type(unit)
    if unit.step == "planning":
        states = {receipt_status(coordinate_root(unit, root, config, field, seed) / "receipt.json",
                                 expected_node_id(unit, config, field, seed), kind, config)
                  for field, seed in unit.coordinates}
        if "INVALID" in states:
            return "INVALID"
        if states == {"DONE"}:
            return "DONE"
        prepared = marker / PREPARED_MARKER
        if prepared.exists():
            document = _read(prepared)
            return "PREPARED" if document and document.get("schema_version") == PREPARED_SCHEMA else "INVALID"
        return "PENDING"
    state = receipt_status(marker, expected_node_id(unit, config), kind, config)
    if state == "PENDING" and unit.step == "observe":
        prepared = marker.parent / PREPARED_MARKER
        if prepared.exists():
            document = _read(prepared)
            return "PREPARED" if document and document.get("schema_version") == PREPARED_SCHEMA else "INVALID"
    return state


def planning_progress(unit: ExperimentUnit, root: Path, config: LoadedExperimentConfig) -> tuple[int, int]:
    """(done, expected) solver coordinates of a planning unit."""

    kind = unit_type(unit)
    done = sum(receipt_status(coordinate_root(unit, root, config, field, seed) / "receipt.json",
                              expected_node_id(unit, config, field, seed), kind, config) == "DONE"
               for field, seed in unit.coordinates)
    return done, len(unit.coordinates)
