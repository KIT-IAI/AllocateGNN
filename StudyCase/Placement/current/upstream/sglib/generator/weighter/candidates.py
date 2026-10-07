"""Pure loader and derived views for the frozen Generator candidate registry.

The registry path is deliberately never inferred here.  A stage entrypoint must
resolve the file from its own configuration and pass either that path or the
validated document explicitly.  This keeps package code independent of pipeline
layout and prevents a second, silently selected registry authority.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any
from sglib.core.infra.paths import resolve_case_path


SCHEMA_VERSION = "cuz_candidate_registry_v2"
EVIDENCE_LAYERS = ("L11", "L22", "LX")
TASKS = ("reconstruction", "siting", "sizing", "connection_cost")
LEARNED_SEEDS = (42, 123, 456)

_FIELDS = {
    "label",
    "family",
    "operator",
    "auxiliary",
    "seed_policy",
    "training_config",
    "training_action",
    "evidence",
    "task_eligibility",
    "deployment_scope",
    "materialize",
    "qa_only",
}
_FAMILIES = frozenset({"Uni", "GPM", "Equal", "MLP", "GNN"})
_OPERATORS = frozenset({"base", "multiply", "add", "prior", "fusion"})
_AUXILIARIES = frozenset({"none", "ntl", "proximity", "ntl_proximity"})
_COUNTRY_PATTERN = re.compile(r"^[A-Z]{2}$")


class CandidateRegistryError(ValueError):
    """Raised when a registry document or query violates the frozen contract."""


def validate_candidate_registry(document: Any) -> dict[str, Any]:
    """Validate a registry document and return a detached copy."""

    if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION:
        raise CandidateRegistryError("legacy or unknown candidate registry schema")
    active_countries = document.get("active_countries")
    if (
        not isinstance(active_countries, list)
        or not active_countries
        or any(
            not isinstance(country, str)
            or _COUNTRY_PATTERN.fullmatch(country) is None
            for country in active_countries
        )
        or len(active_countries) != len(set(active_countries))
    ):
        raise CandidateRegistryError(
            "active_countries must be unique uppercase ISO-like two-letter codes"
        )
    if tuple(document.get("tasks", ())) != TASKS:
        raise CandidateRegistryError("candidate registry task order differs from contract")

    aliases = document.get("aliases")
    configs = document.get("training_configs")
    candidates = document.get("candidates")
    if not isinstance(aliases, dict) or not isinstance(configs, dict):
        raise CandidateRegistryError("aliases and training_configs must be objects")
    if not isinstance(candidates, list) or not candidates:
        raise CandidateRegistryError("candidates must be a non-empty array")

    for name, config in configs.items():
        if not isinstance(name, str) or not name or not isinstance(config, dict):
            raise CandidateRegistryError("invalid training configuration record")
        family = config.get("family")
        value = config.get("config")
        if name == "none":
            if family is not None or value is not None:
                raise CandidateRegistryError("training config 'none' must carry null identity")
        elif family not in {"mlp", "gnn"} or not isinstance(value, str) or not value:
            raise CandidateRegistryError(f"{name}: invalid learned training identity")

    labels: set[str] = set()
    for item in candidates:
        if not isinstance(item, dict) or not _FIELDS <= set(item):
            raise CandidateRegistryError("candidate record is missing required fields")
        label = item.get("label")
        if not isinstance(label, str) or not label or label in labels or label in aliases:
            raise CandidateRegistryError(f"invalid or duplicate canonical label: {label!r}")
        labels.add(label)

        policy = item.get("seed_policy")
        if not isinstance(policy, Mapping) or policy.get("mode") not in {"static", "seeded"}:
            raise CandidateRegistryError(f"{label}: invalid seed policy")
        expected_seeds = () if policy["mode"] == "static" else LEARNED_SEEDS
        if tuple(policy.get("seeds", ())) != expected_seeds:
            raise CandidateRegistryError(f"{label}: seed policy differs from frozen seeds")

        structural = (
            item.get("family") in _FAMILIES
            and item.get("operator") in _OPERATORS
            and item.get("auxiliary") in _AUXILIARIES
            and isinstance(item.get("evidence"), list)
            and set(item["evidence"]) <= set(EVIDENCE_LAYERS)
            and isinstance(item.get("task_eligibility"), list)
            and set(item["task_eligibility"]) <= set(TASKS)
            and item.get("training_config") in configs
            and item.get("training_action") in {"none", "train", "reuse_checkpoint"}
            and type(item.get("materialize")) is bool
            and type(item.get("qa_only")) is bool
        )
        if not structural:
            raise CandidateRegistryError(f"{label}: invalid candidate contract")
        if (item["training_config"] == "none") != (item["training_action"] == "none"):
            raise CandidateRegistryError(f"{label}: inconsistent training action")
        if item["qa_only"] and (
            not item["materialize"]
            or item["evidence"]
            or item["task_eligibility"]
        ):
            raise CandidateRegistryError(
                f"{label}: QA-only identities must materialize without evidence/tasks"
            )
        if item["materialize"] and not item["qa_only"] and not (
            item["evidence"] and item["task_eligibility"]
        ):
            raise CandidateRegistryError(f"{label}: materialized identity lacks evidence/tasks")
        if {"L22", "LX"} <= set(item["evidence"]):
            raise CandidateRegistryError(f"{label}: L22 and LX identities must be disjoint")

        proximity = item["auxiliary"] in {"proximity", "ntl_proximity"}
        if proximity and (
            item.get("deployment_scope") != "transductive_known_sites"
            or set(item["task_eligibility"]) - {"reconstruction"}
        ):
            raise CandidateRegistryError(
                f"{label}: proximity is forbidden outside known-site reconstruction"
            )

    if any(
        not isinstance(alias, str) or not alias or target not in labels
        for alias, target in aliases.items()
    ):
        raise CandidateRegistryError("every alias must map to a canonical label")
    return deepcopy(document)


def load_candidate_registry(path: str | Path) -> dict[str, Any]:
    """Load the one explicitly selected registry JSON file."""

    registry_path = resolve_case_path(path)
    try:
        document = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CandidateRegistryError(f"cannot load candidate registry: {registry_path}") from error
    return validate_candidate_registry(document)


def _registry(document: Mapping[str, Any]) -> dict[str, Any]:
    if document is None:  # type: ignore[comparison-overlap]
        raise CandidateRegistryError("an explicit candidate registry document is required")
    return validate_candidate_registry(dict(document))


def candidate_sets(document: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Derive ordered L11, L22, LX-only, and L31 identities."""

    materialized = [item for item in _registry(document)["candidates"] if item["materialize"]]
    result = {
        "L11": tuple(item["label"] for item in materialized if "L11" in item["evidence"]),
        "L22": tuple(item["label"] for item in materialized if "L22" in item["evidence"]),
        "LX_ONLY": tuple(item["label"] for item in materialized if "LX" in item["evidence"]),
        "L31": tuple(
            item["label"]
            for item in materialized
            if set(item["evidence"]) & {"L22", "LX"}
        ),
    }
    if not set(result["L11"]) <= set(result["L31"]):
        raise CandidateRegistryError("L11 must be contained in L31")
    return result


def candidate_labels(
    document: Mapping[str, Any], layer: str = "L31"
) -> tuple[str, ...]:
    key = "LX_ONLY" if layer == "LX" else layer
    try:
        return candidate_sets(document)[key]
    except KeyError as error:
        raise CandidateRegistryError(f"unknown candidate layer: {layer!r}") from error


def candidate_definitions(
    document: Mapping[str, Any], layer: str = "L31"
) -> tuple[dict[str, Any], ...]:
    registry = _registry(document)
    selected = set(candidate_labels(registry, layer))
    return tuple(
        deepcopy(item) for item in registry["candidates"] if item["label"] in selected
    )


def materialized_candidates(
    document: Mapping[str, Any], layer: str = "L31"
) -> tuple[dict[str, Any], ...]:
    return candidate_definitions(document, layer)


def materialized_labels(
    document: Mapping[str, Any], layer: str = "L31"
) -> tuple[str, ...]:
    return candidate_labels(document, layer)


def qa_only_candidates(document: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(
        deepcopy(item) for item in _registry(document)["candidates"] if item["qa_only"]
    )


def expand_materialized_realizations(
    document: Mapping[str, Any], layer: str = "L31"
) -> tuple[tuple[str, int | None], ...]:
    expanded: list[tuple[str, int | None]] = []
    for item in candidate_definitions(document, layer):
        expanded.extend(
            (item["label"], seed) for seed in item["seed_policy"]["seeds"] or [None]
        )
    return tuple(expanded)


def canonicalize_label(document: Mapping[str, Any], label: str) -> str:
    registry = _registry(document)
    canonical = registry["aliases"].get(label, label)
    if canonical not in {item["label"] for item in registry["candidates"]}:
        raise CandidateRegistryError(f"unknown candidate label: {label!r}")
    return canonical


def task_eligible_labels(
    document: Mapping[str, Any], task: str, layer: str = "L31"
) -> tuple[str, ...]:
    if task not in TASKS:
        raise CandidateRegistryError(f"unknown task: {task!r}")
    return tuple(
        item["label"]
        for item in candidate_definitions(document, layer)
        if task in item["task_eligibility"]
    )


def training_plan(
    document: Mapping[str, Any], layer: str = "L31"
) -> tuple[tuple[str, str], ...]:
    registry = _registry(document)
    plan: list[tuple[str, str]] = []
    for item in candidate_definitions(registry, layer):
        if item["training_action"] != "train":
            continue
        config = registry["training_configs"][item["training_config"]]
        identity = (config["family"], config["config"])
        if identity not in plan:
            plan.append(identity)
    return tuple(plan)


def validate_active_candidate_task_list(
    document: Mapping[str, Any],
    labels: Iterable[str],
    *,
    expected_labels: Iterable[str] | None = None,
    layer: str = "L31",
    task: str | None = None,
) -> tuple[str, ...]:
    registry = _registry(document)
    normalized = tuple(labels)
    if len(normalized) != len(set(normalized)):
        raise CandidateRegistryError("candidate task list contains duplicates")
    if set(normalized) & set(registry["aliases"]):
        raise CandidateRegistryError("formal task lists must use canonical labels")
    allowed = set(candidate_labels(registry, layer))
    if task is not None:
        allowed &= set(task_eligible_labels(registry, task, layer))
    unknown = sorted(set(normalized) - allowed)
    if unknown:
        raise CandidateRegistryError(f"unregistered or ineligible labels: {unknown}")
    if expected_labels is not None and normalized != tuple(expected_labels):
        raise CandidateRegistryError("candidate task list differs from derived contract")
    return normalized


__all__ = [
    "CandidateRegistryError",
    "EVIDENCE_LAYERS",
    "LEARNED_SEEDS",
    "SCHEMA_VERSION",
    "TASKS",
    "candidate_definitions",
    "candidate_labels",
    "candidate_sets",
    "canonicalize_label",
    "expand_materialized_realizations",
    "load_candidate_registry",
    "materialized_candidates",
    "materialized_labels",
    "qa_only_candidates",
    "task_eligible_labels",
    "training_plan",
    "validate_active_candidate_task_list",
    "validate_candidate_registry",
]
