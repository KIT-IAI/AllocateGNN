from __future__ import annotations

from functools import partial
from sglib.core.chain.stage import topological_order as _chain_topological_order

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class GeneratorUnit:
    id: str
    step: str
    country: str
    member: str
    depends_on: tuple[str, ...]


class GeneratorRegistryError(ValueError):
    pass


def build_registry(
    country: str,
    config: Mapping[str, Any],
    candidate_registry: Mapping[str, Any],
    group_families: Mapping[str, str] | None = None,
) -> dict[str, GeneratorUnit]:
    prefix = str(country)
    units: dict[str, GeneratorUnit] = {}
    execution = config.get("execution", {})
    if not isinstance(execution, Mapping):
        raise GeneratorRegistryError("Generator execution configuration must be a mapping")
    civd_enabled = execution.get("civd_enabled", True)
    if type(civd_enabled) is not bool:
        raise GeneratorRegistryError("execution.civd_enabled must be boolean")

    def add(step: str, member: str, dependencies: Iterable[str]) -> str:
        key = f"{prefix}.{step}.{member}"
        if key in units:
            raise GeneratorRegistryError(f"duplicate Generator unit: {key}")
        units[key] = GeneratorUnit(key, step, prefix, member, tuple(dependencies))
        return key

    inputs = add("inputs", "bundle", ())
    assignments = add("static", "assignments", (inputs,))
    uniform = add("static", "uniform", (inputs,))
    gpm = add("static", "gpm", (inputs,))
    proximity = add("static", "proximity", (inputs,))
    public = add("public_activity", "public_activity", (gpm,))
    groups = list(map(str, config["task_groups"]))
    verify_units = []
    infer_units = []
    for group in groups:
        train = add("train", group, (inputs,))
        verify = add("verify", group, (train,))
        infer = add("infer", group, (verify,))
        verify_units.append(verify)
        infer_units.append(infer)
    families = tuple(dict.fromkeys(str(item["family"]) for item in candidate_registry["candidates"]))
    materialized = []
    materialized_by_family = {}
    for family in families:
        if family in {"Uni", "GPM", "Equal"}:
            dependencies = (uniform, gpm, assignments)
        else:
            dependencies = tuple(
                unit
                for unit in infer_units
                if unit.rsplit(".", 1)[-1] in (
                    {f"B-{prefix.upper()}-MLP"} if family == "MLP" else
                    {f"B-{prefix.upper()}-GNN", *(
                        f"{kind}-{prefix.upper()}-{signal}"
                        for kind in ("P", "F") for signal in ("N", "P", "NP")
                    )}
                )
            ) + (gpm,)
        unit = add("materialize", family, dependencies)
        materialized.append(unit)
        materialized_by_family[family] = unit
    finalized = add("finalize", "candidate_index", materialized)
    sweep_units = []
    for name in config["sweeps"]:
        if name == "lambda":
            dependencies = tuple(
                f"{prefix}.infer.{kind}-{prefix.upper()}-{signal}"
                for kind in ("L", "P") for signal in ("N", "P")
            ) + (inputs,)
        elif name == "tau":
            dependencies = (f"{prefix}.infer.T-{prefix.upper()}",
                            f"{prefix}.infer.B-{prefix.upper()}-GNN", inputs)
        else:
            dependencies = (finalized, gpm, inputs)
        sweep_units.append(add("sweeps", name, dependencies))
    audit_dependencies = [assignments, uniform, gpm, proximity, public, finalized, *materialized, *sweep_units]
    if civd_enabled:
        audit_dependencies.append(add("civd", "civd", (inputs,)))
    fixed = add("idr_fixed", "idr_fixed", (public, assignments))
    matched = add("idr_matched", "idr_matched", (finalized, assignments))
    audit_dependencies.extend((fixed, matched, *verify_units, *infer_units))
    add("audit", "audit", audit_dependencies)
    topological_order(units)
    return units


topological_order = partial(_chain_topological_order, label='Generator', method='breadth', error=GeneratorRegistryError)


def unit_done(unit: GeneratorUnit, root: Path) -> bool:
    return unit_status(unit, root) == "DONE"


def unit_status(unit: GeneratorUnit, root: Path) -> str:
    """Read marker contracts only; the final audit verifies artifact contents."""

    path = unit_output_path(unit, root)
    schemas = {
        "inputs": {"sg_generator_inputs_receipt_v1", "sg_generator_inputs_receipt_v2", "sg_generator_inputs_receipt_v3"},
        "static": {"sg_generator_static_index_v1"},
        "public_activity": {"sg_generator_static_index_v1"},
        "train": {"sg_training_group_receipt_v1"},
        "verify": {"sg_training_verify_v1"},
        "infer": {"sg_inference_group_receipt_v1", "sg_inference_group_view_v2"},
        "materialize": {"sg_candidate_family_index_v1"},
        "finalize": {"sg_candidate_index_v1"},
        "sweeps": {"sg_sweep_index_v1"},
        "civd": {"sg_civd_index_v1", "sg_civd_extension_index_v1"},
        "idr_fixed": {"sg_idr_fixed_index_v1"},
        "idr_matched": {"sg_idr_matched_index_v1"},
        "audit": {"sg_generator_audit_v1"},
    }

    def valid(marker: Path, allowed: set[str]):
        try:
            document = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            return None
        return document if isinstance(document, dict) and document.get("schema_version") in allowed else None

    document = valid(path, schemas[unit.step]) if path.exists() else None
    if path.exists() and document is None:
        return "INVALID"
    if document is not None:
        if unit.step == "train" and type(document.get("complete")) is not bool:
            return "INVALID"
        if unit.step == "audit" and document.get("status") != "PASS":
            return "INVALID"
        if unit.step == "train" and document.get("complete") is not True:
            pass
        elif unit.step == "infer":
            verification = root / "inference/verify" / f"{unit.member}.json"
            if verification.exists():
                return "DONE" if valid(verification, {"sg_inference_verify_v1"}) else "INVALID"
        else:
            return "DONE"
    if unit.step in {"train", "infer"}:
        kind = "training" if unit.step == "train" else "inference"
        index = root / kind / "tasks" / unit.member / "index.json"
        if index.exists():
            prepared = valid(index, {f"sg_prepared_{kind}_group_v1"})
            return "PREPARED" if prepared and prepared.get("tasks") else "INVALID"
    return "PENDING"


def unit_output_path(unit: GeneratorUnit, root: Path) -> Path:
    """Return the single contract marker for a Generator DAG unit.

    Site shells use this same mapping rather than maintaining a second view of
    the scientific DAG's completion contracts.
    """

    paths = {
        ("inputs", "bundle"): root / "inputs/receipt.json",
        ("static", unit.member): root / "static" / unit.member / "index.json",
        ("public_activity", "public_activity"): root / "static/public_activity/index.json",
        ("train", unit.member): root / "training/receipts" / f"{unit.member}.json",
        ("verify", unit.member): root / "training/verify" / f"{unit.member}.json",
        ("infer", unit.member): root / "inference/receipts" / f"{unit.member}.json",
        ("materialize", unit.member): root / "candidates" / f"index_{unit.member}.json",
        ("finalize", "candidate_index"): root / "candidates/candidate_index.json",
        ("sweeps", unit.member): root / "sweeps" / f"index_{unit.member}.json",
        ("civd", "civd"): root / "civd/index.json",
        ("idr_fixed", "idr_fixed"): root / "idr_fixed/index.json",
        ("idr_matched", "idr_matched"): root / "idr_matched/index.json",
        ("audit", "audit"): root / "audit.json",
    }
    path = paths.get((unit.step, unit.member))
    if path is None:
        raise GeneratorRegistryError(f"no output contract for Generator unit {unit.id}")
    return path
