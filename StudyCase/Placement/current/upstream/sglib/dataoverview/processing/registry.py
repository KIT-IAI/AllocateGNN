"""Build and validate the country-neutral DataOverview unit DAG from TOML."""

from __future__ import annotations

from collections import defaultdict, deque
import os
from pathlib import Path
from typing import Any, Iterable
import json

import geopandas as gpd

from sglib.core.infra.schema import SchemaValidationError, validate_artifact

from .download import run_static_dataset
from .queries import PROTOCOLS
from .unit import Unit, UnitContext


class RegistryError(ValueError):
    pass


def _source_outputs(repo_root: Path, country: str, spec: dict[str, Any]) -> tuple[Path, ...]:
    raw = repo_root / "data" / "datasets" / "1_raw" / country
    outputs: list[Path] = []
    for file_spec in spec.get("files", []):
        if file_spec.get("kind") == "zip_extract":
            probe = str(file_spec["zip_probe"]).split("/", 1)[-1]
            outputs.append(raw / str(file_spec["extract_root"]) / probe)
        else:
            outputs.append((raw / str(file_spec["filename"])).resolve())
    query = spec.get("query", {})
    if "filename" in query:
        query_target = raw / str(query["filename"])
        outputs.append(query_target)
        if spec.get("query_protocol") == "ogcapi":
            outputs.append(
                query_target.with_suffix(query_target.suffix + ".pagination.json")
            )
    if "output" in query:
        query_target = (raw / str(query["output"])).resolve()
        outputs.append(query_target)
        if spec.get("query_protocol") == "gee":
            outputs.append(query_target.parent / "acquisition_receipt.json")
    if spec.get("query_protocol") == "geoserver":
        outputs.extend(raw / "geoserver" / f"{str(layer).replace(':', '_')}.geojson" for layer in query.get("layers", []))
    return tuple(outputs)


def gee_query(config: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
    """Merge the country ``[ntl]`` block with the dataset query and fill GEE defaults.

    The raster bounds default to the canonical regions table and the output
    CRS to the country's native CRS, so a country overlay only has to override
    them when it deliberately wants something else (NL/NZ do).
    """

    query = {**dict(config["ntl"]), **dict(dataset.get("query", {}))}
    query.setdefault("bounds_source", str(config["canonical"]["regions"]))
    query.setdefault("output_crs", str(config["crs"]["native"]))
    return query


def _producer_of(repo_root: Path, country: str, config: dict[str, Any], relative_path: str) -> str | None:
    """Return the ``<country>.<product>.derive`` unit that produces ``relative_path``."""

    wanted = (repo_root / str(relative_path)).resolve()
    for product_id, spec in config.get("products", {}).items():
        produced = {(repo_root / str(path)).resolve() for path in spec.get("produces", [])}
        if wanted in produced or any(parent in produced for parent in wanted.parents):
            return f"{country}.{product_id}.derive"
    return None


def source_dependencies(repo_root: Path, country: str, config: dict[str, Any], spec: dict[str, Any]) -> tuple[str, ...]:
    """Dependencies of a download unit: explicit ``depends_on`` or the producer of GEE bounds."""

    explicit = tuple(map(str, spec.get("depends_on", [])))
    if explicit or spec.get("query_protocol") != "gee":
        return explicit
    producer = _producer_of(repo_root, country, config, gee_query(config, spec)["bounds_source"])
    return (producer,) if producer else ()


def _run_query(context: UnitContext, unit: Unit) -> None:
    if unit.country is None:
        raise RegistryError("query unit has no country")
    dataset = context.config["datasets"][unit.id.split(".")[1]]
    protocol = str(dataset["query_protocol"])
    try:
        runner = PROTOCOLS[protocol]
    except KeyError as exc:
        raise RegistryError(f"unknown query protocol: {protocol}") from exc
    target = context.repo_root / "data" / "datasets" / "1_raw" / unit.country
    query = dict(dataset.get("query", {}))
    if protocol == "gee":
        query = gee_query(dict(context.config), dataset)
    runner(query, target, refresh=context.refresh)


def _transform_runner(product: str):
    from . import transforms

    return transforms.runner_for(product)


def _grid_done(repo_root: Path, country: str, config: dict[str, Any]) -> bool:
    root = repo_root / "data" / "datasets" / "2_derived" / country / "grid_bplus" / "bundles"
    items = config["regions"]["items"]
    if not items:
        return False
    for item in items:
        metadata_path = root / str(item["id"]) / "grid_metadata.json"
        points_path = root / str(item["id"]) / "grid_points.parquet"
        if not metadata_path.is_file() or not points_path.is_file():
            return False
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        if metadata.get("schema_version") != "sg_grid_bundle_v2":
            return False
    return True


def _features_done(repo_root: Path, country: str, config: dict[str, Any]) -> bool:
    root = repo_root / "data" / "datasets" / "2_derived" / country / "features_bplus"
    receipt = root / "features_receipt.json"
    if not receipt.is_file():
        return False
    try:
        document = json.loads(receipt.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if document.get("schema_version") != "sg_dataoverview_features_receipt_v1":
        return False
    recorded = document.get("grid_metadata_sha256", {})
    row_identity = document.get("grid_row_identity_sha256", {})
    from sglib.core.infra.hashing import sha256_file
    from sglib.core.infra.hashing import sha256_json

    items = config["regions"]["items"]
    if not items:
        return False
    for item in items:
        name = str(item["id"])
        metadata = repo_root / "data" / "datasets" / "2_derived" / country / "grid_bplus" / "bundles" / name / "grid_metadata.json"
        if not metadata.is_file() or recorded.get(name) != sha256_file(metadata):
            return False
        points = metadata.with_name("grid_points.parquet")
        if not points.is_file():
            return False
        frame = gpd.read_parquet(points)
        source_key = str(config["regions"]["source_key"])
        if row_identity.get(name) != sha256_json(frame[source_key].astype(str).tolist()):
            return False
    extracted = root / "extracted"
    return all(
        (extracted / f"{item['id']}_{suffix}.npz").is_file()
        for item in items
        for suffix in ("landuse", "ghsl_built_s", "cuz_support", "ntl")
    )


def _output_present(path: Path) -> bool:
    if path.is_file():
        return path.stat().st_size > 0
    return path.is_dir() and any(item.is_file() for item in path.rglob("*"))


def _canonical_done(
    repo_root: Path,
    country: str,
    config: dict[str, Any],
    produces: tuple[Path, ...],
) -> bool:
    if not all(_output_present(path) for path in produces):
        return False
    canonical = config.get("canonical")
    station_contract = config.get("station_contract")
    if not isinstance(canonical, dict) or not isinstance(station_contract, dict):
        return False
    produced = {path.resolve() for path in produces}
    try:
        region_path = (repo_root / str(canonical["regions"])).resolve()
        station_path = (repo_root / str(canonical["stations"])).resolve()
        for path in (region_path, station_path):
            path.relative_to(repo_root.resolve())
    except (KeyError, ValueError):
        return False
    try:
        if region_path in produced:
            columns = set(gpd.read_file(region_path, rows=1).columns)
            required = {str(station_contract["source_key"])}
            region_demand = str(
                station_contract.get("region_demand_column", "")
            ).strip()
            if region_demand:
                required.add(region_demand)
            if not required <= columns:
                return False
        if station_path in produced:
            columns = set(gpd.read_file(station_path, rows=1).columns)
            required = {
                str(station_contract["id_column"]),
                str(station_contract["demand_column"]),
                str(station_contract["region_column"]),
                "capacity_basis",
            }
            capacity = str(station_contract.get("capacity_column", "")).strip()
            if capacity:
                required.add(capacity)
            for optional in (
                "security_class_column",
                "noncoincident_flag_column",
            ):
                value = str(station_contract.get(optional, "")).strip()
                if value:
                    required.add(value)
            if not required <= columns:
                return False
    except (KeyError, OSError, ValueError):
        return False
    return True


def _evidence_done(
    repo_root: Path,
    config: dict[str, Any],
    produces: tuple[Path, ...],
) -> bool:
    configured = config.get("handoff_artifacts", {})
    if not isinstance(configured, dict):
        return False
    produced = {path.resolve() for path in produces}
    matched = False
    schemas = repo_root / "casestudy" / "1_DataOverview" / "schemas"
    for raw_spec in configured.values():
        spec = {"path": raw_spec} if isinstance(raw_spec, str) else raw_spec
        if not isinstance(spec, dict) or "path" not in spec:
            return False
        path = (repo_root / str(spec["path"])).resolve()
        if path not in produced:
            continue
        matched = True
        if not _output_present(path):
            return False
        try:
            if spec.get("schema"):
                validate_artifact(path, schemas / str(spec["schema"]))
            if spec.get("formal_required") and spec.get("required_status"):
                if path.suffix.lower() != ".json":
                    return False
                document = json.loads(path.read_text(encoding="utf-8"))
                if document.get("status") != str(spec["required_status"]):
                    return False
        except (OSError, UnicodeError, json.JSONDecodeError, SchemaValidationError):
            return False
    return matched


def build_registry(repo_root: Path, country_configs: dict[str, dict[str, Any]]) -> dict[str, Unit]:
    units: dict[str, Unit] = {}
    for country, config in sorted(country_configs.items()):
        for dataset_id, spec in config["datasets"].items():
            unit_id = f"{country}.{dataset_id}.download"
            protocol = spec.get("query_protocol")
            credential = spec.get("query", {}).get("credential_env")
            units[unit_id] = Unit(
                id=unit_id,
                kind="source",
                country=country,
                category=str(spec["category"]),
                phase="processing",
                step="download",
                # GEE queries clip to a derived regions table, so they depend on
                # the unit producing it; static files never depend on anything.
                depends_on=source_dependencies(repo_root, country, config, spec),
                produces=_source_outputs(repo_root, country, spec),
                run=_run_query if protocol else run_static_dataset,
                credential_env=str(credential) if credential else None,
            )
        for product_id, spec in config["products"].items():
            overview = product_id == "inventory"
            step = "overview" if overview else "derive"
            unit_id = f"{country}.{product_id}.{step}"
            produces = tuple(repo_root / str(path) for path in spec.get("produces", []))
            if product_id == "grid":
                done_check = lambda r=repo_root, c=country, cfg=config: _grid_done(r, c, cfg)
            elif product_id == "features":
                done_check = lambda r=repo_root, c=country, cfg=config: _features_done(r, c, cfg)
            else:
                produced_set = {item.resolve() for item in produces}
                canonical_match = any(
                    (repo_root / str(path)).resolve() in produced_set
                    for path in config.get("canonical", {}).values()
                )
                evidence_match = any(
                    (
                        repo_root
                        / str(
                            raw_spec.get("path")
                            if isinstance(raw_spec, dict)
                            else raw_spec
                        )
                    ).resolve()
                    in produced_set
                    for raw_spec in config.get("handoff_artifacts", {}).values()
                )
                if canonical_match or evidence_match:
                    done_check = lambda r=repo_root, c=country, cfg=config, p=produces, canonical=canonical_match, evidence=evidence_match: (
                        (not canonical or _canonical_done(r, c, cfg, p))
                        and (not evidence or _evidence_done(r, cfg, p))
                    )
                else:
                    done_check = None
            units[unit_id] = Unit(
                id=unit_id,
                kind="transform",
                country=country,
                category=str(spec["category"]),
                phase="overview" if overview else "processing",
                step=step,
                depends_on=tuple(map(str, spec.get("depends_on", []))),
                produces=produces,
                run=_transform_runner(product_id),
                done_check=done_check,
            )
    # The matrix is rendered from the TOML ledgers alone, so it carries no
    # dependency on country inventories and can be regenerated at any time.
    units["general.matrix.overview"] = Unit(
        id="general.matrix.overview",
        kind="transform",
        country=None,
        category="features_cuz",
        phase="overview",
        step="overview",
        depends_on=(),
        produces=(repo_root / "casestudy" / "1_DataOverview" / "general" / "data_matrix.md",),
        run=_transform_runner("matrix"),
    )
    validate_registry(units)
    return units


def validate_registry(units: dict[str, Unit]) -> None:
    for unit in units.values():
        missing = set(unit.depends_on) - set(units)
        if missing:
            raise RegistryError(f"{unit.id} has missing dependencies: {sorted(missing)}")
        for dependency in unit.depends_on:
            if unit.phase == "processing" and units[dependency].phase == "overview":
                raise RegistryError(f"processing may not depend on overview: {unit.id}")
    topological_order(units)


def topological_order(units: dict[str, Unit], selected: Iterable[str] | None = None) -> list[Unit]:
    wanted = set(selected or units)
    stack = list(wanted)
    while stack:
        current = stack.pop()
        for dependency in units[current].depends_on:
            if dependency not in wanted:
                wanted.add(dependency)
                stack.append(dependency)
    indegree = {key: 0 for key in wanted}
    downstream: dict[str, list[str]] = defaultdict(list)
    for key in wanted:
        for dependency in units[key].depends_on:
            if dependency in wanted:
                indegree[key] += 1
                downstream[dependency].append(key)
    ready = deque(sorted(key for key, value in indegree.items() if value == 0))
    ordered: list[Unit] = []
    while ready:
        key = ready.popleft()
        ordered.append(units[key])
        for child in sorted(downstream[key]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(ordered) != len(wanted):
        raise RegistryError("unit dependency graph contains a cycle")
    return ordered


def _needs_credential(unit: Unit) -> bool:
    """A query unit needs its credential only when a network fetch is unavoidable.

    The GEE runner rebuilds ``acquisition_receipt.json`` from a raster that is
    already on disk without contacting the service, so a missing receipt alone
    does not block the unit.
    """

    return any(
        not Path(path).exists()
        for path in unit.produces
        if Path(path).name != "acquisition_receipt.json"
    )


def status(unit: Unit) -> str:
    if unit.done():
        return "DONE"
    if unit.credential_env and not os.environ.get(unit.credential_env) and _needs_credential(unit):
        return f"BLOCKED(credential:{unit.credential_env})"
    return "PENDING"
