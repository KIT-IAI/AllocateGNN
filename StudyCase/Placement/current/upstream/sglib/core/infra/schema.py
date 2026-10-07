"""Mechanism-only validators for table, NPZ, and JSON artifact contracts."""

from __future__ import annotations

import json
from pathlib import Path
import tomllib
from typing import Any, Mapping

import numpy as np
import pandas as pd


class SchemaValidationError(ValueError):
    pass


def load_schema(path: Path | str) -> dict[str, Any]:
    source = Path(path)
    with source.open("rb") as handle:
        schema = tomllib.load(handle)
    if schema.get("schema_version") != "sg_artifact_schema_v1":
        raise SchemaValidationError(f"unsupported schema_version: {source}")
    if schema.get("artifact_type") not in {"table", "npz", "json"}:
        raise SchemaValidationError(f"invalid artifact_type: {source}")
    return schema


def _load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".geoparquet"}:
        return pd.read_parquet(path)
    if suffix in {".gpkg", ".geojson", ".shp"}:
        import geopandas as gpd

        return gpd.read_file(path)
    raise SchemaValidationError(f"unsupported table extension: {path}")


def validate_table(table: pd.DataFrame, schema: Mapping[str, Any]) -> None:
    table_schema = schema.get("table", {})
    if "exact_rows" in table_schema and len(table) != int(table_schema["exact_rows"]):
        raise SchemaValidationError(
            f"table has {len(table)} rows, expected exactly {int(table_schema['exact_rows'])}"
        )
    if "min_rows" in table_schema and len(table) < int(table_schema["min_rows"]):
        raise SchemaValidationError(
            f"table has {len(table)} rows, below minimum {int(table_schema['min_rows'])}"
        )
    if "max_rows" in table_schema and len(table) > int(table_schema["max_rows"]):
        raise SchemaValidationError(
            f"table has {len(table)} rows, above maximum {int(table_schema['max_rows'])}"
        )
    columns = table_schema.get("columns", [])
    if not isinstance(columns, list):
        raise SchemaValidationError("table.columns must be an array of tables")
    required_names = [str(spec["name"]) for spec in columns if spec.get("required", True)]
    missing = [name for name in required_names if name not in table.columns]
    if missing:
        raise SchemaValidationError(f"missing table columns: {missing}")
    if not table_schema.get("allow_extra_columns", False):
        allowed = {str(spec["name"]) for spec in columns}
        extra = set(map(str, table.columns)) - allowed
        if extra:
            raise SchemaValidationError(f"unexpected table columns: {sorted(extra)}")
    for spec in columns:
        name = str(spec["name"])
        if name not in table.columns:
            continue
        series = table[name]
        if not spec.get("nullable", True) and series.isna().any():
            raise SchemaValidationError(f"column {name} contains null values")
        expected = spec.get("dtype")
        if expected == "string" and not (
            pd.api.types.is_string_dtype(series.dtype) or series.dropna().map(type).eq(str).all()
        ):
            raise SchemaValidationError(f"column {name} is not string-like")
        if expected == "integer" and not pd.api.types.is_integer_dtype(series.dtype):
            raise SchemaValidationError(f"column {name} is not integer")
        if expected == "number" and not pd.api.types.is_numeric_dtype(series.dtype):
            raise SchemaValidationError(f"column {name} is not numeric")
        if expected == "boolean" and not pd.api.types.is_bool_dtype(series.dtype):
            raise SchemaValidationError(f"column {name} is not boolean")
        if "minimum" in spec:
            values = pd.to_numeric(series.dropna(), errors="raise")
            if (values < float(spec["minimum"])).any():
                raise SchemaValidationError(
                    f"column {name} contains values below {spec['minimum']}"
                )
        if "maximum" in spec:
            values = pd.to_numeric(series.dropna(), errors="raise")
            if (values > float(spec["maximum"])).any():
                raise SchemaValidationError(
                    f"column {name} contains values above {spec['maximum']}"
                )
        allowed_values = spec.get("allowed_values")
        if allowed_values is not None:
            invalid = set(series.dropna().unique()) - set(allowed_values)
            if invalid:
                raise SchemaValidationError(f"column {name} has invalid values: {sorted(map(str, invalid))}")
    for key in table_schema.get("unique_by", []):
        subset = [key] if isinstance(key, str) else list(key)
        if table.duplicated(subset=subset).any():
            raise SchemaValidationError(f"table identity is not unique by {subset}")


def _shape_matches(shape: tuple[int, ...], expected: list[Any], symbols: dict[str, int]) -> bool:
    if len(shape) != len(expected):
        return False
    for actual, requirement in zip(shape, expected, strict=True):
        if isinstance(requirement, int) and actual != requirement:
            return False
        if isinstance(requirement, str):
            previous = symbols.setdefault(requirement, actual)
            if previous != actual:
                return False
    return True


def validate_npz(path: Path, schema: Mapping[str, Any]) -> None:
    specs = schema.get("npz", {}).get("arrays", {})
    if not isinstance(specs, dict):
        raise SchemaValidationError("npz.arrays must be a table")
    with np.load(path, allow_pickle=False) as archive:
        actual_keys = set(archive.files)
        required = {key for key, spec in specs.items() if spec.get("required", True)}
        missing = required - actual_keys
        if missing:
            raise SchemaValidationError(f"missing NPZ arrays: {sorted(missing)}")
        if not schema.get("npz", {}).get("allow_extra_arrays", False):
            extra = actual_keys - set(specs)
            if extra:
                raise SchemaValidationError(f"unexpected NPZ arrays: {sorted(extra)}")
        symbols: dict[str, int] = {}
        for key, spec in specs.items():
            if key not in archive:
                continue
            value = archive[key]
            if "dtype_kind" in spec and value.dtype.kind not in str(spec["dtype_kind"]):
                raise SchemaValidationError(f"NPZ array {key} has dtype {value.dtype}")
            if "shape" in spec and not _shape_matches(value.shape, list(spec["shape"]), symbols):
                raise SchemaValidationError(f"NPZ array {key} has shape {value.shape}")
            if spec.get("finite", False) and not np.isfinite(value).all():
                raise SchemaValidationError(f"NPZ array {key} contains non-finite values")
            if "allowed_values" in spec:
                invalid = set(np.unique(value).tolist()) - set(spec["allowed_values"])
                if invalid:
                    raise SchemaValidationError(f"NPZ array {key} has invalid values: {invalid}")


def _resolve_json_path(document: Any, dotted: str) -> list[Any]:
    values = [document]
    for part in dotted.split("."):
        next_values: list[Any] = []
        for value in values:
            if part == "*" and isinstance(value, list):
                next_values.extend(value)
            elif isinstance(value, Mapping) and part in value:
                next_values.append(value[part])
        values = next_values
    return values


def validate_json(document: Any, schema: Mapping[str, Any]) -> None:
    json_schema = schema.get("json", {})
    for required_path in json_schema.get("required_paths", []):
        if not _resolve_json_path(document, str(required_path)):
            raise SchemaValidationError(f"missing JSON path: {required_path}")
    for identity in json_schema.get("unique_by", []):
        values = _resolve_json_path(document, str(identity))
        frozen = [json.dumps(value, sort_keys=True, ensure_ascii=False) for value in values]
        if len(frozen) != len(set(frozen)):
            raise SchemaValidationError(f"JSON identity is not unique: {identity}")


def validate_artifact(path: Path | str, schema_path: Path | str) -> Path:
    artifact = Path(path)
    if not artifact.is_file():
        raise SchemaValidationError(f"artifact does not exist: {artifact}")
    schema = load_schema(schema_path)
    kind = schema["artifact_type"]
    if kind == "table":
        validate_table(_load_table(artifact), schema)
    elif kind == "npz":
        validate_npz(artifact, schema)
    else:
        with artifact.open("r", encoding="utf-8") as handle:
            validate_json(json.load(handle), schema)
    return artifact
