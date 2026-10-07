"""Scientific matrix loading, coordinate expansion, and validation."""

from __future__ import annotations

import csv
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from ..paths import safe_relative_path
from .tasks import (COUNTRY_PATTERN, MATRIX_SCHEMA, CONFIG_MAP, EXPECTED_GROUPS, EXPECTED_LOGICAL_COORDINATES,
    EXPECTED_REUSE_COORDINATES, EXPECTED_TRAIN_COORDINATES, GROUPS_PER_COUNTRY,
    LOGICAL_COORDINATES_PER_COUNTRY, REUSE_COORDINATES_PER_COUNTRY,
    TRAIN_COORDINATES_PER_COUNTRY, SUPPORTED_AUTHORITY_COUNTRY_COUNTS,
    TaskGroup, TrainingTask, TaskMatrixSummary, TaskMatrixError,
    _REQUIRED_COLUMNS, _LEGACY_OPERATIONAL_COLUMNS)


def load_task_matrix(path: str | Path) -> tuple[TaskGroup, ...]:
    """Load the one explicitly selected task matrix."""

    matrix_path = Path(path)
    try:
        with matrix_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise TaskMatrixError("training matrix header is incomplete")
            fields = set(reader.fieldnames)
            legacy_fields = (_REQUIRED_COLUMNS - {"compute_threads"}) | {
                "cpus",
            } | _LEGACY_OPERATIONAL_COLUMNS
            if _REQUIRED_COLUMNS <= fields:
                groups = tuple(TaskGroup.from_row(row) for row in reader)
            elif legacy_fields <= fields:
                rows = []
                for row in reader:
                    projected = {
                        name: row[name]
                        for name in _REQUIRED_COLUMNS
                        if name != "compute_threads"
                    }
                    projected["compute_threads"] = row["cpus"]
                    rows.append(projected)
                groups = tuple(TaskGroup.from_row(row) for row in rows)
            else:
                raise TaskMatrixError("training matrix header is incomplete")
    except (OSError, UnicodeError, csv.Error) as error:
        raise TaskMatrixError(f"cannot load task matrix: {matrix_path}") from error
    names = [group.task_group for group in groups]
    if len(names) != len(set(names)):
        raise TaskMatrixError("task_group values must be unique")
    return groups



def _select_group(
    matrix: Sequence[TaskGroup], *, country: str, group: str
) -> TaskGroup:
    key = str(country).lower()
    if COUNTRY_PATTERN.fullmatch(key) is None:
        raise TaskMatrixError(f"invalid two-letter country code: {country!r}")
    matches = [item for item in matrix if item.country == key and item.task_group == group]
    if len(matches) != 1:
        raise TaskMatrixError(
            f"task group must resolve exactly once for country={key}: {group!r}"
        )
    return matches[0]



def _is_reuse(group: TaskGroup, value: str) -> bool:
    if group.stage == "lambda_sweep":
        return float(value) == 0.05
    if group.stage == "tau_sweep":
        return float(value) == 0.01
    return False



def _reuse_source(
    matrix: Sequence[TaskGroup], group: TaskGroup, *, seed: int, fold: int
) -> tuple[str, str]:
    if group.stage == "lambda_sweep":
        source_group_name = f"P-{group.country.upper()}-{group.signal}"
        source_fold = 1
    elif group.stage == "tau_sweep":
        source_group_name = f"B-{group.country.upper()}-GNN"
        source_fold = fold
    else:
        raise TaskMatrixError(f"{group.task_group}: non-sweep coordinate cannot be reused")
    source_group = _select_group(
        matrix, country=group.country, group=source_group_name
    )
    output = source_group.output_pattern.format(seed=seed, fold=source_fold, value="default")
    source_id = f"{source_group_name}-S{seed}-F{source_fold}"
    return source_id, safe_relative_path(output)



def logical_tasks_for_group(
    matrix: Sequence[TaskGroup], *, country: str, group: str
) -> tuple[TrainingTask, ...]:
    """Expand all planned coordinates, including direct identity reuse."""

    record = _select_group(matrix, country=country, group=group)
    tasks: list[TrainingTask] = []
    for value in record.parameter_values:
        for seed in record.seeds:
            for fold in record.folds:
                reuse = _is_reuse(record, value)
                source_id = source_output = None
                if reuse:
                    source_id, source_output = _reuse_source(
                        matrix, record, seed=seed, fold=fold
                    )
                output = record.output_pattern.format(seed=seed, fold=fold, value=value)
                tasks.append(
                    TrainingTask(
                        group=record.task_group,
                        country=record.country,
                        family=record.family,
                        config=record.package_config,
                        signal=record.signal,
                        parameter=record.parameter_name,
                        value=value,
                        seed=seed,
                        fold=fold,
                        output_relative=safe_relative_path(output),
                        feature_set=record.feature_set,
                        compute_threads=record.compute_threads,
                        reuse=reuse,
                        source_task_id=source_id,
                        source_output_relative=source_output,
                    )
                )
    if len(tasks) != record.logical_coordinates:
        raise TaskMatrixError(f"{record.task_group}: logical expansion count differs")
    if sum(task.reuse for task in tasks) != record.reuse_coordinates:
        raise TaskMatrixError(f"{record.task_group}: reuse expansion count differs")
    return tuple(tasks)



def tasks_for_group(
    matrix: Sequence[TaskGroup], *, country: str, group: str
) -> tuple[TrainingTask, ...]:
    """Return only unique coordinates that require training."""

    record = _select_group(matrix, country=country, group=group)
    tasks = tuple(
        task
        for task in logical_tasks_for_group(matrix, country=country, group=group)
        if not task.reuse
    )
    if len(tasks) != record.new_train_coordinates:
        raise TaskMatrixError(f"{record.task_group}: training expansion count differs")
    return tasks



def all_training_tasks(matrix: Sequence[TaskGroup]) -> tuple[TrainingTask, ...]:
    countries = {record.country for record in matrix}
    if len(countries) not in SUPPORTED_AUTHORITY_COUNTRY_COUNTS:
        raise TaskMatrixError(
            "training matrix authority must contain either 2 transition countries "
            "or 4 formal countries"
        )
    tasks = tuple(
        task
        for record in matrix
        for task in tasks_for_group(
            matrix, country=record.country, group=record.task_group
        )
    )
    expected = TRAIN_COORDINATES_PER_COUNTRY * len(countries)
    if len(tasks) != expected:
        raise TaskMatrixError(
            f"training matrix must expand to {expected} tasks for "
            f"{len(countries)} discovered countries"
        )
    identities = [task.task_id for task in tasks]
    if len(identities) != len(set(identities)):
        raise TaskMatrixError("expanded training task ids are not unique")
    return tasks



def _load_params(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    try:
        return json.loads(Path(value).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise TaskMatrixError(f"cannot load frozen parameters: {value}") from error



def validate_task_matrix(
    matrix: Sequence[TaskGroup],
    *,
    frozen_params_by_country: Mapping[str, Mapping[str, Any] | str | Path],
) -> TaskMatrixSummary:
    """Validate a two-country transition or four-country formal authority."""

    countries = tuple(sorted({group.country for group in matrix}))
    if len(countries) not in SUPPORTED_AUTHORITY_COUNTRY_COUNTS:
        raise TaskMatrixError(
            "training matrix authority must contain either 2 transition countries "
            "or 4 formal countries"
        )
    if len(matrix) != GROUPS_PER_COUNTRY * len(countries):
        raise TaskMatrixError("training matrix must contain exactly 11 groups per discovered country")
    params_by_country = {
        country: _load_params(frozen_params_by_country[country])
        for country in countries
        if country in frozen_params_by_country
    }
    if set(params_by_country) != set(countries):
        raise TaskMatrixError("frozen parameters are required for every matrix country")
    for country in countries:
        validate_country_task_matrix(matrix, country, params_by_country[country])

    logical = sum(group.logical_coordinates for group in matrix)
    reuse = sum(group.reuse_coordinates for group in matrix)
    training = sum(group.new_train_coordinates for group in matrix)
    expected_totals = (
        LOGICAL_COORDINATES_PER_COUNTRY * len(countries),
        REUSE_COORDINATES_PER_COUNTRY * len(countries),
        TRAIN_COORDINATES_PER_COUNTRY * len(countries),
    )
    if (logical, reuse, training) != expected_totals:
        raise TaskMatrixError(f"training matrix totals must be 126/6/120 per country, got {(logical, reuse, training)}")

    logical_tasks = tuple(
        task
        for record in matrix
        for task in logical_tasks_for_group(
            matrix, country=record.country, group=record.task_group
        )
    )
    if len(logical_tasks) != expected_totals[0]:
        raise TaskMatrixError(
            f"logical task expansion differs from {expected_totals[0]}"
        )
    if len({task.task_id for task in logical_tasks}) != len(logical_tasks):
        raise TaskMatrixError("logical task identities are not unique")
    all_training_tasks(matrix)

    return TaskMatrixSummary(
        schema=MATRIX_SCHEMA,
        groups=len(matrix),
        logical_coordinates=logical,
        reuse_coordinates=reuse,
        train_coordinates=training,
        countries=countries,
    )



def validate_country_task_matrix(
    matrix: Sequence[TaskGroup],
    country: str,
    frozen_params: Mapping[str, Any] | str | Path,
) -> TaskMatrixSummary:
    """Validate only one country's 11/126/6/120 task contract.

    This is the validation entry used by country-scoped preparation.  It never
    loads, expands, or inspects the other country's frozen parameters.
    """

    key = str(country).lower()
    if key not in {group.country for group in matrix}:
        raise TaskMatrixError(f"country is absent from task matrix: {country!r}")
    selected = tuple(group for group in matrix if group.country == key)
    if len(selected) != GROUPS_PER_COUNTRY:
        raise TaskMatrixError(f"{key}: training matrix must contain 11 groups")
    totals = (
        sum(group.logical_coordinates for group in selected),
        sum(group.reuse_coordinates for group in selected),
        sum(group.new_train_coordinates for group in selected),
    )
    if totals != (
        LOGICAL_COORDINATES_PER_COUNTRY,
        REUSE_COORDINATES_PER_COUNTRY,
        TRAIN_COORDINATES_PER_COUNTRY,
    ):
        raise TaskMatrixError(f"{key}: task totals must remain 126/6/120")
    params = _load_params(frozen_params)
    config_map = params.get("config_map")
    if not isinstance(config_map, Mapping):
        raise TaskMatrixError(f"{key}: frozen config_map is missing")
    for group in selected:
        if group.package_config not in config_map:
            raise TaskMatrixError(
                f"{group.task_group}: config {group.package_config!r} is not frozen"
            )
        logical_tasks_for_group(matrix, country=key, group=group.task_group)
        tasks_for_group(matrix, country=key, group=group.task_group)
    tasks = tuple(
        task
        for group in selected
        for task in tasks_for_group(matrix, country=key, group=group.task_group)
    )
    if (
        len(tasks) != TRAIN_COORDINATES_PER_COUNTRY
        or len({task.task_id for task in tasks}) != TRAIN_COORDINATES_PER_COUNTRY
    ):
        raise TaskMatrixError(f"{key}: unique training expansion differs from 120")
    for signal, objective_name in (("N", "ntl_prior"), ("P", "proximity_prior")):
        sweep = _select_group(matrix, country=key, group=f"L-{key.upper()}-{signal}")
        objective = config_map[sweep.package_config].get("objective_weights")
        if (
            0.05 not in {float(value) for value in sweep.parameter_values}
            or not isinstance(objective, Mapping)
            or float(objective.get(objective_name, -1)) != 0.05
        ):
            raise TaskMatrixError(
                f"{key}/{signal}: lambda=0.05 does not directly match frozen objective"
            )
        expected = f"P-{key.upper()}-{signal}_lambda0.05_seed42_fold1"
        if sweep.reuse_source != expected:
            raise TaskMatrixError(f"{sweep.task_group}: reuse_source identity differs")
    tau = _select_group(matrix, country=key, group=f"T-{key.upper()}")
    try:
        frozen_tau = float(params["gnn"]["allocation_temperature_start"])
    except (KeyError, TypeError, ValueError) as error:
        raise TaskMatrixError(
            f"{key}: gnn.allocation_temperature_start is missing"
        ) from error
    if (
        0.01 not in {float(value) for value in tau.parameter_values}
        or frozen_tau != 0.01
        or tau.reuse_source != f"B-{key.upper()}-GNN_tau0.01_seed42_allfolds"
    ):
        raise TaskMatrixError(f"{key}: tau=0.01 direct baseline identity differs")
    return TaskMatrixSummary(
        schema=MATRIX_SCHEMA,
        groups=11,
        logical_coordinates=126,
        reuse_coordinates=6,
        train_coordinates=120,
        countries=(key,),
    )
