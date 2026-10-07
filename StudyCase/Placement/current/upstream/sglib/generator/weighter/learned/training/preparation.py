"""Prepare portable training tasks and relocate their runtime paths."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from sglib.core.infra.artifacts import atomic_json
from sglib.core.infra.hashing import sha256_file
from ..paths import absolute_root, relative_to, join_relative, safe_relative_path
from .tasks import (BACKENDS, CHECKPOINT_SELECTOR, COUNTRY_PATTERN, PREPARED_TASK_SCHEMA,
    PREPARED_TASK_SCHEMA_V2, PreparedTrainingTask, TrainingTask, TrainingTaskError,
    TaskMatrixError)


def _require_sha256(value: str, name: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise TrainingTaskError(f"{name} must be a SHA-256 hex digest")
    return normalized



def prepare_tasks(
    tasks: Sequence[TrainingTask],
    *,
    frozen_params_path: str | Path,
    graph_cache_by_feature_set: Mapping[str, str | Path],
    repo_root: str | Path,
    results_root: str | Path,
    config_fingerprint: str,
    input_fingerprints: Mapping[str, str],
    execution_backend: str,
    selector: str = CHECKPOINT_SELECTOR,
    execution_identity: Mapping[str, Any] | None = None,
) -> tuple[PreparedTrainingTask, ...]:
    """Build runtime paths while serializing only portable relative identity."""

    if execution_backend not in BACKENDS:
        raise TrainingTaskError(f"execution_backend must be one of {BACKENDS}")
    if selector != CHECKPOINT_SELECTOR:
        raise TrainingTaskError(f"checkpoint selector must remain {CHECKPOINT_SELECTOR}")
    repository = absolute_root(repo_root, "repo_root")
    results = absolute_root(results_root, "results_root")
    params_path = Path(frozen_params_path).resolve()
    if not params_path.is_file():
        raise FileNotFoundError(params_path)
    params_relative = relative_to(
        params_path, repository, "frozen_params_path"
    )
    fingerprint = _require_sha256(config_fingerprint, "config_fingerprint")
    if sha256_file(params_path) != fingerprint:
        raise TrainingTaskError("frozen parameter fingerprint does not match file")
    fingerprints = {
        str(name): _require_sha256(value, f"input_fingerprints[{name!r}]")
        for name, value in input_fingerprints.items()
    }
    if not fingerprints:
        raise TrainingTaskError("at least one input fingerprint is required")
    prepared: list[PreparedTrainingTask] = []
    for task in tasks:
        if task.reuse:
            raise TrainingTaskError("reuse coordinates must not be prepared for training")
        if COUNTRY_PATTERN.fullmatch(task.country) is None:
            raise TrainingTaskError(f"invalid task country: {task.country!r}")
        try:
            graph_path = Path(graph_cache_by_feature_set[task.feature_set]).resolve()
        except KeyError as error:
            raise TrainingTaskError(
                f"missing graph cache for feature_set={task.feature_set!r}"
            ) from error
        if not graph_path.is_file():
            raise FileNotFoundError(graph_path)
        graph_relative = relative_to(
            graph_path, results, "graph_cache_path"
        )
        output_relative = safe_relative_path(task.output_relative)
        output = (results / output_relative).resolve()
        if output == results or results not in output.parents:
            raise TrainingTaskError(f"task output escapes output_root: {task.output_relative}")
        task_fingerprints = dict(fingerprints)
        task_fingerprints["graph_cache"] = sha256_file(graph_path)
        prepared.append(
            PreparedTrainingTask(
                group=task.group,
                country=task.country,
                family=task.family,
                config=task.config,
                signal=task.signal,
                parameter=task.parameter,
                value=task.value,
                seed=task.seed,
                fold=task.fold,
                feature_set=task.feature_set,
                repo_root=str(repository),
                results_root=str(results),
                frozen_params_path=str(params_path),
                graph_cache_path=str(graph_path),
                output_path=str(output),
                frozen_params_repo_relative=params_relative,
                graph_cache_results_relative=graph_relative,
                output_results_relative=output_relative,
                config_fingerprint=fingerprint,
                input_fingerprints=task_fingerprints,
                execution_backend=execution_backend,
                compute_threads=task.compute_threads,
                selector=selector,
                execution_identity=dict(execution_identity or {}),
            )
        )
    if len({task.task_id for task in prepared}) != len(prepared):
        raise TrainingTaskError("prepared task ids must be unique")
    return tuple(prepared)



def _parse_prepared_task(
    source: PreparedTrainingTask | Mapping[str, Any] | str | Path,
) -> PreparedTrainingTask:
    if isinstance(source, PreparedTrainingTask):
        return source
    if isinstance(source, Mapping):
        document = dict(source)
    else:
        try:
            document = json.loads(Path(source).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise TrainingTaskError(f"cannot load prepared task: {source}") from error
    document.pop("task_id", None)
    schema = document.get("schema", PREPARED_TASK_SCHEMA_V2)
    if schema == PREPARED_TASK_SCHEMA_V2:
        document.setdefault("compute_threads", 8)
    elif schema == PREPARED_TASK_SCHEMA:
        for name in (
            "repo_root",
            "results_root",
            "frozen_params_path",
            "graph_cache_path",
            "output_path",
        ):
            document.setdefault(name, "")
    try:
        return PreparedTrainingTask(**document)
    except TypeError as error:
        raise TrainingTaskError("invalid prepared task fields") from error



def _validate_prepared_relative_identity(task: PreparedTrainingTask) -> None:
    if task.schema not in {PREPARED_TASK_SCHEMA_V2, PREPARED_TASK_SCHEMA}:
        raise TrainingTaskError("unknown prepared task schema")
    if COUNTRY_PATTERN.fullmatch(task.country) is None or task.execution_backend not in BACKENDS:
        raise TrainingTaskError("prepared task country/backend is invalid")
    if task.selector != CHECKPOINT_SELECTOR:
        raise TrainingTaskError("prepared task checkpoint selector differs")
    if type(task.compute_threads) is not int or task.compute_threads <= 0:
        raise TrainingTaskError(
            "prepared task compute_threads must be a positive integer"
        )
    _require_sha256(task.config_fingerprint, "config_fingerprint")
    if not task.input_fingerprints:
        raise TrainingTaskError("prepared task input fingerprints are empty")
    for name, value in task.input_fingerprints.items():
        _require_sha256(value, f"input_fingerprints[{name!r}]")
    for name in (
        "frozen_params_repo_relative",
        "graph_cache_results_relative",
        "output_results_relative",
    ):
        try:
            safe_relative_path(getattr(task, name))
        except TaskMatrixError as error:
            raise TrainingTaskError(f"{name} is not a safe relative path") from error



def _validate_prepared_origin(task: PreparedTrainingTask) -> None:
    repository = absolute_root(task.repo_root, "prepared repo_root")
    results = absolute_root(task.results_root, "prepared results_root")
    expected = {
        "frozen_params_path": join_relative(
            repository,
            task.frozen_params_repo_relative,
            "frozen_params_repo_relative",
        ),
        "graph_cache_path": join_relative(
            results,
            task.graph_cache_results_relative,
            "graph_cache_results_relative",
        ),
        "output_path": join_relative(
            results,
            task.output_results_relative,
            "output_results_relative",
        ),
    }
    for name, path in expected.items():
        if Path(getattr(task, name)).resolve(strict=False) != path:
            raise TrainingTaskError(f"prepared task {name} differs from relative identity")



def load_prepared_task(
    source: PreparedTrainingTask | Mapping[str, Any] | str | Path,
    *,
    repo_root: str | Path | None = None,
    results_root: str | Path | None = None,
) -> PreparedTrainingTask:
    task = _parse_prepared_task(source)
    _validate_prepared_relative_identity(task)
    if (repo_root is None) != (results_root is None):
        raise TrainingTaskError("repo_root and results_root overrides must be supplied together")
    if repo_root is not None and results_root is not None:
        return rebase_prepared_task(
            task, repo_root=repo_root, results_root=results_root
        )
    if task.schema == PREPARED_TASK_SCHEMA and not all(
        (
            task.repo_root,
            task.results_root,
            task.frozen_params_path,
            task.graph_cache_path,
            task.output_path,
        )
    ):
        raise TrainingTaskError(
            "portable prepared task requires repo_root/results_root overrides"
        )
    _validate_prepared_origin(task)
    return task



def rebase_prepared_task(
    source: PreparedTrainingTask | Mapping[str, Any] | str | Path,
    *,
    repo_root: str | Path,
    results_root: str | Path,
    write_path: str | Path | None = None,
) -> PreparedTrainingTask:
    """Rebuild runtime absolute paths solely from canonical relative identities."""

    task = _parse_prepared_task(source)
    _validate_prepared_relative_identity(task)
    repository = absolute_root(repo_root, "repo_root")
    results = absolute_root(results_root, "results_root")
    rebased = replace(
        task,
        repo_root=str(repository),
        results_root=str(results),
        frozen_params_path=str(
            join_relative(
                repository,
                task.frozen_params_repo_relative,
                "frozen_params_repo_relative",
            )
        ),
        graph_cache_path=str(
            join_relative(
                results,
                task.graph_cache_results_relative,
                "graph_cache_results_relative",
            )
        ),
        output_path=str(
            join_relative(
                results,
                task.output_results_relative,
                "output_results_relative",
            )
        ),
    )
    params_path = Path(rebased.frozen_params_path)
    graph_path = Path(rebased.graph_cache_path)
    if not params_path.is_file() or sha256_file(params_path) != rebased.config_fingerprint:
        raise TrainingTaskError("rebased frozen parameters are missing or hash-drifted")
    expected_graph = rebased.input_fingerprints.get("graph_cache")
    if not graph_path.is_file() or sha256_file(graph_path) != expected_graph:
        raise TrainingTaskError("rebased graph cache is missing or hash-drifted")
    if write_path is not None:
        destination = Path(write_path).resolve(strict=False)
        if destination.exists():
            raise FileExistsError(destination)
        atomic_json(rebased.to_dict(), destination, exclusive=True)
    return rebased



def task_parameters(task: PreparedTrainingTask) -> tuple[dict[str, Any], float | None]:
    params_path = Path(task.frozen_params_path)
    if not params_path.is_file() or sha256_file(params_path) != task.config_fingerprint:
        raise TrainingTaskError("frozen parameter file or fingerprint changed after preparation")
    params = json.loads(params_path.read_text(encoding="utf-8-sig"))
    if task.config not in params.get("config_map", {}):
        raise TrainingTaskError(f"training config is not frozen: {task.config!r}")
    tau_start = None
    if task.parameter == "lambda":
        weight = float(task.value)
        objective = params["config_map"][task.config]["objective_weights"]
        names = {
            "N": ("ntl_prior",),
            "P": ("proximity_prior",),
            "NP": ("ntl_prior", "proximity_prior"),
        }.get(task.signal)
        if not names or any(name not in objective for name in names):
            raise TrainingTaskError("lambda task signal and objective do not match")
        for name in names:
            objective[name] = weight
    elif task.parameter == "tau":
        tau_start = float(task.value)
    return params, tau_start
