"""Prepare inference coordinates and relocate portable input identities."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from sglib.core.infra.artifacts import atomic_json
from sglib.core.infra.hashing import sha256_file
from sglib.core.infra.content_chain import derive_chain_commitment, verify_chain
from ..paths import absolute_root, relative_to, join_relative
from ..training.tasks import BACKENDS, CHECKPOINT_SELECTOR, COUNTRY_PATTERN, PreparedTrainingTask, TrainingTaskError
from ..training.preparation import load_prepared_task, rebase_prepared_task
from ..training.verify import verify_task_artifacts
from ..inputs import load_graph_cache
from .tasks import (PREPARED_INFERENCE_SCHEMA_V2, PREPARED_INFERENCE_SCHEMA_V3,
    PREPARED_INFERENCE_SCHEMA, PreparedInferenceTask, InferenceTaskError,
    _INFERENCE_SCIENTIFIC_FIELDS)
from .identity import inference_code_identity


def _require_sha(value: str, name: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise InferenceTaskError(f"{name} must be a SHA-256 hex digest")
    return normalized



def inference_scientific_parameters(
    task: PreparedInferenceTask,
) -> dict[str, Any]:
    values = {name: getattr(task, name) for name in _INFERENCE_SCIENTIFIC_FIELDS[:-1]}
    values["regions"] = list(task.regions)
    run = task.execution_identity.get("run_identity")
    values["run_fingerprint"] = run.get("run_fingerprint") if isinstance(run, Mapping) else None
    _require_sha(str(values["run_fingerprint"]), "training run_fingerprint")
    return values



def prepare_inference_tasks(
    training_task_paths: Sequence[str | Path],
    *,
    output_root: str | Path,
    repo_root: str | Path,
    results_root: str | Path,
    execution_backend: str,
    execution_identity: Mapping[str, Any] | None = None,
    verified_training: Mapping[
        str, tuple[PreparedTrainingTask, Mapping[str, Any]]
    ] | None = None,
) -> tuple[PreparedInferenceTask, ...]:
    """Derive portable checkpoint, graph, config, region, and output identities."""

    if execution_backend not in BACKENDS:
        raise InferenceTaskError("inference execution backend is invalid")
    repository = absolute_root(repo_root, "repo_root")
    results = absolute_root(results_root, "results_root")
    root = Path(output_root).resolve()
    if root != results and results not in root.parents:
        raise InferenceTaskError("output_root must remain within results_root")
    code_identity = inference_code_identity(repository)
    prepared: list[PreparedInferenceTask] = []
    for source in training_task_paths:
        training_path = Path(source).resolve()
        if not training_path.is_file():
            raise FileNotFoundError(training_path)
        training_relative = relative_to(
            training_path, results, "training_task_path"
        )
        verified = (verified_training or {}).get(str(training_path))
        if verified is None:
            training = load_prepared_task(
                training_path, repo_root=repository, results_root=results
            )
            training_completion = verify_task_artifacts(training)
        else:
            training, training_completion = verified
        if training.selector != CHECKPOINT_SELECTOR:
            raise InferenceTaskError("training checkpoint selector differs")
        checkpoint = Path(training.output_path) / "model.pth"
        checkpoint_relative = relative_to(
            checkpoint, results, "checkpoint_path"
        )
        from ..inputs import GRAPH_CACHE_SCHEMA_V3

        cache = load_graph_cache(
            training.graph_cache_path,
            expected_country=training.country,
            expected_feature_set=training.feature_set,
            required_schema=GRAPH_CACHE_SCHEMA_V3,
        )
        output = (root / f"infer-{training.task_id}").resolve()
        if output == root or root not in output.parents:
            raise InferenceTaskError("inference output escapes output_root")
        params_path = Path(training.frozen_params_path)
        fingerprints = {
            "checkpoint": sha256_file(checkpoint),
            "graph_cache": sha256_file(Path(training.graph_cache_path)),
            "frozen_params": sha256_file(params_path),
        }
        if fingerprints["frozen_params"] != training.config_fingerprint:
            raise InferenceTaskError("training config fingerprint differs at inference prep")
        if training_completion.get("selector") != CHECKPOINT_SELECTOR:
            raise InferenceTaskError("verified training selector differs")
        identity = dict(training.execution_identity)
        identity.update(execution_identity or {})
        task = PreparedInferenceTask(
                country=training.country,
                group=training.group,
                training_task_id=training.task_id,
                family=training.family,
                config=training.config,
                signal=training.signal,
                parameter=training.parameter,
                value=training.value,
                seed=training.seed,
                fold=training.fold,
                feature_set=training.feature_set,
                repo_root=str(repository),
                results_root=str(results),
                training_task_path=str(training_path),
                checkpoint_path=str(checkpoint.resolve()),
                frozen_params_path=str(params_path.resolve()),
                graph_cache_path=str(Path(training.graph_cache_path).resolve()),
                output_path=str(output),
                training_task_results_relative=training_relative,
                checkpoint_results_relative=checkpoint_relative,
                frozen_params_repo_relative=relative_to(
                    params_path, repository, "frozen_params_path"
                ),
                cache_results_relative=relative_to(
                    training.graph_cache_path, results, "graph_cache_path"
                ),
                output_results_relative=relative_to(
                    output, results, "output_path"
                ),
                selector=CHECKPOINT_SELECTOR,
                fingerprints=fingerprints,
                regions=tuple(cache["regions"]),
                execution_backend=execution_backend,
                compute_threads=training.compute_threads,
                execution_identity=identity,
            )
        commitment = derive_chain_commitment(
            f"generator.inference.{training.task_id}",
            inputs=fingerprints,
            scientific_parameters=inference_scientific_parameters(task),
            code_sha256=code_identity["code_sha256"],
        )
        prepared.append(replace(task, chain_commitment=commitment))
    if len({task.task_id for task in prepared}) != len(prepared):
        raise InferenceTaskError("prepared inference task ids are not unique")
    return tuple(prepared)



def _parse_inference_task(
    source: PreparedInferenceTask | Mapping[str, Any] | str | Path,
) -> PreparedInferenceTask:
    if isinstance(source, PreparedInferenceTask):
        return source
    if isinstance(source, Mapping):
        document = dict(source)
    else:
        try:
            document = json.loads(Path(source).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise InferenceTaskError(f"cannot load prepared inference task: {source}") from error
    document.pop("task_id", None)
    schema = document.get("schema", PREPARED_INFERENCE_SCHEMA_V2)
    if schema == PREPARED_INFERENCE_SCHEMA_V2:
        document.setdefault("compute_threads", 8)
        document.setdefault("chain_commitment", {})
    elif schema == PREPARED_INFERENCE_SCHEMA_V3:
        document.setdefault("chain_commitment", {})
    elif schema == PREPARED_INFERENCE_SCHEMA:
        for name in (
            "repo_root",
            "results_root",
            "training_task_path",
            "checkpoint_path",
            "frozen_params_path",
            "graph_cache_path",
            "output_path",
        ):
            document.setdefault(name, "")
        document.setdefault("execution_backend", "local")
    if "regions" in document:
        document["regions"] = tuple(document["regions"])
    try:
        return PreparedInferenceTask(**document)
    except TypeError as error:
        raise InferenceTaskError("invalid prepared inference fields") from error



def _validate_inference_relative_identity(task: PreparedInferenceTask) -> None:
    if task.schema not in {
        PREPARED_INFERENCE_SCHEMA_V2,
        PREPARED_INFERENCE_SCHEMA_V3,
        PREPARED_INFERENCE_SCHEMA,
    }:
        raise InferenceTaskError("unknown prepared inference schema")
    if (
        COUNTRY_PATTERN.fullmatch(task.country) is None
        or task.execution_backend not in BACKENDS
    ):
        raise InferenceTaskError("prepared inference country/backend is invalid")
    if task.selector != CHECKPOINT_SELECTOR or not task.regions:
        raise InferenceTaskError("prepared inference selector/regions are invalid")
    if type(task.compute_threads) is not int or task.compute_threads <= 0:
        raise InferenceTaskError(
            "prepared inference compute_threads must be a positive integer"
        )
    for name in (
        "training_task_results_relative",
        "checkpoint_results_relative",
        "frozen_params_repo_relative",
        "cache_results_relative",
        "output_results_relative",
    ):
        try:
            join_relative(Path.cwd().resolve(), getattr(task, name), name)
        except TrainingTaskError as error:
            raise InferenceTaskError(f"{name} is not a safe relative path") from error
    required = {"checkpoint", "graph_cache", "frozen_params"}
    if task.schema != PREPARED_INFERENCE_SCHEMA:
        required |= {"prepared_training_task", "training_completion"}
    if set(task.fingerprints) != required:
        raise InferenceTaskError("prepared inference fingerprint set differs")
    for name, value in task.fingerprints.items():
        _require_sha(value, f"fingerprints[{name!r}]")
    if task.schema == PREPARED_INFERENCE_SCHEMA:
        commitment = verify_chain(task.chain_commitment)
        expected_parameters = inference_scientific_parameters(task)
        if commitment["node_id"] != f"generator.inference.{task.training_task_id}":
            raise InferenceTaskError("prepared inference chain node identity differs")
        if commitment["inputs"] != dict(sorted(task.fingerprints.items())):
            raise InferenceTaskError("prepared inference chain inputs differ")
        if commitment["scientific_parameters"] != expected_parameters:
            raise InferenceTaskError("prepared inference scientific projection differs")



def _validate_inference_origin(task: PreparedInferenceTask) -> None:
    repository = absolute_root(task.repo_root, "prepared repo_root")
    results = absolute_root(task.results_root, "prepared results_root")
    expected = {
        "training_task_path": join_relative(
            results,
            task.training_task_results_relative,
            "training_task_results_relative",
        ),
        "checkpoint_path": join_relative(
            results,
            task.checkpoint_results_relative,
            "checkpoint_results_relative",
        ),
        "frozen_params_path": join_relative(
            repository,
            task.frozen_params_repo_relative,
            "frozen_params_repo_relative",
        ),
        "graph_cache_path": join_relative(
            results, task.cache_results_relative, "cache_results_relative"
        ),
        "output_path": join_relative(
            results, task.output_results_relative, "output_results_relative"
        ),
    }
    for name, path in expected.items():
        if Path(getattr(task, name)).resolve(strict=False) != path:
            raise InferenceTaskError(f"{name} differs from canonical relative identity")



def fingerprinted_paths(task: PreparedInferenceTask) -> dict[str, Path]:
    paths = {
        "checkpoint": Path(task.checkpoint_path),
        "graph_cache": Path(task.graph_cache_path),
        "frozen_params": Path(task.frozen_params_path),
    }
    if task.schema != PREPARED_INFERENCE_SCHEMA:
        paths["prepared_training_task"] = Path(task.training_task_path)
        paths["training_completion"] = Path(task.checkpoint_path).parent / "task_completion.json"
    return paths



def load_inference_task(
    source: PreparedInferenceTask | Mapping[str, Any] | str | Path,
    *,
    repo_root: str | Path | None = None,
    results_root: str | Path | None = None,
) -> PreparedInferenceTask:
    task = _parse_inference_task(source)
    _validate_inference_relative_identity(task)
    if (repo_root is None) != (results_root is None):
        raise InferenceTaskError("repo_root and results_root overrides must be supplied together")
    if repo_root is not None and results_root is not None:
        return rebase_inference_task(
            task, repo_root=repo_root, results_root=results_root
        )
    if task.schema == PREPARED_INFERENCE_SCHEMA and not all(
        (
            task.repo_root,
            task.results_root,
            task.training_task_path,
            task.checkpoint_path,
            task.frozen_params_path,
            task.graph_cache_path,
            task.output_path,
        )
    ):
        raise InferenceTaskError(
            "portable prepared inference task requires repo_root/results_root overrides"
        )
    _validate_inference_origin(task)
    return task



def rebase_inference_task(
    source: PreparedInferenceTask | Mapping[str, Any] | str | Path,
    *,
    repo_root: str | Path,
    results_root: str | Path,
    write_path: str | Path | None = None,
) -> PreparedInferenceTask:
    """Rebuild every runtime inference path from its canonical relative identity."""

    task = _parse_inference_task(source)
    _validate_inference_relative_identity(task)
    repository = absolute_root(repo_root, "repo_root")
    results = absolute_root(results_root, "results_root")
    rebased = replace(
        task,
        repo_root=str(repository),
        results_root=str(results),
        training_task_path=str(
            join_relative(
                results,
                task.training_task_results_relative,
                "training_task_results_relative",
            )
        ),
        checkpoint_path=str(
            join_relative(
                results,
                task.checkpoint_results_relative,
                "checkpoint_results_relative",
            )
        ),
        frozen_params_path=str(
            join_relative(
                repository,
                task.frozen_params_repo_relative,
                "frozen_params_repo_relative",
            )
        ),
        graph_cache_path=str(
            join_relative(
                results, task.cache_results_relative, "cache_results_relative"
            )
        ),
        output_path=str(
            join_relative(
                results, task.output_results_relative, "output_results_relative"
            )
        ),
    )
    for name, path in fingerprinted_paths(rebased).items():
        if not path.is_file() or sha256_file(path) != rebased.fingerprints[name]:
            raise InferenceTaskError(f"rebased inference input is missing or hash-drifted: {name}")
    training = rebase_prepared_task(
        rebased.training_task_path,
        repo_root=repository,
        results_root=results,
    )
    if training.task_id != rebased.training_task_id:
        raise InferenceTaskError("rebased training/inference identities differ")
    verify_task_artifacts(training)
    if rebased.schema == PREPARED_INFERENCE_SCHEMA:
        observed_code = inference_code_identity(repository)["code_sha256"]
        if rebased.chain_commitment.get("code_sha256") != observed_code:
            raise InferenceTaskError("inference code differs from preregistration")
    if write_path is not None:
        destination = Path(write_path).resolve(strict=False)
        if destination.exists():
            raise FileExistsError(destination)
        atomic_json(rebased.to_dict(), destination, exclusive=True)
    return rebased
