"""Execute one already prepared training task on the selected backend."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import shutil
import time
from typing import Any, Mapping
from sglib.core.infra.artifacts import atomic_json
from sglib.core.infra.hashing import sha256_file
from .tasks import (BACKENDS, DEVICES, CHECKPOINT_SELECTOR, COMPLETION_SCHEMA, FAILURE_SCHEMA,
    COUNTRY_PATTERN, GROUPS_PER_COUNTRY, LOGICAL_COORDINATES_PER_COUNTRY,
    REUSE_COORDINATES_PER_COUNTRY, TRAIN_COORDINATES_PER_COUNTRY,
    SUPPORTED_AUTHORITY_COUNTRY_COUNTS, EXPECTED_GROUPS, EXPECTED_LOGICAL_COORDINATES,
    EXPECTED_REUSE_COORDINATES, EXPECTED_TRAIN_COORDINATES, PreparedTrainingTask,
    TaskGroup, TaskMatrixError, TaskMatrixSummary, TrainingTask, TrainingTaskError)
from .matrix import (all_training_tasks, load_task_matrix, logical_tasks_for_group,
    tasks_for_group, validate_task_matrix, validate_country_task_matrix)
from .preparation import load_prepared_task, prepare_tasks, rebase_prepared_task, task_parameters
from .verify import training_losses, verify_task_artifacts


def _resolve_device(requested: str) -> str:
    if requested not in DEVICES:
        raise TrainingTaskError(f"device must be one of {DEVICES}")
    import torch

    available = torch.cuda.is_available()
    if requested == "auto":
        return "cuda" if available else "cpu"
    if requested == "cuda" and not available:
        raise TrainingTaskError("CUDA was selected but is unavailable")
    return requested



def _assert_backend(task: PreparedTrainingTask, backend: str) -> None:
    if backend not in BACKENDS or backend != task.execution_backend:
        raise TrainingTaskError(
            f"task backend is frozen as {task.execution_backend!r}, not {backend!r}"
        )
    slurm = bool(os.environ.get("SLURM_JOB_ID"))
    if backend == "hpc" and not slurm:
        raise TrainingTaskError("HPC training requires a Slurm allocation")
    if backend == "local" and slurm:
        raise TrainingTaskError("local training cannot run inside a Slurm allocation")



def run_training_task(
    source: PreparedTrainingTask | Mapping[str, Any] | str | Path,
    *,
    backend: str,
    device: str = "auto",
    execution_identity: Mapping[str, Any] | None = None,
) -> Path:
    """Execute one prepared coordinate; failures are recorded and never retried."""

    task = load_prepared_task(source)
    _assert_backend(task, backend)
    resolved_device = _resolve_device(device)
    if backend == "hpc" and resolved_device != "cuda":
        raise TrainingTaskError("formal HPC training must resolve to CUDA")
    final = Path(task.output_path)
    if (final / "task_completion.json").is_file():
        verify_task_artifacts(task)
        return final
    if final.exists():
        reason = "recorded failure" if (final / "task_failure.json").is_file() else "incomplete output"
        raise TrainingTaskError(
            f"{task.task_id}: {reason}; automatic retry is forbidden"
        )

    graph_path = Path(task.graph_cache_path)
    if not graph_path.is_file():
        raise FileNotFoundError(graph_path)
    expected_graph_hash = task.input_fingerprints.get("graph_cache")
    if expected_graph_hash != sha256_file(graph_path):
        raise TrainingTaskError("graph cache fingerprint changed after preparation")

    from ..inputs import GRAPH_CACHE_SCHEMA_V3, load_graph_cache
    from ..common import kfold_splits
    from ..models.gnn.training import train_gnn_fold
    from ..models.mlp import train_mlp_fold
    import torch

    torch.set_num_threads(task.compute_threads)

    run_identity = task.execution_identity.get("run_identity")
    if not isinstance(run_identity, Mapping) or (
        run_identity.get("schema_version")
        != "sg_generator_execution_generation_v3"
    ):
        raise TrainingTaskError(
            "Generator rerun training requires a V3 execution identity"
        )
    cache = load_graph_cache(
        graph_path,
        expected_country=task.country,
        expected_feature_set=task.feature_set,
        required_schema=GRAPH_CACHE_SCHEMA_V3,
    )
    graphs = cache["graphs"]
    params, tau_start = task_parameters(task)
    weighting_policy = params.get("training_weighting_policy")
    if weighting_policy != "uniform_region_cyclic_sgd__source_mean_v1":
        raise TrainingTaskError("unknown or missing training weighting policy")
    if int(params.get("training_batch_size", -1)) != 1:
        raise TrainingTaskError("formal training requires batch_size=1")
    for config_name, config_spec in params.get("config_map", {}).items():
        objectives = set(config_spec.get("objective_weights", {}))
        if "entropy_regularization" in objectives:
            raise TrainingTaskError(f"{config_name}: source-sum entropy loss is forbidden")
    splits = kfold_splits(params["regions"], task.seed, params["n_folds"])
    try:
        train_locs, _test_locs = splits[task.fold - 1]
    except IndexError as error:
        raise TrainingTaskError(f"fold out of range: {task.fold}") from error
    if len(train_locs) != len(set(train_locs)):
        raise TrainingTaskError("each training region must be visited exactly once per epoch")

    temporary = final.with_name(f".{final.name}.{task.task_id}.part")
    if temporary.exists():
        raise TrainingTaskError(
            f"{task.task_id}: interrupted temporary output requires manual resolution"
        )
    temporary.mkdir(parents=True)
    started_wall = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        if task.family == "gnn":
            train_gnn_fold(
                task.config,
                task.seed,
                task.fold - 1,
                train_locs,
                graphs,
                params,
                temporary,
                tau_start=tau_start,
                device=resolved_device,
            )
        elif task.family == "mlp":
            train_mlp_fold(
                task.config,
                task.seed,
                task.fold - 1,
                train_locs,
                graphs,
                params,
                temporary,
                device=resolved_device,
            )
        else:
            raise TrainingTaskError(f"unsupported learned family: {task.family!r}")

        model = temporary / "model.pth"
        log = temporary / "model_training_log.json"
        if not model.is_file() or not log.is_file():
            raise TrainingTaskError("trainer did not produce checkpoint and structured log")
        losses = training_losses(log)
        identity = {**task.execution_identity, **dict(execution_identity or {})}
        completion = {
            "schema": COMPLETION_SCHEMA,
            "task_id": task.task_id,
            "country": task.country,
            "group": task.group,
            "family": task.family,
            "config": task.config,
            "signal": task.signal,
            "parameter": task.parameter,
            "value": task.value,
            "seed": task.seed,
            "fold": task.fold,
            "feature_set": task.feature_set,
            "selector": task.selector,
            "training_weighting_policy": weighting_policy,
            "training_batch_size": 1,
            "compute_threads": task.compute_threads,
            "region_order": list(train_locs),
            "execution_backend": backend,
            "execution_identity": identity,
            "input_fingerprints": task.input_fingerprints,
            "config_fingerprint": task.config_fingerprint,
            "epochs_observed": len(losses),
            "selected_training_loss": min(losses),
            "artifacts": {
                "model.pth": {"sha256": sha256_file(model), "bytes": model.stat().st_size},
                "model_training_log.json": {
                    "sha256": sha256_file(log),
                    "bytes": log.stat().st_size,
                },
            },
            "runtime": {
                "device_requested": device,
                "device_resolved": resolved_device,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "hostname": platform.node(),
                "started_at": started_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": time.monotonic() - started_wall,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                "scheduler_observation": {
                    "requested_gres": os.environ.get("SG_REQUESTED_GRES"),
                    "requested_cpus": os.environ.get("SG_REQUESTED_CPUS"),
                    "requested_time_limit": os.environ.get(
                        "SG_REQUESTED_TIME_LIMIT"
                    ),
                    "max_concurrent": os.environ.get("SG_MAX_CONCURRENT"),
                    "partition": os.environ.get("SG_PARTITION"),
                    "account": os.environ.get("SG_ACCOUNT"),
                    "allocated_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
                },
            },
        }
        atomic_json(completion, temporary / "task_completion.json", exclusive=True)
        final.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(final)
        verify_task_artifacts(task)
        return final
    except Exception as error:
        if temporary.exists():
            shutil.rmtree(temporary)
        final.mkdir(parents=True, exist_ok=False)
        atomic_json({
                "schema": FAILURE_SCHEMA,
                "task_id": task.task_id,
                "execution_backend": backend,
                "error_type": type(error).__name__,
                "error": str(error),
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "automatic_retry": False,
            }, final / "task_failure.json", exclusive=True)
        raise


def main(argv=None):
    """Compatibility entry point for the dedicated command-line module."""
    from .cli import main as cli_main
    return cli_main(argv)


__all__ = [
    "BACKENDS",
    "CHECKPOINT_SELECTOR",
    "COMPLETION_SCHEMA",
    "COUNTRY_PATTERN",
    "GROUPS_PER_COUNTRY",
    "LOGICAL_COORDINATES_PER_COUNTRY",
    "REUSE_COORDINATES_PER_COUNTRY",
    "TRAIN_COORDINATES_PER_COUNTRY",
    "SUPPORTED_AUTHORITY_COUNTRY_COUNTS",
    "EXPECTED_GROUPS",
    "EXPECTED_LOGICAL_COORDINATES",
    "EXPECTED_REUSE_COORDINATES",
    "EXPECTED_TRAIN_COORDINATES",
    "PreparedTrainingTask",
    "TaskGroup",
    "TaskMatrixError",
    "TaskMatrixSummary",
    "TrainingTask",
    "TrainingTaskError",
    "all_training_tasks",
    "load_prepared_task",
    "load_task_matrix",
    "logical_tasks_for_group",
    "main",
    "prepare_tasks",
    "rebase_prepared_task",
    "run_training_task",
    "tasks_for_group",
    "validate_task_matrix",
    "validate_country_task_matrix",
    "verify_task_artifacts",
]


if __name__ == "__main__":  # compatibility with historical worker commands
    raise SystemExit(main())
