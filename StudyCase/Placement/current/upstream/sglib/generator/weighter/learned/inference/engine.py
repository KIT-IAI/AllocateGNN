"""Execute a prepared CPU inference coordinate and publish its completion."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Mapping
import numpy as np
from sglib.core.infra.artifacts import atomic_json
from sglib.core.infra.hashing import sha256_file
from sglib.core.infra.content_chain import derive_chain_receipt
from ..inputs import load_graph_cache
from ..common import kfold_splits
from ..training.tasks import BACKENDS
from .tasks import (PREPARED_INFERENCE_SCHEMA, INFERENCE_COMPLETION_SCHEMA, INFERENCE_COMPLETION_SCHEMA_V1,
    INFERENCE_FAILURE_SCHEMA, PreparedInferenceTask, InferenceTaskError)
from .preparation import (load_inference_task, prepare_inference_tasks,
    rebase_inference_task, inference_scientific_parameters,
    _validate_inference_relative_identity)
from .verify import verify_inputs, verify_inference_artifacts
from .identity import inference_code_identity
from .checkpoint import task_parameters, load_gnn_cpu
from .field import infer_demand_field, write_field


def _assert_backend(task: PreparedInferenceTask, backend: str, device: str) -> None:
    if backend not in BACKENDS:
        raise InferenceTaskError("inference backend is invalid")
    if task.schema != PREPARED_INFERENCE_SCHEMA and backend != task.execution_backend:
        raise InferenceTaskError(
            f"inference backend is frozen as {task.execution_backend!r}, not {backend!r}"
        )
    if device != "cpu":
        raise InferenceTaskError("formal inference is CPU-only")
    slurm = bool(os.environ.get("SLURM_JOB_ID"))
    if backend == "hpc" and not slurm:
        raise InferenceTaskError("HPC inference requires a Slurm allocation")
    if backend == "local" and slurm:
        raise InferenceTaskError("local inference cannot run inside Slurm")



def run_inference_task(
    source: PreparedInferenceTask | Mapping[str, Any] | str | Path,
    *,
    backend: str,
    device: str = "cpu",
    execution_identity: Mapping[str, Any] | None = None,
) -> Path:
    """Run one all-region CPU inference coordinate with atomic publication."""

    task = load_inference_task(source)
    _assert_backend(task, backend, device)
    import torch

    torch.set_num_threads(task.compute_threads)
    final = Path(task.output_path)
    if (final / "inference_completion.json").is_file():
        verify_inference_artifacts(task)
        return final
    if final.exists():
        reason = "recorded failure" if (final / "inference_failure.json").is_file() else "incomplete output"
        raise InferenceTaskError(f"{task.task_id}: {reason}; automatic retry is forbidden")
    verify_inputs(task)
    from ..inputs import GRAPH_CACHE_SCHEMA_V3

    run_identity = task.execution_identity.get("run_identity")
    if not isinstance(run_identity, Mapping) or (
        run_identity.get("schema_version")
        != "sg_generator_execution_generation_v3"
    ):
        raise InferenceTaskError(
            "Generator rerun inference requires a V3 execution identity"
        )
    cache = load_graph_cache(
        task.graph_cache_path,
        expected_country=task.country,
        expected_feature_set=task.feature_set,
        required_schema=GRAPH_CACHE_SCHEMA_V3,
    )
    if tuple(cache["regions"]) != task.regions:
        raise InferenceTaskError("graph cache region order changed after preparation")
    params, tau_start = task_parameters(task)
    graphs = cache["graphs"]
    if task.family == "gnn":
        model = load_gnn_cpu(task, graphs, params, tau_start)
    elif task.family == "mlp":
        from ..models.mlp import load_mlp_checkpoint

        model = load_mlp_checkpoint(Path(task.checkpoint_path).parent, graphs, params, device="cpu")
    else:
        raise InferenceTaskError(f"unsupported inference family: {task.family!r}")

    temporary = final.with_name(f".{final.name}.{task.task_id}.part")
    if temporary.exists():
        raise InferenceTaskError("interrupted temporary inference requires manual resolution")
    (temporary / "fields").mkdir(parents=True)
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        splits = kfold_splits(params["regions"], task.seed, params["n_folds"])
        test_regions = set(splits[task.fold - 1][1])
        rows: list[dict[str, Any]] = []
        for region in task.regions:
            grid = cache["grids"][region][0]
            source_regions = cache["region_dict"][region]
            field = infer_demand_field(
                task.family,
                model,
                graphs[region],
                grid,
                source_regions,
                source_column=cache["source_column"],
                demand_column="Demand (MVA)",
                device="cpu",
            )
            npz_path = temporary / "fields" / f"{region}.npz"
            json_path = temporary / "fields" / f"{region}.json"
            write_field(npz_path, field)
            metadata = {
                "schema": "cuz_inference_field_v1",
                "country": task.country,
                "region": region,
                "training_task_id": task.training_task_id,
                "family": task.family,
                "config": task.config,
                "seed": task.seed,
                "fold": task.fold,
                "role": "TEST" if region in test_regions else "TRAIN",
                "selector": task.selector,
                "checkpoint_sha256": task.fingerprints["checkpoint"],
                "test_sanitizer": "physical_supervision_removal",
            }
            atomic_json(metadata, json_path, exclusive=True)
            rows.append(metadata)
        log_path = temporary / "inference_log.json"
        atomic_json({
                "schema": "cuz_inference_log_v1",
                "task_id": task.task_id,
                "device": "cpu",
                "regions": rows,
            }, log_path, exclusive=True)
        artifact_paths = [log_path] + sorted((temporary / "fields").iterdir())
        artifacts = {
            path.relative_to(temporary).as_posix(): {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in artifact_paths
        }
        runtime = {
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": time.monotonic() - started,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "scheduler_observation": {
                "requested_cpus": os.environ.get("SG_REQUESTED_CPUS"),
                "requested_time_limit": os.environ.get("SG_REQUESTED_TIME_LIMIT"),
                "max_concurrent": os.environ.get("SG_MAX_CONCURRENT"),
                "partition": os.environ.get("SG_PARTITION"),
                "account": os.environ.get("SG_ACCOUNT"),
            },
        }
        completion = {
            "schema": (
                INFERENCE_COMPLETION_SCHEMA
                if task.schema == PREPARED_INFERENCE_SCHEMA
                else INFERENCE_COMPLETION_SCHEMA_V1
            ),
            "task_id": task.task_id,
            "training_task_id": task.training_task_id,
            "country": task.country,
            "group": task.group,
            "family": task.family,
            "config": task.config,
            "signal": task.signal,
            "parameter": task.parameter,
            "value": task.value,
            "feature_set": task.feature_set,
            "seed": task.seed,
            "fold": task.fold,
            "selector": task.selector,
            "execution_backend": backend,
            "device": "cpu",
            "compute_threads": task.compute_threads,
            "input_fingerprints": task.fingerprints,
            "execution_identity": {
                **task.execution_identity,
                **dict(execution_identity or {}),
            },
            "n_regions": len(task.regions),
            "n_test_regions": len(test_regions),
            "supervision_sanitized": True,
            "artifacts": artifacts,
            "runtime": runtime,
        }
        if task.schema == PREPARED_INFERENCE_SCHEMA:
            completion["chain"] = derive_chain_receipt(
                task.chain_commitment,
                outputs={name: record["sha256"] for name, record in artifacts.items()},
                observations={
                    "execution_backend": backend,
                    "device": "cpu",
                    "runtime": runtime,
                },
            )
        atomic_json(completion, temporary / "inference_completion.json", exclusive=True)
        final.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(final)
        verify_inference_artifacts(task)
        return final
    except Exception as error:
        if temporary.exists():
            shutil.rmtree(temporary)
        final.mkdir(parents=True, exist_ok=False)
        atomic_json({
                "schema": INFERENCE_FAILURE_SCHEMA,
                "task_id": task.task_id,
                "execution_backend": backend,
                "error_type": type(error).__name__,
                "error": str(error),
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "automatic_retry": False,
            }, final / "inference_failure.json", exclusive=True)
        raise


def main(argv=None):
    """Compatibility entry point for the dedicated command-line module."""
    from .cli import main as cli_main
    return cli_main(argv)


__all__ = [
    "INFERENCE_COMPLETION_SCHEMA",
    "InferenceTaskError",
    "PREPARED_INFERENCE_SCHEMA",
    "PreparedInferenceTask",
    "inference_code_identity",
    "load_inference_task",
    "main",
    "prepare_inference_tasks",
    "rebase_inference_task",
    "run_inference_task",
    "verify_inference_artifacts",
]


if __name__ == "__main__":  # compatibility with historical worker commands
    raise SystemExit(main())
