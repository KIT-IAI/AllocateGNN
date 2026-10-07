"""Validate source fingerprints and embedded inference completion receipts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
from sglib.core.infra.hashing import sha256_file
from sglib.core.infra.content_chain import verify_chain
from ..training.tasks import BACKENDS, CHECKPOINT_SELECTOR, PreparedTrainingTask
from ..training.preparation import load_prepared_task
from ..training.verify import verify_task_artifacts
from .tasks import (PREPARED_INFERENCE_SCHEMA, INFERENCE_COMPLETION_SCHEMA,
    INFERENCE_COMPLETION_SCHEMA_V1, PreparedInferenceTask, InferenceTaskError)
from .preparation import fingerprinted_paths, load_inference_task
from .identity import inference_code_identity


def verify_inputs(task: PreparedInferenceTask) -> PreparedTrainingTask:
    for name, path in fingerprinted_paths(task).items():
        if not path.is_file() or sha256_file(path) != task.fingerprints[name]:
            raise InferenceTaskError(f"inference input changed or is missing: {name}")
    training = load_prepared_task(
        task.training_task_path,
        repo_root=task.repo_root,
        results_root=task.results_root,
    )
    if training.task_id != task.training_task_id:
        raise InferenceTaskError("training and inference task identities differ")
    verify_task_artifacts(training)
    if task.schema == PREPARED_INFERENCE_SCHEMA:
        observed_code = inference_code_identity(task.repo_root)["code_sha256"]
        if task.chain_commitment.get("code_sha256") != observed_code:
            raise InferenceTaskError("inference code differs from preregistration")
    return training



def verify_inference_artifacts(
    source: PreparedInferenceTask | Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    task = load_inference_task(source)
    root = Path(task.output_path)
    completion_path = root / "inference_completion.json"
    log_path = root / "inference_log.json"
    if not completion_path.is_file() or not log_path.is_file():
        raise InferenceTaskError(f"{task.task_id}: inference completion/log is missing")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    expected_schema = (
        INFERENCE_COMPLETION_SCHEMA
        if task.schema == PREPARED_INFERENCE_SCHEMA
        else INFERENCE_COMPLETION_SCHEMA_V1
    )
    if completion.get("schema") != expected_schema:
        raise InferenceTaskError("inference completion schema differs")
    if completion.get("task_id") != task.task_id:
        raise InferenceTaskError("inference completion task identity differs")
    identity_fields = (
        "training_task_id",
        "country",
        "group",
        "family",
        "config",
        "seed",
        "fold",
    )
    mismatched = [
        name for name in identity_fields
        if completion.get(name) != getattr(task, name)
    ]
    if mismatched:
        raise InferenceTaskError(
            f"inference completion task/config identity differs: {mismatched}"
        )
    if task.schema == PREPARED_INFERENCE_SCHEMA:
        if completion.get("execution_backend") not in BACKENDS:
            raise InferenceTaskError("inference completion backend observation is invalid")
    elif completion.get("execution_backend") != task.execution_backend:
        raise InferenceTaskError("inference completion backend differs")
    observed_threads = completion.get("compute_threads")
    if (
        task.schema == PREPARED_INFERENCE_SCHEMA
        and observed_threads != task.compute_threads
    ):
        raise InferenceTaskError("inference completion compute_threads differs")
    if observed_threads is not None and observed_threads != task.compute_threads:
        raise InferenceTaskError("inference completion compute_threads differs")
    if completion.get("selector") != CHECKPOINT_SELECTOR:
        raise InferenceTaskError("inference completion selector differs")
    if completion.get("input_fingerprints") != task.fingerprints:
        raise InferenceTaskError("inference completion input fingerprints differ")
    completion_identity = completion.get("execution_identity")
    if not isinstance(completion_identity, Mapping):
        raise InferenceTaskError("inference completion execution generation differs")
    if task.schema != PREPARED_INFERENCE_SCHEMA and any(
        completion_identity.get(name) != value
        for name, value in task.execution_identity.items()
    ):
        raise InferenceTaskError("inference completion execution generation differs")
    run_identity = task.execution_identity.get("run_identity")
    is_v3 = isinstance(run_identity, Mapping) and (
        run_identity.get("schema_version") == "sg_generator_execution_generation_v3"
    )
    optional_identity = ("signal", "parameter", "value", "feature_set")
    for name in optional_identity:
        observed = completion.get(name)
        if observed is not None and observed != getattr(task, name):
            raise InferenceTaskError(f"inference completion {name} identity differs")
        if is_v3 and observed != getattr(task, name):
            raise InferenceTaskError(f"V3 inference completion lacks {name} identity")
    if int(completion.get("n_regions", -1)) != len(task.regions):
        raise InferenceTaskError("inference completion region count differs")
    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise InferenceTaskError("inference artifact inventory is missing")
    expected_paths = {"inference_log.json", *{
        f"fields/{region}.npz" for region in task.regions
    }, *{
        f"fields/{region}.json" for region in task.regions
    }}
    if set(artifacts) != expected_paths:
        raise InferenceTaskError("inference artifact inventory differs from regions")
    for relative, record in artifacts.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise InferenceTaskError(f"inference artifact hash differs: {relative}")
    if task.schema == PREPARED_INFERENCE_SCHEMA:
        chain = verify_chain(completion.get("chain", {}))
        if chain["commitment"] != task.chain_commitment:
            raise InferenceTaskError("inference completion commitment differs")
        output_hashes = {
            name: record["sha256"] for name, record in artifacts.items()
        }
        if chain["outputs"] != dict(sorted(output_hashes.items())):
            raise InferenceTaskError("inference chain outputs differ from artifacts")
    return completion
