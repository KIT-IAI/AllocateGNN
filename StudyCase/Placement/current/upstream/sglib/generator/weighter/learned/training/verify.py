"""Verify historical and new training outputs against prepared identity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
from sglib.core.infra.hashing import sha256_file
from .tasks import COMPLETION_SCHEMA, PreparedTrainingTask, TrainingTaskError
from .preparation import load_prepared_task


def training_losses(path: Path) -> list[float]:
    document = json.loads(path.read_text(encoding="utf-8"))
    losses = document.get("train_losses")
    if isinstance(losses, Mapping):
        losses = losses.get("total")
    if not isinstance(losses, list) or not losses:
        raise TrainingTaskError("model_training_log.json has no train loss series")
    values = [float(value) for value in losses]
    if not all(value == value and abs(value) != float("inf") for value in values):
        raise TrainingTaskError("training loss series contains non-finite values")
    return values



def verify_task_artifacts(
    source: PreparedTrainingTask | Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """Verify checkpoint/log/completion identity and hashes."""

    task = load_prepared_task(source)
    root = Path(task.output_path)
    model = root / "model.pth"
    log = root / "model_training_log.json"
    completion_path = root / "task_completion.json"
    missing = [path.name for path in (model, log, completion_path) if not path.is_file()]
    if missing:
        raise TrainingTaskError(f"{task.task_id}: missing task artifacts {missing}")
    losses = training_losses(log)
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("schema") != COMPLETION_SCHEMA:
        raise TrainingTaskError("training completion schema differs")
    if completion.get("task_id") != task.task_id:
        raise TrainingTaskError("training completion task identity differs")
    identity_fields = (
        "country",
        "group",
        "family",
        "config",
        "signal",
        "parameter",
        "value",
        "seed",
        "fold",
        "feature_set",
    )
    mismatched = [
        name for name in identity_fields
        if completion.get(name) != getattr(task, name)
    ]
    if mismatched:
        raise TrainingTaskError(
            f"training completion task/config identity differs: {mismatched}"
        )
    # The execution backend is operational provenance.  A task may be rebased
    # from the historical HPC preparation to the local post-training chain
    # without changing what was computed.
    if completion.get("selector") != task.selector:
        raise TrainingTaskError("training completion selector differs")
    if completion.get("training_weighting_policy") != "uniform_region_cyclic_sgd__source_mean_v1":
        raise TrainingTaskError("training completion weighting policy differs")
    if completion.get("training_batch_size") != 1:
        raise TrainingTaskError("training completion batch size differs")
    observed_threads = completion.get("compute_threads")
    # Early V3 receipts predate the explicit telemetry field.  Their matching
    # config fingerprint still binds the unchanged worker parameters.
    if observed_threads is not None and observed_threads != task.compute_threads:
        raise TrainingTaskError("training completion compute_threads differs")
    completion_inputs = completion.get("input_fingerprints")
    if not isinstance(completion_inputs, Mapping):
        raise TrainingTaskError("training completion input fingerprints differ")
    observed_inputs = dict(completion_inputs)
    expected_inputs = dict(task.input_fingerprints)
    for fingerprints in (observed_inputs, expected_inputs):
        legacy_receipt = fingerprints.pop("inputs_receipt", None)
        current_receipt = fingerprints.get("inputs_receipt_sha256")
        if legacy_receipt is not None:
            if current_receipt is not None and current_receipt != legacy_receipt:
                raise TrainingTaskError("training completion input fingerprints differ")
            fingerprints["inputs_receipt_sha256"] = legacy_receipt
    if observed_inputs != expected_inputs:
        raise TrainingTaskError("training completion input fingerprints differ")
    completion_identity = completion.get("execution_identity")
    if not isinstance(completion_identity, Mapping) or any(
        completion_identity.get(name) != value
        for name, value in task.execution_identity.items()
    ):
        raise TrainingTaskError("training completion execution generation differs")
    run_identity = task.execution_identity.get("run_identity")
    is_v3 = isinstance(run_identity, Mapping) and (
        run_identity.get("schema_version") == "sg_generator_execution_generation_v3"
    )
    observed_config = completion.get("config_fingerprint")
    if observed_config is not None and observed_config != task.config_fingerprint:
        raise TrainingTaskError("training completion config fingerprint differs")
    if is_v3 and observed_config != task.config_fingerprint:
        raise TrainingTaskError("V3 training completion lacks current config fingerprint")
    region_order = completion.get("region_order")
    if not isinstance(region_order, list) or not region_order or len(region_order) != len(set(region_order)):
        raise TrainingTaskError("training completion region order is invalid")
    artifacts = completion.get("artifacts")
    expected = {
        "model.pth": sha256_file(model),
        "model_training_log.json": sha256_file(log),
    }
    if not isinstance(artifacts, Mapping) or {
        name: artifacts.get(name, {}).get("sha256") for name in expected
    } != expected:
        raise TrainingTaskError("training artifact hashes differ from completion")
    if int(completion.get("epochs_observed", -1)) != len(losses):
        raise TrainingTaskError("training completion epoch count differs")
    return completion
