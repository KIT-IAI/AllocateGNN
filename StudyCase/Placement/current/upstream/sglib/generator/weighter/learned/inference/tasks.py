"""Portable inference identity and scientific schema constants."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


PREPARED_INFERENCE_SCHEMA_V2 = "cuz_prepared_inference_task_v2"
PREPARED_INFERENCE_SCHEMA_V3 = "cuz_prepared_inference_task_v3"
PREPARED_INFERENCE_SCHEMA = "cuz_prepared_inference_task_v4"
INFERENCE_COMPLETION_SCHEMA_V1 = "cuz_inference_task_completion_v1"
INFERENCE_COMPLETION_SCHEMA = "cuz_inference_task_completion_v2"
INFERENCE_FAILURE_SCHEMA = "cuz_inference_task_failure_v1"

class InferenceTaskError(RuntimeError):
    """Raised for an invalid, failed, or incomplete inference coordinate."""



@dataclass(frozen=True)
class PreparedInferenceTask:
    country: str
    group: str
    training_task_id: str
    family: str
    config: str
    signal: str
    parameter: str
    value: str
    seed: int
    fold: int
    feature_set: str
    repo_root: str
    results_root: str
    training_task_path: str
    checkpoint_path: str
    frozen_params_path: str
    graph_cache_path: str
    output_path: str
    training_task_results_relative: str
    checkpoint_results_relative: str
    frozen_params_repo_relative: str
    cache_results_relative: str
    output_results_relative: str
    selector: str
    fingerprints: dict[str, str]
    regions: tuple[str, ...]
    execution_backend: str
    compute_threads: int = 8
    schema: str = PREPARED_INFERENCE_SCHEMA
    execution_identity: dict[str, Any] = field(default_factory=dict)
    chain_commitment: dict[str, Any] = field(default_factory=dict)

    @property
    def task_id(self) -> str:
        return f"infer-{self.training_task_id}"

    def to_dict(self) -> dict[str, Any]:
        document = asdict(self)
        if self.schema == PREPARED_INFERENCE_SCHEMA:
            for name in (
                "repo_root",
                "results_root",
                "training_task_path",
                "checkpoint_path",
                "frozen_params_path",
                "graph_cache_path",
                "output_path",
                "execution_backend",
            ):
                document.pop(name)
        return {**document, "task_id": self.task_id}


_INFERENCE_SCIENTIFIC_FIELDS = (
    "training_task_id",
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
    "selector",
    "regions",
    "compute_threads",
    "run_fingerprint",
)
