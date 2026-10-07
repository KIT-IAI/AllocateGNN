"""Training task types, scientific constants, and row parsing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any, Mapping
from ..errors import TaskMatrixError, TrainingTaskError
from ..paths import safe_relative_path


MATRIX_SCHEMA = "cuz_training_task_matrix_v2"
PREPARED_TASK_SCHEMA_V2 = "cuz_prepared_training_task_v2"
PREPARED_TASK_SCHEMA = "cuz_prepared_training_task_v3"
COMPLETION_SCHEMA = "cuz_training_task_completion_v1"
FAILURE_SCHEMA = "cuz_training_task_failure_v1"
COUNTRY_PATTERN = re.compile(r"^[a-z]{2}$")
BACKENDS = ("local", "hpc")
DEVICES = ("auto", "cpu", "cuda")
CHECKPOINT_SELECTOR = "outer_train_min_training_loss"
GROUPS_PER_COUNTRY = 11
LOGICAL_COORDINATES_PER_COUNTRY = 126
REUSE_COORDINATES_PER_COUNTRY = 6
TRAIN_COORDINATES_PER_COUNTRY = 120

# 004 expands the formal authority to UK/AU/NL/NZ.  The loader deliberately
# continues to accept the two-country authority while that file is being
# upgraded, but these exported totals describe the completed four-country
# contract.
EXPECTED_GROUPS = 44
EXPECTED_LOGICAL_COORDINATES = 504
EXPECTED_REUSE_COORDINATES = 24
EXPECTED_TRAIN_COORDINATES = 480
SUPPORTED_AUTHORITY_COUNTRY_COUNTS = frozenset({2, 4})

CONFIG_MAP = {
    "base": "baseline",
    "priorN": "prior_ntl",
    "priorP": "prior_proximity",
    "priorNP": "prior_ntl_proximity",
    "fusionN": "fusion_ntl",
    "fusionP": "fusion_proximity",
    "fusionNP": "fusion_ntl_proximity",
}

_REQUIRED_COLUMNS = {
    "task_group",
    "stage",
    "country",
    "family",
    "config",
    "signal",
    "parameter_name",
    "parameter_values",
    "seeds",
    "folds",
    "logical_coordinates",
    "reuse_coordinates",
    "new_train_coordinates",
    "compute_threads",
    "depends_on",
    "reuse_source",
}
_LEGACY_OPERATIONAL_COLUMNS = {
    "execution_policy",
    "gres",
    "cpus",
    "time_limit",
    "array_formula",
    "max_concurrent",
    "output_pattern",
}

_COUNTRY_DIRECTORIES = {
    "uk": "1_UK",
    "au": "2_AU",
    "nl": "4_NL",
    "nz": "5_NZ",
}


def _parts(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(value).split("|") if part.strip())



def _positive_int(value: Any, field_name: str, *, allow_zero: bool = False) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise TaskMatrixError(f"{field_name} must be an integer") from error
    minimum = 0 if allow_zero else 1
    if result < minimum:
        raise TaskMatrixError(f"{field_name} must be >= {minimum}")
    return result



@dataclass(frozen=True)
class TaskGroup:
    task_group: str
    stage: str
    country: str
    family: str
    config: str
    signal: str
    parameter_name: str
    parameter_values: tuple[str, ...]
    seeds: tuple[int, ...]
    folds: tuple[int, ...]
    logical_coordinates: int
    reuse_coordinates: int
    new_train_coordinates: int
    compute_threads: int
    depends_on: str
    reuse_source: str

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> "TaskGroup":
        missing = sorted(_REQUIRED_COLUMNS - set(row))
        if missing:
            raise TaskMatrixError(f"task matrix row is missing columns: {missing}")
        country = str(row["country"]).lower()
        if COUNTRY_PATTERN.fullmatch(country) is None:
            raise TaskMatrixError(f"invalid two-letter country code: {country!r}")
        group = str(row["task_group"]).strip()
        family = str(row["family"]).lower()
        config = str(row["config"]).strip()
        values = _parts(row["parameter_values"])
        seeds = tuple(_positive_int(value, "seed") for value in _parts(row["seeds"]))
        folds = tuple(_positive_int(value, "fold") for value in _parts(row["folds"]))
        if not group or family not in {"gnn", "mlp"} or config not in CONFIG_MAP:
            raise TaskMatrixError(f"{group or '<empty>'}: invalid family/config")
        if not values or not seeds or not folds:
            raise TaskMatrixError(f"{group}: values, seeds, and folds must be non-empty")
        logical = _positive_int(row["logical_coordinates"], "logical_coordinates")
        reuse = _positive_int(
            row["reuse_coordinates"], "reuse_coordinates", allow_zero=True
        )
        new = _positive_int(row["new_train_coordinates"], "new_train_coordinates")
        if logical != len(values) * len(seeds) * len(folds):
            raise TaskMatrixError(f"{group}: logical coordinate product is inconsistent")
        if logical != reuse + new:
            raise TaskMatrixError(f"{group}: logical != reuse + new")
        return cls(
            task_group=group,
            stage=str(row["stage"]).strip(),
            country=country,
            family=family,
            config=config,
            signal=str(row["signal"]).strip(),
            parameter_name=str(row["parameter_name"]).strip(),
            parameter_values=values,
            seeds=seeds,
            folds=folds,
            logical_coordinates=logical,
            reuse_coordinates=reuse,
            new_train_coordinates=new,
            compute_threads=_positive_int(row["compute_threads"], "compute_threads"),
            depends_on=str(row["depends_on"]).strip(),
            reuse_source=str(row["reuse_source"]).strip(),
        )

    @property
    def package_config(self) -> str:
        return CONFIG_MAP[self.config]

    @property
    def feature_set(self) -> str:
        return self.config if self.config.startswith("fusion") else "lu5"

    @property
    def execution_policy(self) -> str:
        return "USER_SELECTED_LOCAL_OR_HPC"

    @property
    def array_formula(self) -> str:
        if self.stage == "lambda_sweep":
            return "index=new_lambda_index"
        if self.stage == "tau_sweep":
            return "index=new_tau_index*4+fold_minus_1"
        return "index=seed_index*4+fold_minus_1"

    @property
    def output_pattern(self) -> str:
        try:
            prefix = f"2_Generator/{_COUNTRY_DIRECTORIES[self.country]}/2_Weighter"
        except KeyError as exc:
            raise TaskMatrixError(
                f"{self.task_group}: no operational output layout for {self.country!r}"
            ) from exc
        if self.stage == "base":
            suffix = f"base/{self.family}/seed_{{seed}}/fold{{fold}}"
        elif self.stage == "lambda_sweep":
            suffix = (
                f"sweeps/lambda/{self.config}/lambda_{{value}}/"
                "seed_{seed}/fold{fold}"
            )
        elif self.stage == "tau_sweep":
            suffix = "sweeps/tau/tau_{value}/seed_{seed}/fold{fold}"
        else:
            suffix = f"{self.config}/seed_{{seed}}/fold{{fold}}"
        return safe_relative_path(f"{prefix}/{suffix}")



@dataclass(frozen=True)
class TrainingTask:
    group: str
    country: str
    family: str
    config: str
    signal: str
    parameter: str
    value: str
    seed: int
    fold: int
    output_relative: str
    feature_set: str
    compute_threads: int = 8
    reuse: bool = False
    source_task_id: str | None = None
    source_output_relative: str | None = None

    @property
    def task_id(self) -> str:
        parameter = (
            f"-{self.parameter}{self.value}"
            if self.parameter not in {"fixed", "input_dimension"}
            else ""
        )
        return f"{self.group}{parameter}-S{self.seed}-F{self.fold}"



@dataclass(frozen=True)
class TaskMatrixSummary:
    schema: str
    groups: int
    logical_coordinates: int
    reuse_coordinates: int
    train_coordinates: int
    countries: tuple[str, ...]



@dataclass(frozen=True)
class PreparedTrainingTask:
    group: str
    country: str
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
    frozen_params_path: str
    graph_cache_path: str
    output_path: str
    frozen_params_repo_relative: str
    graph_cache_results_relative: str
    output_results_relative: str
    config_fingerprint: str
    input_fingerprints: dict[str, str]
    execution_backend: str
    compute_threads: int = 8
    selector: str = CHECKPOINT_SELECTOR
    schema: str = PREPARED_TASK_SCHEMA
    execution_identity: dict[str, Any] = field(default_factory=dict)

    @property
    def task_id(self) -> str:
        parameter = (
            f"-{self.parameter}{self.value}"
            if self.parameter not in {"fixed", "input_dimension"}
            else ""
        )
        return f"{self.group}{parameter}-S{self.seed}-F{self.fold}"

    def to_dict(self) -> dict[str, Any]:
        document = asdict(self)
        if self.schema == PREPARED_TASK_SCHEMA:
            for name in (
                "repo_root",
                "results_root",
                "frozen_params_path",
                "graph_cache_path",
                "output_path",
            ):
                document.pop(name)
        return {**document, "task_id": self.task_id}


def __getattr__(name):
    """Keep established task API imports while implementations own their modules."""
    if name in {"all_training_tasks", "load_task_matrix", "logical_tasks_for_group",
                "tasks_for_group", "validate_country_task_matrix", "validate_task_matrix"}:
        from . import matrix
        return getattr(matrix, name)
    if name in {"load_prepared_task", "prepare_tasks", "rebase_prepared_task"}:
        from . import preparation
        return getattr(preparation, name)
    raise AttributeError(name)
