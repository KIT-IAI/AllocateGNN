"""Public path validation shared by training and inference."""

from __future__ import annotations

from pathlib import Path
from sglib.core.infra.paths import resolve_case_path
from .errors import TaskMatrixError, TrainingTaskError


def safe_relative_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise TaskMatrixError(f"output_pattern must be a safe relative path: {value!r}")
    return path.as_posix()



def absolute_root(value: str | Path, name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise TrainingTaskError(f"{name} must be absolute")
    return path.resolve(strict=False)



def relative_to(path: str | Path, root: Path, name: str) -> str:
    absolute = Path(path).expanduser()
    if not absolute.is_absolute():
        raise TrainingTaskError(f"{name} must be absolute")
    absolute = absolute.resolve(strict=False)
    try:
        relative = absolute.relative_to(root)
    except ValueError as error:
        raise TrainingTaskError(f"{name} must remain within {root}") from error
    try:
        return safe_relative_path(relative.as_posix())
    except TaskMatrixError as error:
        raise TrainingTaskError(f"{name} is not a safe relative path") from error



def join_relative(root: Path, value: str, name: str) -> Path:
    try:
        relative = safe_relative_path(value)
    except TaskMatrixError as error:
        raise TrainingTaskError(f"{name} is not a safe relative path") from error
    result = resolve_case_path(root / relative).resolve(strict=False)
    parts = Path(relative).parts
    # The completed V3 tree was promoted from results/v3_final to results.
    # Retain the recorded locator, resolving only an absent historical target;
    # never search archived runs or substitute an already existing file.
    if not result.exists() and parts[:2] == ("results", "v3_final"):
        promoted = (root / "results" / Path(*parts[2:])).resolve(strict=False)
        if promoted.is_file():
            result = promoted
    if result == root or root not in result.parents:
        raise TrainingTaskError(f"{name} escapes its declared root")
    return result
