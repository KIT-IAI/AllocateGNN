"""Path constants and boundary-safe path construction."""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath
import re

SMOKE_ROOT = "_smoke"
DIAGNOSTIC_ROOT = "_diagnostic"


class PathBoundaryError(ValueError):
    pass


def resolved(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def scoped_path(root: Path | str, *parts: str | Path) -> Path:
    boundary = resolved(root)
    candidate = boundary.joinpath(*parts).resolve()
    try:
        candidate.relative_to(boundary)
    except ValueError as exc:
        raise PathBoundaryError(f"path escapes boundary {boundary}: {candidate}") from exc
    return candidate


def portable_path(path: Path | str, root: Path | str) -> str:
    """Serialise ``path`` as a POSIX path relative to ``root`` for persisted records.

    Records never store machine-specific absolute locations; readers resolve the
    value against their own root, so moving a checkout does not break them.
    """

    boundary = resolved(root)
    candidate = resolved(path)
    try:
        return candidate.relative_to(boundary).as_posix()
    except ValueError as exc:
        raise PathBoundaryError(f"path escapes boundary {boundary}: {candidate}") from exc


_WINDOWS_ABSOLUTE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")


def case_study_root(root: Path | str) -> Path:
    """Locate live cases or the unchanged case tree inside an older snapshot."""
    base = Path(root)
    current = base / "casestudy"
    historical = base / "casestudy2"
    return historical if not current.exists() and historical.is_dir() else current


def resolve_case_path(path: Path | str) -> Path:
    """Resolve a case locator without modifying the record that supplied it."""
    candidate = Path(path)
    if candidate.exists():
        return candidate
    parts = candidate.parts
    for index, part in enumerate(parts):
        if part in {"casestudy", "casestudy2"}:
            alternate = "casestudy" if part == "casestudy2" else "casestudy2"
            located = Path(*parts[:index], alternate, *parts[index + 1:])
            if located.exists():
                return located
    return candidate


def resolve_recorded_path(value: Path | str, root: Path | str) -> Path:
    """Locate a path read from a persisted record, independent of the working directory.

    Current records hold POSIX paths relative to ``root`` (see ``portable_path``).
    Older records may hold absolute paths, possibly from another checkout location;
    one that no longer lies under ``root`` is re-rooted at the first component
    naming a top-level entry of ``root`` whose re-rooted file exists.
    """

    boundary = resolved(root)
    text = str(value)
    pure = PureWindowsPath(text) if _WINDOWS_ABSOLUTE.match(text) else PurePosixPath(text.replace("\\", "/"))
    if not pure.is_absolute():
        return resolve_case_path(scoped_path(boundary, *pure.parts))
    direct = Path(text)
    if direct.is_absolute():
        candidate = resolve_case_path(direct).resolve()
        if candidate.is_relative_to(boundary) and candidate.exists():
            return candidate
    entries = {child.name for child in boundary.iterdir()}
    parts = pure.parts[1:]
    for index, part in enumerate(parts):
        if part in entries or part in {"casestudy", "casestudy2"}:
            candidate = resolve_case_path(boundary.joinpath(*parts[index:]))
            if candidate.exists():
                return candidate
    raise PathBoundaryError(f"recorded path cannot be located under {boundary}: {text}")


def find_repo_root(start: Path | str | None = None) -> Path:
    """Find the checkout root without relying on the current working directory."""

    cursor = resolved(start or Path.cwd())
    if cursor.is_file():
        cursor = cursor.parent
    for candidate in (cursor, *cursor.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "data" / "metadata.toml").is_file():
            return candidate
    raise FileNotFoundError(f"cannot locate repository root from {cursor}")
