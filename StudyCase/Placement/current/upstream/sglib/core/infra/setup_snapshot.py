"""Setup snapshots: the run provenance a results root carries with itself.

A results root that is sealed (or about to be) records under ``setup/`` the
configuration files it actually read, the code commit, the environment and,
for every content-chain receipt it holds, whether the current code still
projects to the receipt's ``code_sha256``. Closed roots are then judged by
this record instead of by the evolving configuration in Git.

``setup/`` is run provenance: nothing in it enters scientific identity.

Layout::

    setup/
      setup.json        this record (schema ``sg_setup_snapshot_v1``)
      config/<repo-relative path>   configuration originals, same layout as the repository
      env.txt           dependency listing of the interpreter that wrote the snapshot
      dirty.patch       only when the worktree was not clean and that was allowed
      orchestration/    copies of private orchestration scripts (reconstructed setups)
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Callable, Iterable, Mapping

from .artifacts import atomic_json, atomic_text
from .content_chain import RECEIPT
from .hashing import sha256_file
from .paths import portable_path


SCHEMA_VERSION = "sg_setup_snapshot_v1"
SETUP_DIRECTORY = "setup"
CONFIG_DIRECTORY = "config"
ORCHESTRATION_DIRECTORY = "orchestration"
ENVIRONMENT_FILE = "env.txt"
PATCH_FILE = "dirty.patch"
RECORD_FILE = "setup.json"
SOURCE_SEALED = "sealed"


class SetupSnapshotError(RuntimeError):
    pass


def setup_root(root: Path | str) -> Path:
    return Path(root) / SETUP_DIRECTORY


def dirty_allowed(repo_root: Path | str, results_root: Path | str, profile: str) -> bool:
    """Formal sealing inside the repository's results tree requires a clean worktree.

    Smoke runs, and roots outside ``results/`` (test and scratch roots), may
    snapshot a dirty worktree and keep the diff as ``dirty.patch``.
    """

    if profile == "smoke":
        return True
    results = Path(results_root).resolve()
    return not results.is_relative_to(Path(repo_root).resolve() / "results")


def config_root(root: Path | str) -> Path | None:
    """The configuration snapshot of a root, when it carries one."""

    candidate = setup_root(root) / CONFIG_DIRECTORY
    return candidate if candidate.is_dir() else None


def _git(repo_root: Path, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments], cwd=repo_root, capture_output=True, text=True, encoding="utf-8", check=False,
        )
    except (OSError, ValueError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def git_state(repo_root: Path | str) -> dict[str, Any]:
    """Commit and cleanliness of the checkout; ``commit`` is None outside a repository."""

    repo = Path(repo_root).resolve()
    commit = _git(repo, "rev-parse", "HEAD")
    status = _git(repo, "status", "--porcelain", "--untracked-files=all")
    changed: list[str] = []
    untracked: list[str] = []
    for line in (status or "").splitlines():
        if not line.strip():
            continue
        marker, name = line[:2], line[3:].strip()
        (untracked if marker == "??" else changed).append(name)
    return {
        "commit": commit.strip() if commit else None,
        "dirty": bool(changed or untracked),
        "changed_files": sorted(changed),
        "untracked_files": sorted(untracked),
    }


def git_patch(repo_root: Path | str) -> str:
    return _git(Path(repo_root).resolve(), "diff", "HEAD", "--no-color") or ""


def environment_listing() -> str:
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, encoding="utf-8", check=False,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            return "\n".join(sorted(completed.stdout.splitlines())) + "\n"
    except (OSError, ValueError):
        pass
    from importlib import metadata

    rows = sorted(f"{item.metadata['Name']}=={item.version}" for item in metadata.distributions())
    return "\n".join(rows) + "\n"


def iter_receipts(root: Path | str):
    """Yield standalone and embedded receipts, including inference completions.

    The JSON pointer identifies an embedded receipt without altering its owner.
    Snapshot copies are excluded because they describe a different production.
    """
    base = Path(root)

    def embedded(value, pointer=""):
        if isinstance(value, dict):
            if value.get("schema_version") == RECEIPT:
                yield pointer, value
                return
            for key, item in value.items():
                escaped = str(key).replace("~", "~0").replace("/", "~1")
                yield from embedded(item, pointer + "/" + escaped)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                yield from embedded(item, pointer + "/" + str(index))

    for path in sorted(base.rglob("*.json")):
        if SETUP_DIRECTORY in path.relative_to(base).parts:
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            continue
        for pointer, receipt in embedded(document):
            yield path, pointer, receipt


def receipt_code_checks(root: Path | str, current_code: Callable[[str, Mapping[str, Any]], str | None]) -> list[dict[str, Any]]:
    """Compare every content-chain receipt under ``root`` with the current code projection.

    ``current_code(node_id, receipt)`` returns the projection the current code
    yields for that node, or None when the node's code is not projectable from
    the public packages. Nothing is silent: each receipt yields one row.
    """

    rows = []
    base = Path(root)
    for path, pointer, document in iter_receipts(base):
        node_id = str(document.get("node_id", ""))
        recorded = str(document.get("commitment", {}).get("code_sha256", ""))
        current = current_code(node_id, document)
        rows.append({
            "node_id": node_id,
            "receipt": path.relative_to(base).as_posix() + ("#" + pointer if pointer else ""),
            "receipt_code_sha256": recorded,
            "current_code_sha256": current,
            "match": None if current is None else current == recorded,
        })
    return rows


def _summarize(checks: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(checks)
    mismatched = sorted(row["node_id"] for row in rows if row["match"] is False)
    unknown = sorted(row["node_id"] for row in rows if row["match"] is None)
    return {
        "checked": len(rows),
        "matching": sum(1 for row in rows if row["match"] is True),
        "mismatched": mismatched,
        "not_projectable": unknown,
    }


def write_setup(
    root: Path | str,
    *,
    repo_root: Path | str,
    stage: str,
    country: str,
    profile: str,
    config_files: Iterable[Path | str],
    source: str = SOURCE_SEALED,
    allow_dirty: bool = False,
    code_checks: Iterable[Mapping[str, Any]] | None = None,
    orchestration: Mapping[str, Path | str] | None = None,
    notes: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``root/setup`` once; refuse an existing snapshot.

    ``config_files`` are repository-relative (or absolute inside the
    repository) paths of the configuration files this stage and country
    actually read; they are copied byte for byte under ``setup/config`` in the
    repository layout. ``orchestration`` maps a name under ``setup/orchestration``
    to the script it copies. A formal snapshot refuses a dirty worktree;
    ``allow_dirty`` (smoke) stores the diff as ``dirty.patch`` instead.
    """

    repo = Path(repo_root).resolve()
    target = setup_root(root)
    if target.exists():
        raise SetupSnapshotError(f"setup snapshot already exists: {target}")
    state = git_state(repo)
    if state["dirty"] and not allow_dirty:
        raise SetupSnapshotError(
            "sealing requires a clean worktree; commit or stash first "
            f"(changed={state['changed_files']}, untracked={state['untracked_files']})"
        )
    files: dict[str, str] = {}
    configs: list[str] = []
    for item in config_files:
        source_path = Path(item)
        if not source_path.is_absolute():
            source_path = repo / source_path
        relative = portable_path(source_path, repo)
        if not source_path.is_file():
            raise SetupSnapshotError(f"configuration file is missing: {relative}")
        destination = target / CONFIG_DIRECTORY / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)
        files[f"{CONFIG_DIRECTORY}/{relative}"] = sha256_file(destination)
        configs.append(relative)
    environment = environment_listing()
    atomic_text(target / ENVIRONMENT_FILE, environment)
    files[ENVIRONMENT_FILE] = sha256_file(target / ENVIRONMENT_FILE)
    if state["dirty"]:
        atomic_text(target / PATCH_FILE, git_patch(repo))
        files[PATCH_FILE] = sha256_file(target / PATCH_FILE)
    scripts: list[dict[str, str]] = []
    for name, origin in (orchestration or {}).items():
        origin_path = Path(origin)
        if not origin_path.is_absolute():
            origin_path = repo / origin_path
        if not origin_path.is_file():
            raise SetupSnapshotError(f"orchestration script is missing: {origin_path}")
        destination = target / ORCHESTRATION_DIRECTORY / str(name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin_path, destination)
        key = f"{ORCHESTRATION_DIRECTORY}/{name}"
        files[key] = sha256_file(destination)
        scripts.append({"path": key, "origin": portable_path(origin_path, repo), "sha256": files[key]})
    checks = [dict(row) for row in (code_checks or [])]
    record = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "country": country,
        "profile": profile,
        "source": source,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": state,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies_sha256": files[ENVIRONMENT_FILE],
        "config_files": sorted(configs),
        "orchestration": scripts,
        "files": dict(sorted(files.items())),
        "code_projection": {**_summarize(checks), "checks": checks},
        "notes": dict(notes or {}),
    }
    atomic_json(record, target / RECORD_FILE)
    return target


def read_setup(root: Path | str) -> dict[str, Any] | None:
    path = setup_root(root) / RECORD_FILE
    if not path.is_file():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION:
        raise SetupSnapshotError(f"unsupported setup snapshot: {path}")
    return document


def verify_setup(root: Path | str) -> dict[str, Any]:
    """Compare the files under ``setup/`` with the hashes recorded in ``setup.json``."""

    target = setup_root(root)
    if not target.is_dir():
        return {"status": "ABSENT", "missing": [], "changed": [], "extra": []}
    record = read_setup(root)
    if record is None:
        return {"status": "FAIL", "missing": [RECORD_FILE], "changed": [], "extra": []}
    expected = dict(record.get("files", {}))
    present = {
        path.relative_to(target).as_posix()
        for path in target.rglob("*") if path.is_file() and path.name != RECORD_FILE
    }
    missing = sorted(set(expected) - present)
    extra = sorted(present - set(expected))
    changed = sorted(name for name in set(expected) & present if sha256_file(target / name) != expected[name])
    status = "PASS" if not (missing or extra or changed) else "FAIL"
    return {"status": status, "missing": missing, "changed": changed, "extra": extra,
            "source": record.get("source"), "commit": record.get("git", {}).get("commit")}


def setup_files(root: Path | str) -> list[str]:
    """Root-relative paths of every file under ``setup/`` (empty when absent)."""

    target = setup_root(root)
    if not target.is_dir():
        return []
    return sorted(path.relative_to(Path(root)).as_posix() for path in target.rglob("*") if path.is_file())


__all__ = [
    "CONFIG_DIRECTORY",
    "SCHEMA_VERSION",
    "SETUP_DIRECTORY",
    "SOURCE_SEALED",
    "SetupSnapshotError",
    "config_root",
    "environment_listing",
    "git_patch",
    "git_state",
    "read_setup",
    "receipt_code_checks",
    "setup_files",
    "setup_root",
    "verify_setup",
    "write_setup",
]
