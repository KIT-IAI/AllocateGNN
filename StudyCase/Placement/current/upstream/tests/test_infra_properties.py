"""Domain-free artifact writing, hashing and path-boundary behaviour of sglib.core.infra."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from sglib.core.infra.artifacts import atomic_csv, atomic_json
from sglib.core.infra.hashing import canonical_json, sha256_file, sha256_json
from sglib.core.infra.paths import PathBoundaryError, portable_path, resolve_recorded_path, scoped_path

pytestmark = pytest.mark.produce


def test_atomic_artifacts_and_canonical_hashing(tmp_path):
    document = {"z": 1, "a": [2, 3]}
    path = tmp_path / "nested" / "document.json"
    atomic_json(document, path)
    assert json.loads(path.read_text(encoding="utf-8")) == document
    assert sha256_json(document) == sha256_json({"a": [2, 3], "z": 1})
    assert len(sha256_file(path)) == 64
    table = tmp_path / "table.csv"
    atomic_csv(pd.DataFrame({"value": [1, 2]}), table)
    assert pd.read_csv(table)["value"].tolist() == [1, 2]
    assert canonical_json(document) == '{"a":[2,3],"z":1}'


def test_scoped_path_rejects_escape(tmp_path):
    assert scoped_path(tmp_path, "child.txt").parent == tmp_path.resolve()
    with pytest.raises(PathBoundaryError, match="escapes"):
        scoped_path(tmp_path, "..", "escape")


def test_portable_path_records_root_relative_posix_paths(tmp_path):
    target = tmp_path / "results" / "stage" / "file.json"
    assert portable_path(target, tmp_path) == "results/stage/file.json"
    assert (tmp_path / portable_path(target, tmp_path)).resolve() == target.resolve()
    with pytest.raises(PathBoundaryError, match="escapes"):
        portable_path(tmp_path.parent / "outside.json", tmp_path)


def test_recorded_paths_resolve_against_root_not_working_directory(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    target = root / "results" / "stage" / "file.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert resolve_recorded_path("results/stage/file.json", root) == target.resolve()
    assert resolve_recorded_path(str(target), root) == target.resolve()
    old_windows = "C:" + r"\old\checkout\results\stage\file.json"
    old_posix = "/old/checkout/results/stage/file.json"
    assert resolve_recorded_path(old_windows, root) == target.resolve()
    assert resolve_recorded_path(old_posix, root) == target.resolve()
    with pytest.raises(PathBoundaryError):
        resolve_recorded_path("../outside.json", root)
    with pytest.raises(PathBoundaryError):
        resolve_recorded_path("/old/checkout/results/stage/missing.json", root)


def test_case_rename_resolves_both_live_and_frozen_paths_without_rewriting(tmp_path):
    from sglib.core.infra.paths import case_study_root, resolve_case_path, resolve_recorded_path
    live = tmp_path / "live"
    frozen = tmp_path / "frozen"
    current = live / "casestudy/config/countries/uk.toml"
    historical = frozen / "casestudy2/config/countries/uk.toml"
    for path in (current, historical):
        path.parent.mkdir(parents=True)
        path.write_bytes(b"frozen configuration")
    assert case_study_root(live) == live / "casestudy"
    assert case_study_root(frozen) == frozen / "casestudy2"
    assert resolve_case_path(frozen / "casestudy/config/countries/uk.toml") == historical
    assert resolve_recorded_path("casestudy2/config/countries/uk.toml", live) == current
    assert resolve_recorded_path("/old/checkout/casestudy2/config/countries/uk.toml", live) == current
    assert historical.read_bytes() == b"frozen configuration"
