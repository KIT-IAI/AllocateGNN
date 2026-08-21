"""Smoke tests for the paper-facing StudyCase/Placement scripts."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def run_case(relative_path: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, relative_path, *arguments],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_british_reconstruction_script() -> None:
    completed = run_case("StudyCase/Placement/British/001_reconstruction.py")
    payload = json.loads(completed.stdout)
    assert payload["median_change_pct"] == pytest.approx(-26.182790015227475)
    assert set(payload["per_seed"]) == {"42", "123", "456"}


def test_australian_connection_script() -> None:
    completed = run_case("StudyCase/Placement/Australia/003_connection.py")
    payload = json.loads(completed.stdout)
    assert payload["median_change_pct"] == pytest.approx(-6.663739158049074)
    assert payload["p_signflip"] == pytest.approx(0.08203125)


def test_complete_case_entrypoint() -> None:
    completed = run_case("StudyCase/Placement/000_reproduce_all.py", "--verify")
    assert "verification=PASS" in completed.stdout
