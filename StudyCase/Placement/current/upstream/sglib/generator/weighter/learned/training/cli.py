"""Command-line interface for prepared training workers and verification."""

from __future__ import annotations

import argparse
import json
from typing import Sequence
from .tasks import BACKENDS, DEVICES
from .preparation import load_prepared_task
from .verify import verify_task_artifacts
from .engine import run_training_task


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--task-json", required=True)
    run_parser.add_argument("--backend", choices=BACKENDS, required=True)
    run_parser.add_argument("--device", choices=DEVICES, default="auto")
    run_parser.add_argument("--repo-root")
    run_parser.add_argument("--results-root")
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--task-json", required=True)
    verify_parser.add_argument("--repo-root")
    verify_parser.add_argument("--results-root")
    args = parser.parse_args(argv)
    task = load_prepared_task(
        args.task_json,
        repo_root=args.repo_root,
        results_root=args.results_root,
    ) if args.repo_root is not None or args.results_root is not None else load_prepared_task(args.task_json)
    if args.command == "run":
        print(run_training_task(task, backend=args.backend, device=args.device))
    else:
        print(json.dumps(verify_task_artifacts(task), indent=2))
    return 0
