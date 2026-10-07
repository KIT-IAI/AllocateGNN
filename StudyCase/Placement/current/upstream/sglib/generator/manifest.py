"""Map existing Generator registry units to the plan 003b file partitions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from sglib.core.infra.artifacts import atomic_json
from sglib.core.infra.leaf_manifest import build, read_current, verify

from .registry import GeneratorUnit


def _read_references(path: Path) -> dict[str, Any]:
    # Missing or damaged indexes/tasks are still hashed by the gate and fail
    # their training leaf; they cannot supply checkpoint ownership this time.
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return {}
    return document if isinstance(document, dict) else {}


def enumerate_leaves(
    root: Path,
    units: Mapping[str, GeneratorUnit],
    candidates: Mapping[str, Any],
) -> tuple[dict[str, list[str]], list[str]]:
    """Enumerate current files independently of any baseline manifest."""

    files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*") if path.is_file()
        and path != root / "manifest.json"
        and "_views" not in path.relative_to(root).parts
    }
    leaves: dict[str, set[str]] = {}

    def claim(leaf_id: str, *roots: str) -> None:
        selected = leaves.setdefault(leaf_id, set())
        for prefix in roots:
            selected.update(
                path for path in files if path == prefix or path.startswith(prefix + "/")
            )

    for unit in units.values():
        step, member = unit.step, unit.member
        if step == "inputs":
            claim("inputs", "inputs")
        elif step in {"static", "public_activity"}:
            claim("static", "static")
        elif step in {"train", "verify"}:
            leaf_id = f"train/{member}"
            claim(leaf_id, f"training/tasks/{member}",
                  f"training/receipts/{member}.json", f"training/verify/{member}.json")
            if step == "train":
                index = _read_references(root / f"training/tasks/{member}/index.json")
                for reference in index.get("tasks", []):
                    task = _read_references(root / reference["path"])
                    output = task.get("output_results_relative")
                    if not output:
                        continue
                    try:
                        relative = Path(output).relative_to(Path("2_Generator") / root.name)
                    except ValueError:
                        continue
                    if relative.parts and relative.parts[0] == "2_Weighter" and ".." not in relative.parts:
                        claim(leaf_id, relative.as_posix())
        elif step == "infer":
            claim(f"infer/{member}", f"inference/tasks/{member}",
                  f"inference/outputs/{member}", f"inference/receipts/{member}.json",
                  f"inference/verify/{member}.json")
        elif step == "materialize":
            labels = [str(item["label"]) for item in candidates["candidates"]
                      if item["family"] == member]
            claim(f"materialize/{member}", f"candidates/index_{member}.json",
                  *(f"candidates/{label}" for label in labels))
        elif step == "finalize":
            claim("finalize", "candidates/candidate_index.json", "candidates/candidate_qa_index.json")
        elif step == "sweeps":
            claim(f"sweeps/{member}", f"sweeps/{member}", f"sweeps/index_{member}.json")
        elif step in {"idr_fixed", "idr_matched", "civd"}:
            claim(step, step)
        elif step == "audit":
            claim("audit", "audit.json")
        else:
            raise ValueError(f"no manifest partition for Generator unit {unit.id}")
    if (root / "setup").is_dir():
        claim("setup", "setup")
    claimed = set().union(*leaves.values())
    return ({leaf_id: sorted(paths) for leaf_id, paths in sorted(leaves.items())},
            sorted(files - claimed))


def write_setup(ctx) -> dict[str, Any]:
    """Record the run setup of a writable root once, before its manifest baseline is built.

    A closed root keeps the setup it already carries; the record is verified,
    never rewritten, and a missing record of a closed root is reported as ABSENT.
    """

    from sglib.core.infra.setup_snapshot import dirty_allowed, verify_setup, write_setup as snapshot
    from .stage import code_checks, setup_config_files

    existing = verify_setup(ctx.root)
    if existing["status"] != "ABSENT" or ctx.read_only:
        print(f"setup {existing['status']}: {ctx.root / 'setup'}", flush=True)
        return existing
    snapshot(ctx.root, repo_root=ctx.repo, stage="2_Generator", country=ctx.country, profile=ctx.profile,
             config_files=setup_config_files(ctx.loaded, ctx.repo), code_checks=code_checks(ctx),
             allow_dirty=dirty_allowed(ctx.repo, ctx.results, ctx.profile))
    report = verify_setup(ctx.root)
    print(f"setup WRITTEN ({report['status']}): {ctx.root / 'setup'}", flush=True)
    return report


def run_gate(ctx) -> dict[str, Any]:
    """Compare a baseline, or generate it in a writable root (closed roots use views)."""

    from .stage import figures_root

    baseline_path = ctx.root / "manifest.json"
    baseline = (json.loads(baseline_path.read_text(encoding="utf-8"))
                if baseline_path.is_file() else None)
    current, unclaimed = enumerate_leaves(ctx.root, ctx.units, ctx.candidates)
    view = figures_root(ctx)
    if baseline is None:
        generated = build("2_Generator", ctx.country, read_current(current, ctx.root))
        atomic_json(generated, view / "manifest.json" if ctx.read_only else baseline_path)
        report = {
            "stage": "2_Generator",
            "country": ctx.country,
            "status": "GENERATED",
            "message": "首次生成，无比对",
            "baseline_root_fingerprint": None,
            "root_fingerprint": generated["root_fingerprint"],
            "leaves": {
                leaf_id: {"status": "GENERATED", "fingerprint": leaf["fingerprint"]}
                for leaf_id, leaf in generated["leaves"].items()
            },
            "unclaimed": unclaimed,
        }
    else:
        report = verify(baseline, current, ctx.root)
        report["unclaimed"] = unclaimed
    atomic_json(report, view / "gate.json")
    for leaf_id, leaf in report["leaves"].items():
        differences = " ".join(
            f"{kind}={leaf[kind]}" for kind in ("missing", "extra", "changed") if leaf.get(kind)
        )
        print(f"{leaf['status']} {leaf_id}" + (f" {differences}" if differences else ""))
    print(report.get("message", f"{report['status']} gate")
          + f"; root_fingerprint={report['root_fingerprint']}; unclaimed={len(unclaimed)}")
    return report
