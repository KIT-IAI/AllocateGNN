"""Leaf partitions of the Experiment root for the 07 hash gate (plan 006a §4: leaf = unit)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sglib.core.infra.artifacts import atomic_json
from sglib.core.infra.leaf_manifest import build, read_current, verify


def enumerate_leaves(root: Path, regions: list[str]) -> tuple[dict[str, list[str]], list[str]]:
    """Enumerate current files independently of any baseline manifest."""

    files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
             and path != root / "manifest.json" and "_views" not in path.relative_to(root).parts}
    leaves: dict[str, set[str]] = {}

    def claim(leaf_id: str, *roots: str) -> None:
        selected = leaves.setdefault(leaf_id, set())
        for prefix in roots:
            selected.update(path for path in files if path == prefix or path.startswith(prefix + "/"))

    for member in ("connection", "planning_pool"):
        claim(f"preflight/{member}", f"preflight/{member}")
    for region in regions:
        claim(f"observe/{region}", f"observations/{region}")
        claim(f"planning/{region}", f"planning/{region}")
    claim("bounds", "bounds")
    claim("defense", "defense")
    claim("audit", "audit.json")
    if (root / "setup").is_dir():
        claim("setup", "setup")
    claimed = set().union(*leaves.values())
    return ({leaf_id: sorted(paths) for leaf_id, paths in sorted(leaves.items())}, sorted(files - claimed))


def write_setup(ctx) -> dict[str, Any]:
    """Record the run setup of a writable root once, before its manifest baseline is built."""

    from sglib.core.infra.setup_snapshot import dirty_allowed, verify_setup, write_setup as snapshot
    from .stage import code_checks, setup_config_files

    existing = verify_setup(ctx.root)
    if existing["status"] != "ABSENT" or ctx.read_only:
        print(f"setup {existing['status']}: {ctx.root / 'setup'}", flush=True)
        return existing
    snapshot(ctx.root, repo_root=ctx.repo, stage="3_Experiment", country=ctx.country, profile=ctx.profile,
             config_files=setup_config_files(ctx), code_checks=code_checks(ctx),
             allow_dirty=dirty_allowed(ctx.repo, ctx.results, ctx.profile))
    report = verify_setup(ctx.root)
    print(f"setup WRITTEN ({report['status']}): {ctx.root / 'setup'}", flush=True)
    return report


def run_gate(ctx) -> dict[str, Any]:
    """Compare a baseline, or generate it (writable root: into the root; closed root: into views)."""

    from .stage import figures_root

    baseline_path = ctx.root / "manifest.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.is_file() else None
    current, unclaimed = enumerate_leaves(ctx.root, ctx.loaded.values["regions"])
    view = figures_root(ctx)
    if baseline is None:
        if ctx.read_only and not (ctx.root / "audit.json").is_file():
            raise RuntimeError("closed root without a manifest: place the generated audit.json into the root first, "
                               "otherwise the baseline would record an empty audit leaf")
        generated = build("3_Experiment", ctx.country, read_current(current, ctx.root))
        atomic_json(generated, view / "manifest.json" if ctx.read_only else baseline_path)
        report = {"stage": "3_Experiment", "country": ctx.country, "status": "GENERATED", "message": "首次生成，无比对",
                  "baseline_root_fingerprint": None, "root_fingerprint": generated["root_fingerprint"],
                  "leaves": {leaf_id: {"status": "GENERATED", "fingerprint": leaf["fingerprint"]} for leaf_id, leaf in generated["leaves"].items()},
                  "unclaimed": unclaimed}
    else:
        report = verify(baseline, current, ctx.root)
        report["unclaimed"] = unclaimed
    atomic_json(report, view / "gate.json")
    for leaf_id, leaf in report["leaves"].items():
        differences = " ".join(f"{kind}={leaf[kind]}" for kind in ("missing", "extra", "changed") if leaf.get(kind))
        print(f"{leaf['status']} {leaf_id}" + (f" {differences}" if differences else ""))
    print(report.get("message", f"{report['status']} gate") + f"; root_fingerprint={report['root_fingerprint']}; unclaimed={len(unclaimed)}")
    return report
