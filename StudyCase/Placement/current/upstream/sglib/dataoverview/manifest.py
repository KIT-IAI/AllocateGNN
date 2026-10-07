"""Adapt existing inventory records without expanding their coverage."""

from __future__ import annotations

import json
from typing import Any, Mapping

from sglib.core.infra.artifacts import atomic_json
from sglib.core.infra.leaf_manifest import build, verify


def from_inventory(inventory: Mapping[str, Any]) -> dict[str, Any]:
    """Build the single leaf exclusively from the recorded artifacts."""

    return build("1_DataOverview", inventory["country"], {"inventory": inventory["artifacts"]})


def current_paths(inventory: Mapping[str, Any]) -> dict[str, list[str]]:
    """Declare precisely the inventory coverage; never scan for extras."""

    return {"inventory": [item["path"] for item in inventory["artifacts"]]}


def setup_config_files(context) -> list[str]:
    """Repository-relative configuration files a DataOverview country actually reads."""

    repo = context.repo_root
    code = context.country_code
    directory = context.merged["country"]["directory"]
    stage = repo / "casestudy/1_DataOverview"
    files = {stage / "general/general.toml", repo / "casestudy/config/countries" / f"{code}.toml",
             repo / "data/metadata.toml", *stage.joinpath(directory).glob("*.toml")}
    if (stage / directory / "admission_contract.toml").is_file():
        files.add(repo / "casestudy/config/authority/engineering_admission.toml")
    return sorted(path.relative_to(repo).as_posix() for path in files if path.is_file())


def write_setup(context) -> dict[str, Any]:
    """Record the run setup of a country root once; an existing record is verified, never rewritten."""

    from sglib.core.infra.setup_snapshot import dirty_allowed, verify_setup, write_setup as snapshot

    root = context.repo_root / "results/1_DataOverview" / context.merged["country"]["directory"]
    existing = verify_setup(root)
    if existing["status"] != "ABSENT":
        print(f"setup {existing['status']}: {root / 'setup'}", flush=True)
        return existing
    if not (root / "data_inventory.json").is_file():
        raise RuntimeError("write the inventory before recording the DataOverview setup")
    snapshot(root, repo_root=context.repo_root, stage="1_DataOverview", country=context.country_code,
             profile="formal", config_files=setup_config_files(context),
             allow_dirty=dirty_allowed(context.repo_root, context.repo_root / "results", "formal"))
    report = verify_setup(root)
    print(f"setup WRITTEN ({report['status']}): {root / 'setup'}", flush=True)
    return report


def run_gate(context) -> dict[str, Any]:
    """Compare the existing inventory, or generate only a view and stop.

    A recorded ``setup/`` beside the inventory is verified as its own leaf.
    """

    from sglib.core.infra.setup_snapshot import verify_setup

    directory = context.merged["country"]["directory"]
    root = context.repo_root / "results/1_DataOverview" / directory
    baseline_path = root / "data_inventory.json"
    view = context.repo_root / "results/_views/1_DataOverview" / directory
    if baseline_path.is_file():
        inventory = json.loads(baseline_path.read_text(encoding="utf-8"))
        report = verify(from_inventory(inventory), current_paths(inventory), context.repo_root)
        if (root / "setup").is_dir():
            setup = verify_setup(root)
            report["leaves"]["setup"] = {"status": setup["status"], "missing": setup["missing"],
                                         "extra": setup["extra"], "changed": setup["changed"]}
            if setup["status"] != "PASS":
                report["status"] = "FAIL"
    else:
        from .overview.inventory import write_inventory

        generated_path = write_inventory(context, output_path=view / "data_inventory.json")
        inventory = json.loads(generated_path.read_text(encoding="utf-8"))
        generated = from_inventory(inventory)
        report = {
            "stage": "1_DataOverview",
            "country": context.country_code,
            "status": "GENERATED",
            "message": "首次生成，无比对",
            "baseline_root_fingerprint": None,
            "root_fingerprint": generated["root_fingerprint"],
            "leaves": {
                "inventory": {"status": "GENERATED", "fingerprint": generated["leaves"]["inventory"]["fingerprint"]},
            },
            "unclaimed": [],
        }
    atomic_json(report, view / "gate.json")
    for leaf_id, leaf in report["leaves"].items():
        differences = " ".join(
            f"{kind}={leaf[kind]}" for kind in ("missing", "changed") if leaf.get(kind)
        )
        print(f"{leaf['status']} {leaf_id}" + (f" {differences}" if differences else ""))
    print(report.get("message", f"{report['status']} gate")
          + f"; root_fingerprint={report['root_fingerprint']}")
    return report
