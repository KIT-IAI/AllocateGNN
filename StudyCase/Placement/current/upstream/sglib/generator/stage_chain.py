"""Explicit training and inference closure tools for a selected results root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from sglib.core.infra.artifacts import atomic_json
from sglib.core.infra.content_chain import (
    RECEIPT,
    derive_chain_closure,
    verify_chain,
)
from sglib.core.infra.hashing import sha256_file, sha256_json
from sglib.core.infra.paths import find_repo_root
from sglib.core.infra.terms import discover_country_profiles

from .weighter.learned.training.matrix import all_training_tasks, load_task_matrix, tasks_for_group
from .weighter.learned.training.preparation import load_prepared_task
from .weighter.learned.training.verify import verify_task_artifacts


TRAINING_CLOSURE_SCHEMA = "sg_generator_hpc_training_closure_v2"
LOCAL_PREREG_SCHEMA = "sg_generator_inference_preregistration_v2"
LOCAL_CLOSURE_SCHEMA = "sg_generator_hpc_inference_closure_v2"


class GeneratorStageChainError(RuntimeError):
    pass


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GeneratorStageChainError(f"missing or unreadable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise GeneratorStageChainError(f"JSON root is not an object: {path}")
    return value


def _context(repo_root: Path | str, results_root: Path | str):
    repo, results = Path(repo_root).resolve(), Path(results_root).resolve()
    matrix = load_task_matrix(
        repo / "casestudy/2_Generator/general/training_task_matrix_scientific_v2.csv"
    )
    profiles = discover_country_profiles(repo / "casestudy/config/countries")
    return repo, results, results / "2_Generator", matrix, profiles


def _safe(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=False)
    if path == root or root.resolve() not in path.parents:
        raise GeneratorStageChainError(f"path escapes execution root: {relative}")
    return path


def _inventory(
    repo_root: Path | str,
    results_root: Path | str,
    *,
    verify: bool = False,
) -> Iterable[dict[str, Any]]:
    """The authority-coordinate iterator used by preflight and closure."""

    repo, results, stage, matrix, profiles = _context(repo_root, results_root)
    for group in matrix:
        country_root = stage / profiles[group.country].directory
        index_path = country_root / "training/tasks" / group.task_group / "index.json"
        entries = {}
        if index_path.is_file():
            rows = _json(index_path).get("tasks", [])
            entries = {
                str(row.get("task_id")): row
                for row in rows
                if isinstance(row, Mapping)
            }
        expected = tasks_for_group(
            matrix, country=group.country, group=group.task_group
        )
        if entries and set(entries) != {item.task_id for item in expected}:
            raise GeneratorStageChainError(f"training index differs: {group.task_group}")
        for coordinate in expected:
            row = entries.get(coordinate.task_id)
            task_path = _safe(country_root, str(row["path"])) if row else None
            task_document = _json(task_path) if task_path and task_path.is_file() else None
            completion_path = (
                _safe(
                    results,
                    str(task_document["output_results_relative"])
                    + "/task_completion.json",
                )
                if task_document
                else None
            )
            record: dict[str, Any] = {
                "country": group.country,
                "directory": profiles[group.country].directory,
                "group": group.task_group,
                "task_id": coordinate.task_id,
                "task_path": task_path,
                "completion_path": completion_path,
                "prepared": bool(task_path and task_path.is_file()),
                "completed": bool(completion_path and completion_path.is_file()),
            }
            if verify:
                if not record["prepared"] or not record["completed"]:
                    raise GeneratorStageChainError(
                        f"training coordinate is incomplete: {coordinate.task_id}"
                    )
                task = load_prepared_task(
                    task_path, repo_root=repo, results_root=results
                )
                record.update(task=task, completion=verify_task_artifacts(task))
            yield record


def _paths(results_root: Path | str) -> tuple[Path, Path, Path]:
    stage = Path(results_root).resolve() / "2_Generator"
    return (
        stage / "_closures/hpc_training_v2.json",
        stage / "_preregistration/inference_v2.json",
        stage / "_closures/hpc_inference_v2.json",
    )


def training_preflight(repo_root: Path | str, results_root: Path | str) -> dict[str, Any]:
    records = list(_inventory(repo_root, results_root))
    stage = Path(results_root).resolve() / "2_Generator"
    groups = {(item["directory"], item["group"]) for item in records}
    verified = sum(
        (stage / directory / "training/verify" / f"{group}.json").is_file()
        for directory, group in groups
    )
    prepared = sum(item["prepared"] for item in records)
    completed = sum(item["completed"] for item in records)
    ready = (prepared, completed, verified) == (len(records), len(records), len(groups))
    return {
        "schema_version": "sg_generator_local_inference_preflight_v1",
        "status": "READY_FOR_TRAINING_CLOSURE" if ready else "WAITING_FOR_TRAINING_SYNC_VERIFY",
        "ready": ready,
        "expected_training_groups": len(groups),
        "expected_training_coordinates": len(records),
        "prepared_training_coordinates": prepared,
        "completed_training_coordinates": completed,
        "verified_training_groups": verified,
    }


def _immutable(document: Mapping[str, Any], path: Path) -> Path:
    if path.exists():
        if _json(path) != dict(document):
            raise GeneratorStageChainError(f"append-only artifact differs: {path}")
        return path
    return atomic_json(dict(document), path)


def _training_node(task: Any, completion: Mapping[str, Any]) -> dict[str, Any]:
    science = {
        name: getattr(task, name)
        for name in (
            "country", "group", "family", "config", "signal", "parameter",
            "value", "seed", "fold", "feature_set", "selector", "compute_threads",
        )
    }
    science.update(
        training_weighting_policy=completion["training_weighting_policy"],
        training_batch_size=completion["training_batch_size"],
        region_order=completion["region_order"],
        input_fingerprints=dict(sorted(task.input_fingerprints.items())),
        config_fingerprint=task.config_fingerprint,
        run_fingerprint=task.execution_identity["run_identity"]["run_fingerprint"],
    )
    return {
        "node_id": f"generator.training.{task.task_id}",
        "task_id": task.task_id,
        "country": task.country,
        "group": task.group,
        "scientific_parameters_sha256": sha256_json(science),
        "outputs": {
            name: value["sha256"]
            for name, value in sorted(completion["artifacts"].items())
        },
    }


def _group_verifications(records: Iterable[Mapping[str, Any]], stage: Path) -> list[dict[str, Any]]:
    groups = sorted({(item["directory"], item["group"]) for item in records})
    result = []
    for directory, group in groups:
        path = stage / directory / "training/verify" / f"{group}.json"
        if not path.is_file():
            raise GeneratorStageChainError(f"training group verify is missing: {group}")
        result.append({"group": group, "path": path.relative_to(stage).as_posix(), "sha256": sha256_file(path)})
    return result


def build_training_closure(repo_root: Path | str, results_root: Path | str) -> Path:
    records = list(_inventory(repo_root, results_root, verify=True))
    stage = Path(results_root).resolve() / "2_Generator"
    nodes = sorted(
        (_training_node(item["task"], item["completion"]) for item in records),
        key=lambda item: item["node_id"],
    )
    sources = sorted(
        (
            {
                "task_id": item["task_id"],
                "path": item["completion_path"].relative_to(stage).as_posix(),
                "sha256": sha256_file(item["completion_path"]),
            }
            for item in records
        ),
        key=lambda item: item["task_id"],
    )
    science = {"node_count": len(nodes), "nodes": nodes}
    backends = sorted({item["task"].execution_backend for item in records})
    document = {
        "schema_version": TRAINING_CLOSURE_SCHEMA,
        "status": "PASS",
        "execution_topology": {"backend": backends[0] if len(backends) == 1 else "mixed", "identity_binding": "observation_only"},
        "scientific_view": science,
        "scientific_fingerprint": sha256_json(science),
        "terminal_receipts": sources,
        "group_verifications": _group_verifications(records, stage),
    }
    return _immutable(document, _paths(results_root)[0])


def _validated_training_closure(repo_root: Path | str, results_root: Path | str):
    document = _json(_paths(results_root)[0])
    science, sources = document.get("scientific_view"), document.get("terminal_receipts")
    records = list(_inventory(repo_root, results_root, verify=True))
    stage = Path(results_root).resolve() / "2_Generator"
    current_nodes = sorted(
        (_training_node(item["task"], item["completion"]) for item in records),
        key=lambda item: item["node_id"],
    )
    if (
        document.get("schema_version") != TRAINING_CLOSURE_SCHEMA
        or document.get("status") != "PASS"
        or not isinstance(science, Mapping)
        or science.get("node_count") != len(records)
        or document.get("scientific_fingerprint") != sha256_json(science)
        or not isinstance(sources, list)
        or len(sources) != len(records)
        or science.get("nodes") != current_nodes
        or document.get("group_verifications") != _group_verifications(records, stage)
    ):
        raise GeneratorStageChainError("training closure differs")
    current = {item["task_id"]: item for item in _inventory(repo_root, results_root)}
    for source in sources:
        item = current.get(str(source.get("task_id")))
        path = item and item["completion_path"]
        if not path or not path.is_file() or sha256_file(path) != source.get("sha256"):
            raise GeneratorStageChainError(f"training receipt differs: {source.get('task_id')}")
    return document, records


def verify_training_closure(repo_root: Path | str, results_root: Path | str) -> dict[str, Any]:
    return _validated_training_closure(repo_root, results_root)[0]


def _group_tasks(country_root: Path, group: str, repo: Path, results: Path):
    from .weighter.learned.inference.preparation import load_inference_task

    index = _json(country_root / "inference/tasks" / group / "index.json")
    return [
        load_inference_task(
            _safe(country_root, str(row["path"])), repo_root=repo, results_root=results
        )
        for row in index.get("tasks", [])
    ]


def prepare_local_inference_preregistration(repo_root: Path | str, results_root: Path | str, *, backend: str = "local") -> Path:
    destination = _paths(results_root)[1]
    if destination.exists():
        verify_local_inference_preregistration(repo_root, results_root)
        return destination
    repo, results, stage, matrix, profiles = _context(repo_root, results_root)
    training, records = _validated_training_closure(repo, results)
    if list(stage.glob("*/inference/outputs/**/inference_completion.json")):
        raise GeneratorStageChainError("inference output predates preregistration")
    from .execution import prepare_inference_group
    from .weighter.learned.inference.identity import inference_code_identity

    parents = {
        item["task_id"]: item for item in training["scientific_view"]["nodes"]
    }
    commitments = []
    for group in matrix:
        root = stage / profiles[group.country].directory
        verified = {
            str(item["task_path"]): (item["task"], item["completion"])
            for item in records
            if item["group"] == group.task_group
        }
        prepare_inference_group(
            root,
            repo_root=repo,
            results_root=results,
            group=group.task_group,
            execution_backend=backend,
            verified_training=verified,
        )
        for task in _group_tasks(root, group.task_group, repo, results):
            commitment = verify_chain(task.chain_commitment)
            if commitment["inputs"]["checkpoint"] != parents[task.training_task_id]["outputs"]["model.pth"]:
                raise GeneratorStageChainError(f"training/inference link differs: {task.task_id}")
            commitments.append(commitment)
    commitments.sort(key=lambda item: item["node_id"])
    expected = len(all_training_tasks(matrix))
    if len(commitments) != expected or len({item["node_id"] for item in commitments}) != expected:
        raise GeneratorStageChainError("inference preregistration differs from authority coordinates")
    science = {
        "training_closure_scientific_fingerprint": training["scientific_fingerprint"],
        "coordinate_count": expected,
        "commitments": commitments,
    }
    document = {
        "schema_version": LOCAL_PREREG_SCHEMA,
        "status": "FROZEN",
        "execution_topology": {"backend": backend, "device": "cpu", "identity_binding": "observation_only"},
        "code_identity": inference_code_identity(repo),
        "scientific_view": science,
        "scientific_fingerprint": sha256_json(science),
    }
    return _immutable(document, destination)


def verify_local_inference_preregistration(repo_root: Path | str, results_root: Path | str) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    training = verify_training_closure(repo, results_root)
    document = _json(_paths(results_root)[1])
    science = document.get("scientific_view")
    expected = training["scientific_view"]["node_count"]
    commitments = science.get("commitments") if isinstance(science, Mapping) else None
    if (
        document.get("schema_version") != LOCAL_PREREG_SCHEMA
        or document.get("status") != "FROZEN"
        or not isinstance(commitments, list)
        or len(commitments) != expected
        or document.get("scientific_fingerprint") != sha256_json(science)
        or science.get("training_closure_scientific_fingerprint") != training["scientific_fingerprint"]
    ):
        raise GeneratorStageChainError("inference preregistration differs")
    normalized = [verify_chain(item) for item in commitments]
    if normalized != commitments or len({item["node_id"] for item in normalized}) != expected:
        raise GeneratorStageChainError("local inference commitments differ")
    from .weighter.learned.inference.identity import inference_code_identity

    current, frozen = inference_code_identity(repo), document.get("code_identity", {})
    for name in ("schema_version", "symbols", "code_sha256"):
        if current.get(name) != frozen.get(name):
            raise GeneratorStageChainError("inference code differs from freeze")
    return document


def _require_group(
    prereg: Mapping[str, Any], country_root: Path, group: str, repo: Path, results: Path
) -> list[Any]:
    tasks = _group_tasks(country_root, group, repo, results)
    frozen = {
        item["node_id"]: item
        for item in prereg["scientific_view"]["commitments"]
        if item["scientific_parameters"]["group"] == group
    }
    if {task.chain_commitment["node_id"]: task.chain_commitment for task in tasks} != frozen:
        raise GeneratorStageChainError(f"inference group differs from freeze: {group}")
    return tasks


def require_local_inference_group(
    repo_root: Path | str, results_root: Path | str, *, country: str, group: str
) -> None:
    repo, results, stage, _, profiles = _context(repo_root, results_root)
    prereg = verify_local_inference_preregistration(repo, results)
    if country not in profiles:
        raise GeneratorStageChainError(f"unknown country: {country}")
    _require_group(prereg, stage / profiles[country].directory, group, repo, results)


def _inference_receipts(repo: Path, results: Path, prereg: Mapping[str, Any]):
    _, _, stage, matrix, profiles = _context(repo, results)
    from .weighter.learned.inference.verify import verify_inference_artifacts

    receipts, sources = [], []
    for group in matrix:
        root = stage / profiles[group.country].directory
        for task in _require_group(prereg, root, group.task_group, repo, results):
            completion = verify_inference_artifacts(task)
            receipt = verify_chain(completion["chain"])
            if receipt["schema_version"] != RECEIPT:
                raise GeneratorStageChainError("inference completion lacks chain receipt")
            receipts.append(receipt)
            path = Path(task.output_path) / "inference_completion.json"
            sources.append({"task_id": task.task_id, "path": path.relative_to(stage).as_posix(), "sha256": sha256_file(path)})
    return receipts, sorted(sources, key=lambda item: item["task_id"])


def build_local_inference_closure(repo_root: Path | str, results_root: Path | str) -> Path:
    repo, results, _, _, _ = _context(repo_root, results_root)
    prereg = verify_local_inference_preregistration(repo, results)
    receipts, sources = _inference_receipts(repo, results, prereg)
    closure = derive_chain_closure(receipts)
    if closure["node_count"] != prereg["scientific_view"]["coordinate_count"]:
        raise GeneratorStageChainError("inference closure differs from registered coordinates")
    document = {
        "schema_version": LOCAL_CLOSURE_SCHEMA,
        "status": "PASS",
        "preregistration_scientific_fingerprint": prereg["scientific_fingerprint"],
        "chain_closure": closure,
        "terminal_receipts": sources,
    }
    return _immutable(document, _paths(results)[2])


def run_local_inference(
    repo_root: Path | str,
    results_root: Path | str,
    *,
    country: str | None = None,
    group: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    repo, results, stage, matrix, profiles = _context(repo_root, results_root)
    prereg = verify_local_inference_preregistration(repo, results)
    groups = [
        item for item in matrix
        if (country is None or item.country == country)
        and (group is None or item.task_group == group)
    ]
    if not groups:
        raise GeneratorStageChainError("inference filters select no group")
    selected = sum(len(tasks_for_group(matrix, country=item.country, group=item.task_group)) for item in groups)
    if dry_run:
        return {"schema_version": "sg_generator_local_inference_run_plan_v1", "status": "DRY_RUN", "coordinates": selected, "would_execute": False}
    from .execution import run_inference_group_local, verify_inference_group

    for item in groups:
        root = stage / profiles[item.country].directory
        _require_group(prereg, root, item.task_group, repo, results)
        run_inference_group_local(root, item.task_group)
        verify_inference_group(root, item.task_group)
    completed = len(list(stage.glob("*/inference/outputs/**/inference_completion.json")))
    expected = len(all_training_tasks(matrix))
    if completed == expected:
        build_local_inference_closure(repo, results)
    return {"schema_version": "sg_generator_local_inference_run_status_v1", "status": "PASS" if completed == expected else "PARTIAL", "selected_coordinates": selected, "completed_coordinates": completed, "expected_coordinates": expected}


def verify_local_inference_closure(repo_root: Path | str, results_root: Path | str) -> dict[str, Any]:
    repo, results = Path(repo_root).resolve(), Path(results_root).resolve()
    prereg = verify_local_inference_preregistration(repo, results)
    document = _json(_paths(results_root)[2])
    receipts, sources = _inference_receipts(repo, results, prereg)
    closure = derive_chain_closure(receipts)
    if (
        document.get("schema_version") != LOCAL_CLOSURE_SCHEMA
        or document.get("status") != "PASS"
        or closure.get("node_count") != prereg["scientific_view"]["coordinate_count"]
        or document.get("chain_closure") != closure
        or document.get("terminal_receipts") != sources
        or document.get("preregistration_scientific_fingerprint") != prereg["scientific_fingerprint"]
    ):
        raise GeneratorStageChainError("inference closure differs")
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "close-training", "freeze", "run", "close-inference"))
    parser.add_argument("--repo-root")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--backend", choices=("local", "hpc"), default="local")
    parser.add_argument("--country")
    parser.add_argument("--group")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    repo = Path(args.repo_root).resolve() if args.repo_root else find_repo_root(__file__)
    results = Path(args.results_root)
    results = results if results.is_absolute() else repo / results
    actions = {
        "preflight": lambda: training_preflight(repo, results),
        "close-training": lambda: {"status": "PASS", "path": str(build_training_closure(repo, results))},
        "freeze": lambda: {"status": "FROZEN", "path": str(prepare_local_inference_preregistration(repo, results, backend=args.backend))},
        "run": lambda: run_local_inference(repo, results, country=args.country, group=args.group, dry_run=args.dry_run),
        "close-inference": lambda: {"status": "PASS", "path": str(build_local_inference_closure(repo, results))},
    }
    print(f"backend={args.backend} results_root={results}", flush=True)
    document = actions[args.command]()
    print(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if document.get("ready", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
