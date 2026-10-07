"""Repository gates for the active sglib + casestudy surface.

Responsibilities: active sources never load the retired packages (statically
or dynamically) and those packages stay removed; every five-stage numbered
entry exists; the source roots hold no result artifacts beyond registered
authorities; and current engineering documents remain indexed with working local links.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re

import pytest

pytestmark = pytest.mark.gate


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = REPO_ROOT / "sglib"
CASE_STUDY = REPO_ROOT / "casestudy"
ACTIVE_ROOTS = (PACKAGE, CASE_STUDY)

LEGACY_MODULE = re.compile(r"^(?:SpatialGranularity|CaseStudy|data\.fetcher)(?:\.|$)")
LEGACY_PATH = re.compile(r"(?:^|[/\\])(?:SpatialGranularity|CaseStudy|data[/\\]fetcher)(?:[/\\]|$)")
LOADERS = {
    "__import__",
    "import_module",
    "run_module",
    "run_path",
    "SourceFileLoader",
    "spec_from_file_location",
}

FOUR_COUNTRIES = ("1_UK", "2_AU", "4_NL", "5_NZ")
STAGE_ENTRIES = {
    "1_DataOverview": (
        ("1_UK", "2_AU", "3_DE", "4_NL", "5_NZ"),
        ("01_download.py", "02_regions_stations.ipynb", "03_grid.ipynb", "04_features.py",
         "05_features_overview.ipynb", "06_inventory.ipynb"),
    ),
    "2_Generator": (
        FOUR_COUNTRIES,
        ("01_inputs.ipynb", "02_static.ipynb", "03_train.py", "04_infer.py", "05_materialize.ipynb",
         "06_allocators.ipynb", "07_audit_handoff.ipynb"),
    ),
    "3_Experiment": (
        FOUR_COUNTRIES,
        ("01_preflight.ipynb", "02_observe.py", "03_planning.py", "04_bounds.py", "05_defense.py",
         "06_audit.ipynb"),
    ),
    "4_Analysis": (FOUR_COUNTRIES, ("01_core.py", "02_support.py", "03_overview.ipynb")),
}
STAGE_LEVEL_ENTRIES = (
    "1_DataOverview/general/00_data_matrix.ipynb",
    "1_DataOverview/run_all.py",
    "2_Generator/run_all.py",
    "4_Analysis/9_CrossCountry/01_synthesis.py",
    "4_Analysis/9_CrossCountry/02_overview.ipynb",
    "5_Report/01_render.py",
    "5_Report/02_audit.ipynb",
)

GENERATED_DOCUMENTS = {"casestudy/1_DataOverview/general/data_matrix.md"}

RESULT_LIKE_SUFFIXES = {
    ".csv", ".arrow", ".bin", ".ckpt", ".dat", ".db", ".dbf", ".feather", ".geojson", ".gpkg", ".gz",
    ".h5", ".hdf5", ".html", ".joblib", ".json", ".log", ".npy", ".npz", ".nc", ".onnx", ".out",
    ".parquet", ".pdf", ".pkl", ".pickle", ".png", ".prj", ".pt", ".pth", ".safetensors", ".sav",
    ".shp", ".sqlite", ".sqlite3", ".svg", ".tar", ".tif", ".tiff", ".xls", ".xlsx", ".zip",
}
REGISTERED_AUTHORITIES = (
    re.compile(r"casestudy/2_Generator/general/(?:candidate_registry|idr_mainline_contract)\.json"),
    re.compile(r"casestudy/2_Generator/general/training_task_matrix(?:_scientific_v2)?\.csv"),
    re.compile(r"casestudy/(?:3_Experiment|4_Analysis)/(?:general|\d_[A-Za-z]+)/registrations\.json"),
    re.compile(r"casestudy/config/authority/evidence/[^/]+\.json"),
    re.compile(r"sglib/dataoverview/processing/derive/au/au_registry/[^/]+\.json"),
)


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _string_value(node: ast.AST | None, strings: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return strings.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _string_value(node.left, strings)
        right = _string_value(node.right, strings)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.JoinedStr):
        parts = [value.value for value in node.values if isinstance(value, ast.Constant)]
        return "".join(parts) if len(parts) == len(node.values) else None
    return None


def _constant_strings(tree: ast.AST) -> dict[str, str]:
    strings: dict[str, str] = {}
    assignments = [node for node in ast.walk(tree) if isinstance(node, (ast.Assign, ast.AnnAssign))]
    for _ in range(len(assignments) + 1):
        changed = False
        for node in assignments:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = _string_value(node.value, strings)
            for target in targets:
                if isinstance(target, ast.Name) and value is not None and strings.get(target.id) != value:
                    strings[target.id] = value
                    changed = True
        if not changed:
            break
    return strings


def _call_name(node: ast.Call) -> str:
    function = node.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return ""


def _modifies_sys_path(node: ast.Call) -> bool:
    function = node.func
    return (
        isinstance(function, ast.Attribute)
        and function.attr in {"insert", "append", "extend"}
        and isinstance(function.value, ast.Attribute)
        and function.value.attr == "path"
        and isinstance(function.value.value, ast.Name)
        and function.value.value.id == "sys"
    )


def _legacy_loads(source: str, label: str) -> list[str]:
    tree = ast.parse(source, filename=label)
    strings = _constant_strings(tree)
    loaders = set(LOADERS)
    failures = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""] if not node.level else []
            loaders.update(alias.asname or alias.name for alias in node.names if alias.name in LOADERS)
        else:
            continue
        failures.extend(f"{label}:{node.lineno}: imports {name}" for name in names if LEGACY_MODULE.match(name))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not (_call_name(node) in loaders or _modifies_sys_path(node)):
            continue
        for argument in node.args:
            value = _string_value(argument, strings)
            if value is not None and (LEGACY_MODULE.match(value) or LEGACY_PATH.search(value)):
                failures.append(f"{label}:{node.lineno}: loads {value}")
    return failures


def _notebook_code(path: Path) -> str:
    cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    lines = []
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        text = "".join(source) if isinstance(source, list) else source
        lines.extend(line for line in text.splitlines() if not line.lstrip().startswith(("%", "!")))
    return "\n".join(lines)


def test_active_sources_never_load_retired_packages():
    failures = []
    for root in ACTIVE_ROOTS:
        for path in sorted(root.rglob("*.py")):
            failures.extend(_legacy_loads(path.read_text(encoding="utf-8-sig"), _relative(path)))
        for path in sorted(root.rglob("*.ipynb")):
            failures.extend(_legacy_loads(_notebook_code(path), _relative(path)))
    assert not failures, "active sources load retired packages:\n" + "\n".join(failures)


ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/](?![\\/])|/mnt/[a-z]/|/hkfs/|/home/[A-Za-z]|\\\\\?\\")
PORTABLE_SCAN_ROOTS = (PACKAGE, CASE_STUDY, REPO_ROOT / "docs", REPO_ROOT / "tests")
PORTABLE_SCAN_FILES = ("README.md", "pyproject.toml", "pytest.ini", "data/README.md", "data/metadata.toml")
PORTABLE_SUFFIXES = {".py", ".ipynb", ".toml", ".json", ".csv", ".md", ".ini", ".sh", ".ps1"}


def test_sources_configs_and_docs_hold_no_absolute_paths():
    files = [
        path
        for root in PORTABLE_SCAN_ROOTS
        for path in root.rglob("*")
        if path.is_file() and path.suffix in PORTABLE_SUFFIXES and "__pycache__" not in path.parts
    ]
    files += [REPO_ROOT / name for name in PORTABLE_SCAN_FILES]
    this_gate = Path(__file__).resolve()
    offenders = [
        f"{_relative(path)}:{text.count(chr(10), 0, match.start()) + 1}: {match.group(0)}"
        for path in files
        if path.resolve() != this_gate
        for text in [path.read_text(encoding="utf-8", errors="replace")]
        for match in ABSOLUTE_PATH.finditer(text)
    ]
    assert not offenders, "absolute paths must be written relative to a root:\n" + "\n".join(offenders)


ARCHIVE_REFERENCE = re.compile(r"results[/\\]_archive")
ARCHIVE_SCAN_ROOTS = (PACKAGE, CASE_STUDY, REPO_ROOT / "tests")


def test_active_code_never_reads_the_results_archive():
    # The results archive holds superseded batches as records only: sources, configs,
    # notebooks and tests must not name it, and results roots inside it are refused
    # at resolution. Human READMEs may point readers to archived records.
    this_gate = Path(__file__).resolve()
    offenders = [
        f"{_relative(path)}:{text.count(chr(10), 0, match.start()) + 1}: {match.group(0)}"
        for root in ARCHIVE_SCAN_ROOTS
        for path in root.rglob("*")
        if path.is_file() and path.suffix in PORTABLE_SUFFIXES - {".md"} and "__pycache__" not in path.parts
        and path.resolve() != this_gate
        for text in [path.read_text(encoding="utf-8", errors="replace")]
        for match in ARCHIVE_REFERENCE.finditer(text)
    ]
    assert not offenders, "active code references the results archive:\n" + "\n".join(offenders)


HISTORICAL_COUNT_PIN = re.compile(
    r"^\s*(?:expected_[a-z0-9_]+|formal_truth_rows|section_lineage_rows)\s*=|^\s*\[regions\.checks\]",
    re.MULTILINE,
)


def test_configs_hold_no_historical_count_pins():
    # Counts of one materialisation are recorded by inventories and gates of
    # each rebuild; configurations register identities (IDs, order), not counts.
    offenders = [
        f"{_relative(path)}:{text.count(chr(10), 0, match.start()) + 1}: {match.group(0).strip()}"
        for path in sorted(CASE_STUDY.rglob("*.toml"))
        for text in [path.read_text(encoding="utf-8")]
        for match in HISTORICAL_COUNT_PIN.finditer(text)
    ]
    assert not offenders, "historical count pins in configurations:\n" + "\n".join(offenders)


def test_retired_package_directories_are_absent():
    for name in ("SpatialGranularity", "CaseStudy", "data/fetcher", "memory-bank", "paper", "_quarto.yml", "index.md"):
        parts = Path(name).parts
        parent = REPO_ROOT.joinpath(*parts[:-1])
        assert not parent.is_dir() or parts[-1] not in {p.name for p in parent.iterdir()}, name


def _assert_documentation_index_and_links(repo_root: Path):
    from urllib.parse import unquote, urlsplit

    documented_roots = (repo_root / "docs", repo_root / "sglib", repo_root / "casestudy", repo_root / "tests", repo_root / "examples")
    found = {path.resolve() for root in documented_roots for path in root.rglob("*.md")}
    found |= {path.resolve() for path in repo_root.glob("*.md")}
    found |= {path.resolve() for path in (repo_root / "data").glob("*.md")}
    found -= {(repo_root / name).resolve() for name in GENERATED_DOCUMENTS}
    reached, pending, failures = set(), [(repo_root / "README.md").resolve()], []
    while pending:
        path = pending.pop()
        if path in reached:
            continue
        reached.add(path)
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            target = target.strip().split(' "', 1)[0].strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or not parsed.path:
                continue
            linked = (path.parent / unquote(parsed.path)).resolve()
            if not linked.exists():
                failures.append(f"{path.relative_to(repo_root)}: {target}")
            elif linked in found and linked not in reached:
                pending.append(linked)
    assert not failures, "broken local documentation links: " + repr(failures)
    assert not found - reached, "unindexed engineering documents: " + repr(sorted(str(p.relative_to(repo_root)) for p in found - reached))



def test_engineering_documents_are_indexed_and_local_links_exist():
    _assert_documentation_index_and_links(REPO_ROOT)


def test_public_documentation_gate_passes_without_private_data_or_results(tmp_path):
    """Exercise the real index/link checks in a checkout of current tracked files only."""
    import shutil
    import subprocess

    if not (REPO_ROOT / ".git").exists():
        pytest.skip("tracked-file export requires Git; the direct documentation gate also runs in source archives")
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=REPO_ROOT).split(b"\0")
    clean = tmp_path / "public-checkout"
    clean.mkdir()
    for entry in filter(None, tracked):
        relative = Path(entry.decode("utf-8"))
        source, target = REPO_ROOT / relative, clean / relative
        assert source.is_file(), f"tracked source is absent: {relative}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    assert not (clean / "private").exists()
    assert not (clean / "results").exists()
    assert not (clean / "data/datasets").exists()
    assert not (clean / "data/secrets").exists()
    _assert_documentation_index_and_links(clean)
    # The isolated check must still reject a genuine missing local target.
    with (clean / "README.md").open("a", encoding="utf-8") as handle:
        handle.write("\n[missing document](missing-document.md)\n")
    with pytest.raises(AssertionError, match="broken local documentation links"):
        _assert_documentation_index_and_links(clean)


def test_five_stage_numbered_entries_are_complete():
    missing = [
        f"{stage}/{country}/{entry}"
        for stage, (countries, entries) in STAGE_ENTRIES.items()
        for country in countries
        for entry in entries
        if not (CASE_STUDY / stage / country / entry).is_file()
    ]
    missing += [entry for entry in STAGE_LEVEL_ENTRIES if not (CASE_STUDY / entry).is_file()]
    missing += [
        f"{stage}/README.md"
        for stage in (*STAGE_ENTRIES, "5_Report")
        if not (CASE_STUDY / stage / "README.md").is_file()
    ]
    assert not missing, f"missing five-stage entries: {missing}"
    assert not (CASE_STUDY / "2_Generator" / "3_DE").exists(), "DE must stay DataOverview-only"


def test_source_roots_hold_only_registered_authorities():
    violations = [
        _relative(path)
        for root in ACTIVE_ROOTS
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix.lower() in RESULT_LIKE_SUFFIXES
        and not any(pattern.fullmatch(_relative(path)) for pattern in REGISTERED_AUTHORITIES)
    ]
    assert not violations, "result-like files in source roots:\n" + "\n".join(violations)


@pytest.mark.parametrize(
    "source",
    [
        "from CaseStudy.hidden import run",
        "import SpatialGranularity.Weighter as weighter",
        'import importlib\nname = "Spatial" + "Granularity.Weighter"\nimportlib.import_module(name)',
        'import runpy\nrunpy.run_path("Case" + "Study/hidden.py")',
        'from importlib.machinery import SourceFileLoader as Loader\n'
        'Loader("hidden", "data/" + "fetcher/core.py").load_module()',
        'import sys\nroot = "C:/repo"\nsys.path.insert(0, root + "/SpatialGranularity")',
        '__import__("data.fetcher.generator")',
    ],
)
def test_malicious_legacy_loads_are_rejected(source: str):
    assert _legacy_loads(source, "malicious.py")


@pytest.mark.parametrize(
    "source",
    [
        'import importlib\nimportlib.import_module("sglib.generator.stage")',
        'import runpy\nrunpy.run_path("casestudy/2_Generator/run_all.py")',
        'headers = {"User-Agent": "SpatialGranularity-research/2.0"}',
        "from .fetchers import osm_fetcher",
    ],
)
def test_active_names_are_not_mistaken_for_legacy(source: str):
    assert not _legacy_loads(source, "benign.py")


@pytest.mark.parametrize("suffix", [".pickle", ".gpkg", ".xlsx", ".geojson", ".svg", ".tif", ".onnx"])
def test_result_suffixes_are_classified(suffix: str):
    assert suffix in RESULT_LIKE_SUFFIXES
