"""Receipt-source integrity during repository relocation."""
import pytest

pytestmark = pytest.mark.gate


def test_civd_audit_sources_resolve_after_repository_move_from_another_directory(monkeypatch, tmp_path):
    import hashlib
    from pathlib import Path
    import shutil
    from sglib.report import civd_audit as audit

    original = tmp_path / "original" / "spatialgranularity"
    corrected = original / "results/_releases/fixture/4_Analysis/1_UK/C3/region_metrics.csv"
    historical = original / "results/4_Analysis/1_UK/C3/region_metrics.csv"
    for file, text in ((corrected, "region,value\nr0,1\n"), (historical, "region,value\nr0,2\n")):
        file.parent.mkdir(parents=True)
        file.write_text(text, encoding="utf-8")
    digest = {file: hashlib.sha256(file.read_bytes()).hexdigest() for file in (corrected, historical)}
    provenance = {
        # New records hold repository-relative paths; old records hold absolute paths of the original checkout.
        "corrected_sources": {"uk": {"path": corrected.relative_to(original).as_posix(), "sha256": digest[corrected]}},
        "original_invalid_implementation_sources": {"uk": {"path": str(historical), "sha256": digest[historical]}},
    }
    comparison = {"commitment": {"inputs": {
        "analysis:uk:C3:region_metrics.csv": digest[corrected],
        "historical:uk:C3:region_metrics.csv": digest[historical],
    }}}

    moved = tmp_path / "moved" / "elsewhere" / "spatialgranularity"
    moved.parent.mkdir(parents=True)
    shutil.move(str(original), str(moved))
    elsewhere = tmp_path / "unrelated_working_directory"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    audit._verify_comparison_sources(moved, provenance, comparison)
    tampered = {**provenance, "corrected_sources": {"uk": {**provenance["corrected_sources"]["uk"], "sha256": "0" * 64}}}
    with pytest.raises(ValueError, match="hash differs"):
        audit._verify_comparison_sources(moved, tampered, comparison)


def _case_entry(monkeypatch):
    import runpy
    from pathlib import Path
    from sglib.core.infra.paths import case_study_root
    repo = Path(__file__).resolve().parents[1]
    directory = case_study_root(repo) / "3_Experiment"
    monkeypatch.syspath_prepend(str(directory))
    return runpy.run_path(str(directory / "rerun_civd.py"))


def test_prepare_retains_inherited_bytes_but_omits_setup_and_closures(tmp_path, monkeypatch):
    entry = _case_entry(monkeypatch)
    source = tmp_path / "results"
    files = {"3_Experiment/1_UK/planning/r/receipt.json": "original receipt\n",
        "3_Experiment/1_UK/setup/setup.json": "historical setup\n",
        "3_Experiment/_closures/old.json": "historical closure\n",
        "3_Experiment/1_UK/observations/r/allocator/metrics.csv": "regenerate\n",
        "3_Experiment/1_UK/observations/r/reconstruction/metrics.csv": "inherited\n",
        "4_Analysis/1_UK/C3/receipt.json": "regenerate\n",
        "4_Analysis/1_UK/C1/receipt.json": "original analysis\n",
        "5_Report/setup/setup.json": "historical report setup\n"}
    for relative, content in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    target = source / "_releases/fixture"
    entry["prepare"](tmp_path, target)
    for relative in ("3_Experiment/1_UK/planning/r/receipt.json", "4_Analysis/1_UK/C1/receipt.json",
                     "3_Experiment/1_UK/observations/r/reconstruction/metrics.csv"):
        assert (target / relative).read_bytes() == (source / relative).read_bytes()
    assert not list(target.rglob("setup")) and not list(target.rglob("_closures"))
    assert not (target / "4_Analysis/1_UK/C3").exists()
    for relative, content in files.items():
        assert (source / relative).read_text(encoding="utf-8") == content


def test_seal_requires_clean_commit_and_new_formal_setup_before_closures(tmp_path, monkeypatch):
    from sglib.core.infra import setup_snapshot
    entry = _case_entry(monkeypatch)
    results = tmp_path / "results/_releases/fixture"
    results.mkdir(parents=True)
    monkeypatch.setattr(setup_snapshot, "git_state", lambda _: {"commit": "a" * 40, "dirty": True})
    with pytest.raises(ValueError, match="clean working tree"):
        entry["seal"](tmp_path, results)
    monkeypatch.setattr(setup_snapshot, "git_state", lambda _: {"commit": "a" * 40, "dirty": False})
    with pytest.raises(ValueError, match="missing or invalid setup"):
        entry["seal"](tmp_path, results)
    assert not list(results.rglob("_closures"))


@pytest.mark.parametrize("stage_name", ["2_Generator", "3_Experiment", "4_Analysis", "5_Report"])
def test_civd_producers_refuse_preexisting_closure_before_writing(tmp_path, stage_name):
    from types import SimpleNamespace
    from sglib.generator.civd_extension import extend_country
    from sglib.experiment.civd_observations import observe_country
    from sglib.analysis.civd_comparison import build_comparison
    from sglib.report.civd_audit import audit_release

    results = tmp_path / "results/_releases/fixture"
    marker = results / stage_name / "_closures/frozen.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("sealed", encoding="utf-8")
    if stage_name == "2_Generator":
        with pytest.raises(ValueError, match="sealed Generator"):
            extend_country(tmp_path, "nl", results, data=None, generator=None, regions=())
    elif stage_name == "3_Experiment":
        context = SimpleNamespace(repo=tmp_path, results=results, country="nl", read_only=False)
        with pytest.raises(ValueError, match="sealed Experiment"):
            observe_country(context, input_loader=None, correction_id="fixture")
    elif stage_name == "4_Analysis":
        with pytest.raises(ValueError, match="sealed Analysis"):
            build_comparison(tmp_path, results)
    with pytest.raises(ValueError, match="sealed results"):
        audit_release(tmp_path, results, country_inputs=None, observation_code=None,
            scientific_invariants=None, verify_links=None, correction_id="fixture")
    assert [path for path in results.rglob("*") if path.is_file()] == [marker]
