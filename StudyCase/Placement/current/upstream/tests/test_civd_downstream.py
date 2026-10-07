"""CIVD downstream regeneration scope and explicit scientific registrations."""
import pandas as pd
import pytest

from sglib.analysis.config import COUNTRIES, DIRECTORIES

pytestmark = pytest.mark.gate


def test_civd_rerun_omits_all_four_country_units_and_refuses_shared_roots(tmp_path):
    from sglib.analysis import civd_downstream
    module = vars(civd_downstream)
    assert module['CHANGED_COUNTRIES'] == COUNTRIES
    excluded = set(module['regenerated_paths']())
    assert all(f'4_Analysis/{directory}/{member}' in excluded
               for directory in DIRECTORIES.values() for member in ('C3', 'support', 'audit.json', 'manifest.json'))
    for name in ('_staging', '_releases'):
        container = tmp_path / 'results' / name
        results = container / 'correction'
        results.mkdir(parents=True)
        assert module['fresh_root'](tmp_path, results) == (tmp_path.resolve(), results.resolve())
        with pytest.raises(ValueError, match='dedicated'):
            module['fresh_root'](tmp_path, container)
        (results / '4_Analysis/4_NL/C3').mkdir(parents=True)
        with pytest.raises(ValueError, match='omit regenerated'):
            module['fresh_root'](tmp_path, results)
    with pytest.raises(ValueError, match='dedicated'):
        module['fresh_root'](tmp_path, tmp_path / 'results')


def test_civd_support_effective_scope_is_explicit_and_default_remains_historical():
    from sglib.analysis.support import c3_support
    gates = pd.DataFrame([{'country':'nl','candidate':'GNN','allocator':'CIVD',
        'assignment_sha256':'a'*64,'region':'r','changed_grid_count':1,'fallback':False,
        'g0_pass':None,'g1_pass':None,'candidate_tv_mass':.1,'max_station_mass_change':2.}])
    metrics = pd.DataFrame([{'country':'nl','candidate':'GNN','allocator':'CIVD','region':'r',
        'metric':'rmse','status':'VALID','value':1.,'unit':'MW'}])
    empty = pd.DataFrame()
    historical = c3_support(gates, empty, empty, metrics)['C3_CIVD_summary']
    corrected = c3_support(gates, empty, empty, metrics,
        correction_scope='four_country_posthoc_bugfix_comparison')['C3_CIVD_summary']
    assert historical.evidence.tolist() == ['historical_defense_not_four_country_core_axis']
    assert corrected.evidence.tolist() == ['four_country_posthoc_bugfix_comparison']
    pd.testing.assert_frame_equal(historical.drop(columns='evidence'), corrected.drop(columns='evidence'))


def _target_schema_fixture():
    context = pd.DataFrame({"country": ["nl", "nl"], "region": ["r0", "r1"],
        "n_source": [1, 2], "n_target": [2, 3], "granularity_ratio": [2., 1.5],
        "source_total": [10., 20.], "target_total": [10., 20.]})
    current = context.copy()
    current["region_nonempty"] = True
    current["granularity_ratio_consistent"] = True
    current["mass_reconciled"] = True
    current["country_target_count_from_regions"] = 5
    prior = current[["country", "region", "n_source", "n_target", "country_target_count_from_regions"]].copy()
    prior["registered_target_count"] = 5
    prior["target_count_matches_current_frozen_specification"] = True
    return prior, current, context


def test_target_schema_transition_validates_all_new_and_retired_columns():
    from sglib.analysis.civd_downstream import _target_schema_transition
    prior, current, context = _target_schema_fixture()
    result = _target_schema_transition(prior, current, context)
    assert result["comparison"] == "exact_inherited_counts_and_receipted_context_reconciliation"
    assert set(result["independently_validated_columns"]) == {"granularity_ratio", "source_total", "target_total",
        "region_nonempty", "granularity_ratio_consistent", "mass_reconciled"}
    assert set(result["retired_columns"]) == {"registered_target_count", "target_count_matches_current_frozen_specification"}


@pytest.mark.parametrize("side,column,value", [
    ("prior", "registered_target_count", 6),
    ("prior", "target_count_matches_current_frozen_specification", False),
    ("current", "n_target", 4),
    ("current", "source_total", 11.),
    ("current", "target_total", 11.),
    ("current", "granularity_ratio", 2.1),
    ("current", "region_nonempty", False),
    ("current", "granularity_ratio_consistent", False),
    ("current", "mass_reconciled", False),
    ("current", "country_target_count_from_regions", 6),
    ("context", "region", "unregistered"),
])
def test_target_schema_transition_rejects_every_scientific_or_audit_change(side, column, value):
    from sglib.analysis.civd_downstream import _target_schema_transition
    prior, current, context = _target_schema_fixture()
    {"prior": prior, "current": current, "context": context}[side].loc[0, column] = value
    with pytest.raises((AssertionError, ValueError)):
        _target_schema_transition(prior, current, context)


def test_target_schema_transition_does_not_allow_arbitrary_schema_drift():
    from sglib.analysis.civd_downstream import _target_schema_transition
    prior, current, context = _target_schema_fixture()
    current["unexpected"] = True
    with pytest.raises(ValueError, match="unrecognized"):
        _target_schema_transition(prior, current, context)


def test_target_schema_reconciliation_independently_rechecks_mass_and_ratio():
    from sglib.analysis.civd_downstream import _target_schema_transition
    prior, current, context = _target_schema_fixture()
    for table in (current, context):
        table.loc[0, "target_total"] += 1.
    with pytest.raises(ValueError, match="mass_reconciled"):
        _target_schema_transition(prior, current, context)
    prior, current, context = _target_schema_fixture()
    for table in (current, context):
        table.loc[0, "granularity_ratio"] += 1.
    with pytest.raises(ValueError, match="granularity_ratio_consistent"):
        _target_schema_transition(prior, current, context)
