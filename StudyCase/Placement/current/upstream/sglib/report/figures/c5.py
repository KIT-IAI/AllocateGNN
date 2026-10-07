"""Mechanical registered figure rendering."""
from .common import plt, np, source as _source, save_at as _save, forest as _forest, line_facets as _line_facets, outputs
from ..tables import combine as _combine, unpack

def render(tables, spec, target):
    root = target
    country_data, synthesis = unpack(tables, spec['shared']['countries'])
    c5 = _combine({c: {'core': country_data[c]['core']['C5']} for c in spec['shared']['countries']}, 'core', 'contrasts')
    _forest(root, spec['figures'][0], c5, 'C5 · Aggregate-minus-cell association with selection regret', shared=spec['shared'])
    scales = _combine(country_data, 'support', 'C5_scale_curves')
    _line_facets(root, spec['figures'][1], scales[scales.metric.eq(spec['scale_metric']) & scales.candidate.isin(spec['scale_candidates'])], 'C5 · Complete task-scale curves', spec['scale_axes']['x'], spec['scale_axes']['y'], spec['scale_axes']['series'], shared=spec['shared'])
    support5 = _combine(country_data, 'support', 'C5_support_summary')
    _line_facets(root, spec['figures'][2], support5, 'C5 · Support and saturation diagnostics', spec['support_axes']['x'], spec['support_axes']['y'], spec['support_axes']['series'], shared=spec['shared'])
    return outputs(target)
