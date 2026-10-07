"""Mechanical registered figure rendering."""
from .common import plt, np, source as _source, save_at as _save, forest as _forest, line_facets as _line_facets, outputs
from ..tables import combine as _combine, unpack

def render(tables, spec, target):
    root = target
    country_data, synthesis = unpack(tables, spec['shared']['countries'])
    curves = _combine(country_data, 'support', 'C6_tolerance_curves')
    _line_facets(root, spec['figures'][0], curves, 'C6 · Conditional sufficiency curves', spec['tolerance_axes']['x'], spec['tolerance_axes']['y'], spec['tolerance_axes']['series'], selections=spec['tolerance_curve'], shared=spec['shared'])
    audit6 = _combine(country_data, 'support', 'C6_bounds_audit')
    _line_facets(root, spec['figures'][1], audit6, 'C6 · Budget realization across eta', spec['budget_axes']['x'], spec['budget_axes']['y'], spec['budget_axes']['series'], selections=spec['budget_realization'], shared=spec['shared'])
    sat = _combine(country_data, 'support', 'C6_saturation')
    _line_facets(root, spec['figures'][2], sat, 'C6 · Saturation and zero-regret diagnostics', spec['saturation_axes']['x'], spec['saturation_axes']['y'], spec['saturation_axes']['series'], selections=spec['saturation'], shared=spec['shared'])
    return outputs(target)
