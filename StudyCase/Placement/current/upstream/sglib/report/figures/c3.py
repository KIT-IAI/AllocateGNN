"""Mechanical registered figure rendering."""
from .common import plt, np, source as _source, save_at as _save, forest as _forest, line_facets as _line_facets, outputs
from ..tables import combine as _combine, unpack

def render(tables, spec, target):
    root = target
    country_data, synthesis = unpack(tables, spec['shared']['countries'])
    c3 = _combine({c: {'core': country_data[c]['core']['C3']} for c in spec['shared']['countries']}, 'core', 'contrasts')
    _forest(root, spec['figures'][0], c3[c3.order.le(spec['fixed_order_max'])], 'C3 · Fixed-boundary effects and interaction', shared=spec['shared'])
    _forest(root, spec['figures'][1], c3[c3.order.ge(spec['program_order_min'])], 'C3 · Matched-program effects including fallback', shared=spec['shared'])
    gates = _combine(country_data, 'support', 'C3_gate_summary')
    _source(root, spec['figures'][2], gates)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()
    for ax, country in zip(axes, spec['shared']['countries'], strict=True):
        t = gates[gates.country.eq(country)].groupby(spec['gate_group'], sort=False).agg(produced=('produced', 'sum'), fallback=('fallback', 'sum')).reset_index()
        ax.bar(t.allocator, t.fallback / np.maximum(t.produced, 1))
        ax.set(title=country.upper(), ylabel='Fallback share')
        ax.tick_params(axis='x', rotation=30)
    fig.suptitle('C3 · Gate fallback share')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(root, spec['figures'][2], fig)
    return outputs(target)
