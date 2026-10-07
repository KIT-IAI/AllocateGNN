"""Mechanical registered figure rendering."""
from .common import plt, np, source as _source, save_at as _save, forest as _forest, line_facets as _line_facets, outputs
from ..tables import combine as _combine, unpack

def render(tables, spec, target):
    root = target
    country_data, synthesis = unpack(tables, spec['shared']['countries'])
    c2 = _combine({c: {'core': country_data[c]['core']['C2']} for c in spec['shared']['countries']}, 'core', 'contrasts')
    _forest(root, spec['figures'][0], c2, 'C2 · Configuration interactions and registered secondary levels', shared=spec['shared'])
    mechanism = _combine(country_data, 'support', 'C2_mechanism_summary')
    _source(root, spec['figures'][1], mechanism)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()
    for ax, country in zip(axes, spec['shared']['countries'], strict=True):
        t = mechanism[mechanism.country.eq(country)]
        ax.scatter(t.majority_class_baseline, t.threshold_accuracy_log_region_mean, c=np.where(t.operator.eq(spec['mechanism_operator']), 0, 1), alpha=0.55)
        ax.plot([0, 1], [0, 1], ls='--', color='#555555', lw=0.8)
        ax.set(xlabel='Majority-class baseline', ylabel='Old-threshold accuracy', title=country.upper())
        ax.grid(alpha=0.15)
    fig.suptitle('C2 · Retrospective threshold versus majority baseline')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(root, spec['figures'][1], fig)
    sweeps = _combine(country_data, 'support', 'C2_sweep_curves')
    _line_facets(root, spec['figures'][2], sweeps[sweeps.metric.eq(spec['scan_metric'])], 'C2 · Complete fixed parameter scans', spec['scan_axes']['x'], spec['scan_axes']['y'], spec['scan_axes']['series'], shared=spec['shared'])
    return outputs(target)
