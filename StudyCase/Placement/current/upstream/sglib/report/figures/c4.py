"""Mechanical registered figure rendering."""
from .common import plt, np, source as _source, save_at as _save, forest as _forest, line_facets as _line_facets, outputs
from ..tables import combine as _combine, unpack

def render(tables, spec, target):
    root = target
    country_data, synthesis = unpack(tables, spec['shared']['countries'])
    c4 = _combine({c: {'core': country_data[c]['core']['C4']} for c in spec['shared']['countries']}, 'core', 'contrasts')
    _forest(root, spec['figures'][0], c4, 'C4 · One upstream change across four tasks', shared=spec['shared'])
    maps = synthesis['connection_map']
    _source(root, spec['figures'][1], maps)
    fig, axes = plt.subplots(len(spec['map_values']), len(spec['shared']['countries']), figsize=(20, 9), squeeze=False)
    for col, country in enumerate(spec['shared']['countries']):
        t = maps[maps.country.eq(country) & maps.candidate.eq(spec['map_candidate'])]
        for row, (value, label) in enumerate(zip(spec['map_values'], spec['map_labels'], strict=True)):
            ax = axes[row, col]
            s = ax.scatter(t.x / spec['map_coordinate_scale'], t.y / spec['map_coordinate_scale'], c=t[value], s=8, cmap='viridis')
            sel = t[spec['map_selectors'][row]]
            ax.scatter(t.loc[sel, 'x'] / spec['map_coordinate_scale'], t.loc[sel, 'y'] / spec['map_coordinate_scale'], s=16, facecolors='none', edgecolors='#d62728')
            ax.set_aspect('equal')
            ax.set_title(f'{country.upper()} · {label}')
            fig.colorbar(s, ax=ax, shrink=0.7)
    fig.suptitle('C4 · Fixed candidate pool, neighbourhood requirement and shortlist')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(root, spec['figures'][1], fig)
    secondary = _combine(country_data, 'support', 'C4_secondary_effects')
    _forest(root, spec['figures'][2], secondary[secondary.status.eq(spec['secondary_status'])], 'C4 · Registered secondary effects', label=spec['secondary_label'], shared=spec['shared'])
    return outputs(target)
