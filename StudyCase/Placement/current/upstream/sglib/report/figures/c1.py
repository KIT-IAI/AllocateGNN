"""Mechanical registered figure rendering."""
from .common import plt, np, source as _source, save_at as _save, forest as _forest, line_facets as _line_facets, outputs
from ..tables import combine as _combine, unpack

def render(tables, spec, target):
    root = target
    country_data, synthesis = unpack(tables, spec['shared']['countries'])
    c1 = _combine({c: {'core': country_data[c]['core']['C1']} for c in spec['shared']['countries']}, 'core', 'contrasts')
    _forest(root, spec['figures'][0], c1, 'C1 · Ten preregistered reconstruction comparisons', shared=spec['shared'])
    c1_abs = _combine({c: {'core': country_data[c]['core']['C1']} for c in spec['shared']['countries']}, 'core', 'region_metrics')
    c1_abs = c1_abs[c1_abs.metric.isin(spec['metrics'])]
    _source(root, spec['figures'][1], c1_abs)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.ravel()
    for ax, country in zip(axes, spec['shared']['countries'], strict=True):
        table = c1_abs[c1_abs.country.eq(country)]
        groups = dict(tuple(table[table.metric.eq(spec['scatter_metric'])].groupby('candidate', sort=False)))
        for candidate in spec['candidate_order']:
            if candidate not in groups:
                continue
            group = groups[candidate]
            ax.scatter([candidate] * len(group), group.value * spec['percent_scale'], s=12, alpha=0.45, label=candidate)
        ax.set_title(country.upper())
        ax.set_ylabel('WAPE (%)')
        ax.tick_params(axis='x', rotation=60)
        ax.grid(axis='y', alpha=0.15)
    fig.suptitle('C1 · Absolute regional prediction error')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(root, spec['figures'][1], fig)
    routes = _combine({c: {'core': country_data[c]['core']['C1']} for c in spec['shared']['countries']}, 'core', 'equal_route_regions')
    _source(root, spec['figures'][2], routes)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    for ax, country in zip(axes, spec['shared']['countries'], strict=True):
        table = routes[routes.country.eq(country)].reset_index(drop=True)
        y = np.arange(len(table))
        ax.plot(table[spec['route_columns'][0]], y, 'o', label=spec['route_labels'][0])
        ax.plot(table[spec['route_columns'][1]], y, 'o', label=spec['route_labels'][1])
        for i, row in table.iterrows():
            ax.plot([row[spec['route_columns'][0]], row[spec['route_columns'][1]]], [i, i], color='#aaaaaa', lw=0.7)
        ax.set_yticks(y, table.region)
        ax.invert_yaxis()
        ax.set_title(country.upper())
        ax.legend(fontsize=8)
    fig.suptitle('C1 · Direct versus grid Equal routing')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(root, spec['figures'][2], fig)
    return outputs(target)
