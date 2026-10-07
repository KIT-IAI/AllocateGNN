"""Registered evidence matrix; dimensions follow claims and countries."""
from .common import plt, np, pd, source, save_at, outputs


def render(tables, spec, target):
    evidence = tables['cross:synthesis']['claim_evidence_status'][spec['status_columns']]
    identifier = spec['figures'][0]
    source(target, identifier, evidence)
    claims, countries = spec['shared']['claims'], spec['shared']['countries']
    index = spec['heatmap_index']
    table = evidence.set_index(index).reindex(pd.MultiIndex.from_product([claims, countries], names=index))
    numerator = table[spec['heatmap_numerator']].to_numpy(float)
    denominator = table[spec['heatmap_denominator']].to_numpy(float)
    # Correct the inherited stale shape using the registered matrix dimensions.
    ratios = np.divide(numerator, denominator, out=np.zeros(len(table), float),
                       where=denominator > 0).reshape(len(claims), len(countries))
    fig, ax = plt.subplots(figsize=(7, 7))
    image = ax.imshow(ratios, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(range(len(countries)), [cc.upper() for cc in countries])
    ax.set_yticks(range(len(claims)), claims)
    for i in range(len(claims)):
        for j in range(len(countries)):
            row = table.iloc[i*len(countries)+j]
            reason = row.get('reason', '')
            limited = pd.notna(reason) and bool(str(reason).strip())
            label = f"{int(row[spec['heatmap_numerator']])}/{int(row[spec['heatmap_denominator']])}"
            ax.text(j, i, label + (' †' if limited else ''), ha='center', va='center', fontsize=9)
    ax.set_title(f'{identifier} · Evidence units')
    fig.colorbar(image, ax=ax, label='Valid / expected evidence units')
    fig.text(.02, .01, '† Nonempty recorded reason.', fontsize=8)
    fig.tight_layout(rect=(0, .03, 1, 1))
    save_at(target, identifier, fig)
    return outputs(target)
