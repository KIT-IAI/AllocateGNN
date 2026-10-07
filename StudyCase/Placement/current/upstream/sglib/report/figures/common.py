"""Shared drawing primitives; all data selections are supplied by registrations."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..tables import write_csv


def source(root, identifier, frame):
    write_csv(frame, root / 'sources' / f'{identifier}.csv')


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix('.png'), dpi=180, bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


def save_at(root, identifier, fig):
    save(fig, root / 'figures' / identifier)


def empty(root, identifier, title, reason='No assessable observations'):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')
    ax.text(.5, .62, title, ha='center', va='center', fontsize=14)
    ax.text(.5, .35, reason, ha='center', va='center', fontsize=11)
    save_at(root, identifier, fig)


def facets(shared, size):
    columns = 2
    rows = (len(shared['countries']) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=size, squeeze=False)
    for ax in axes.ravel()[len(shared['countries']):]:
        ax.set_axis_off()
    return fig, axes.ravel()[:len(shared['countries'])]


def forest(root, identifier, frame, title, *, shared, label='contrast_id', effect='effect', low='ci_low', high='ci_high'):
    source(root, identifier, frame)
    if frame.empty or effect not in frame:
        return empty(root, identifier, title)
    fig, axes = facets(shared, (15, 11))
    for ax, country in zip(axes, shared['countries'], strict=True):
        table = frame[frame.country.eq(country)].copy()
        if 'order' in table:
            table = table.sort_values('order', kind='stable')
        table = table[table[effect].notna()]
        if table.empty:
            ax.text(.5, .5, 'Not assessable', ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            continue
        y = np.arange(len(table))
        values = table[effect].to_numpy(float)
        lo = table[low].to_numpy(float) if low in table else values
        hi = table[high].to_numpy(float) if high in table else values
        filled = table.get('claim_supported', pd.Series(False, index=table.index)).fillna(False).to_numpy(bool)
        ax.errorbar(values, y, xerr=np.vstack([values-lo, hi-values]), fmt='none', color='#666666', lw=1)
        ax.scatter(values, y, s=35, facecolors=np.where(filled, shared['colors'][country], 'white'),
                   edgecolors=shared['colors'][country], zorder=3)
        ax.axvline(0, color='#333333', ls='--', lw=.8)
        labels = table[label].astype(str).tolist() if label in table else [str(i+1) for i in y]
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_title(country.upper())
        ax.grid(axis='x', alpha=.15)
        ax.set_xlabel('Registered effect (left − right)')
    fig.suptitle(title)
    fig.text(.02, .01, 'Intervals: ci_low, ci_high. Filled markers: claim_supported.', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .96))
    save_at(root, identifier, fig)


def select(frame, selections):
    mask = pd.Series(True, index=frame.index)
    for column, value in selections.items():
        mask &= frame[column].eq(value)
    return frame[mask]


def line_facets(root, identifier, frame, title, x, y, series, *, shared, selections=None):
    # C6 sources retain the entire table; filtering only affects drawing.
    source(root, identifier, frame)
    plot = select(frame, selections) if selections is not None else frame.copy()
    if plot.empty or x not in plot or y not in plot:
        return empty(root, identifier, title)
    fig, axes = facets(shared, (14, 9))
    for ax, country in zip(axes, shared['countries'], strict=True):
        table = plot[plot.country.eq(country)]
        for name, group in table.groupby(series, sort=False, dropna=False):
            group = group.sort_values(x, kind='stable')
            ax.plot(group[x], group[y], marker='o', ms=3, lw=1, label=str(name))
        ax.set_title(country.upper())
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(alpha=.15)
        if table[series].nunique(dropna=False) <= 8:
            ax.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, .96))
    save_at(root, identifier, fig)


def outputs(target):
    return tuple(sorted(path.relative_to(target).as_posix() for path in target.rglob('*') if path.is_file()))
