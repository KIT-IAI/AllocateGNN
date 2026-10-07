"""C1–C5 effect/interval previews only; no statistical calculation and no C6 forest."""
import numpy as np
from ..stage import figures_root
from .tables import contrasts


def forests(ctx):
    import matplotlib.pyplot as plt

    table = contrasts(ctx)
    paths = []
    for claim, rows in table.groupby('claim_id', sort=False):
        groups = list(rows.groupby('unit', sort=False, dropna=False))
        heights = [max(2., len(group) * .38 + .9) for _, group in groups]
        fig, axes = plt.subplots(len(groups), 1, figsize=(11, sum(heights)), squeeze=False,
                                 gridspec_kw={'height_ratios': heights}, layout='constrained')
        for ax, (unit, group) in zip(axes[:, 0], groups, strict=True):
            y = np.arange(len(group))
            valid = np.isfinite(group[['effect', 'ci_low', 'ci_high']].to_numpy(float)).all(axis=1)
            finite = group.loc[valid]
            ax.hlines(y[valid], finite.ci_low, finite.ci_high, color='#64748b', linewidth=2)
            ax.scatter(finite.effect, y[valid], s=32, color='#12618a', zorder=3)
            labels = []
            for row in group.itertuples():
                if isinstance(getattr(row, 'left', None), str) and isinstance(getattr(row, 'right', None), str):
                    description = f'{row.left} − {row.right}'
                else:
                    description = str(getattr(row, 'expression', ''))
                if claim == 'C4':
                    description = f'{row.task_coordinate} {row.metric}: {description}'
                if len(description) > 63:
                    description = description[:60] + '…'
                labels.append(f'{row.contrast_id}  {description}')
            ax.set_yticks(y, labels, fontsize=9)
            ax.invert_yaxis()
            ax.axvline(0, color='#94a3b8', linewidth=1, linestyle='--')
            ax.grid(axis='x', color='#e2e8f0', linewidth=.7)
            ax.set_axisbelow(True)
            ax.set_xlabel(f'Registered effect and interval ({unit})')
            for spine in ('top', 'right', 'left'):
                ax.spines[spine].set_visible(False)
            if (~valid).any():
                ax.text(.99, .01, 'Rows without finite intervals are listed without marks',
                        transform=ax.transAxes, ha='right', fontsize=8)
        fig.suptitle(f'{ctx.country.upper()} · {claim} · Analysis preview', fontsize=14)
        path = figures_root(ctx) / f'{claim}_forest_preview.png'
        fig.savefig(path, dpi=150, facecolor='white')
        plt.close(fig)
        paths.append(path)
    return paths
