# -*- coding: utf-8 -*-
"""
020 - Backfill German fields in the three-case comparison table

The three_case_comparison.csv produced by Australia/013_statistical_evaluation.py
has the German GNN arm and the static additive arm left as placeholders. This
script backfills those fields from the de_full_matrix.csv produced by 019,
writing the result to results/three_case_final.csv without modifying 013's
output or the original AU artifact.

Backfill rules (all values are sourced from the 019 output, not hand-entered):
- The 7 GNN arms (GNN / GNNpost* / GNNadd*): de_rmse_boerde = the rmse row
  value from 019 (mean across 3 seeds of the baseline config); de_status =
  'ok (019 rerun)'.
- The 6 static additive arms (UniAdd* / GPMadd*): also sourced from 019;
  UniAdd* is flagged as a structural degeneration (identical to Uni, which
  019 verifies exactly).
- All other rows (static multiplicative arms already filled in) are left
  unchanged.

Usage: `python 020_backfill_three_case.py` (cwd = StudyCase/Germany).
"""

from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
AU_CSV = (PROJECT_ROOT / 'StudyCase' / 'Australia' / 'data' / 'processed'
          / 'evaluation' / 'three_case_comparison.csv')
DE_MATRIX_CSV = SCRIPT_DIR / 'results' / 'de_full_matrix.csv'
OUT_CSV = SCRIPT_DIR / 'results' / 'three_case_final.csv'

GNN_ARMS = ['GNN', 'GNNpostN', 'GNNpostP', 'GNNpostNP',
            'GNNaddN', 'GNNaddP', 'GNNaddNP']
STATIC_ADD_ARMS = ['UniAddN', 'UniAddP', 'UniAddNP',
                   'GPMaddN', 'GPMaddP', 'GPMaddNP']
DEGENERATE_ARMS = {'UniAddN', 'UniAddP', 'UniAddNP'}


def main():
    three = pd.read_csv(AU_CSV, encoding='utf-8-sig')
    de = pd.read_csv(DE_MATRIX_CSV)
    de_rmse = de[de['metric'] == 'rmse'].set_index('arm')

    n_filled = 0
    for arm in GNN_ARMS + STATIC_ADD_ARMS:
        assert arm in de_rmse.index, f'Arm {arm} missing from the 019 matrix'
        mask = three['arm'] == arm
        assert mask.sum() == 1, f'Row count for arm {arm} in the three-case table != 1'

        value = float(de_rmse.loc[arm, 'value'])
        three.loc[mask, 'de_rmse_boerde'] = value

        if arm in GNN_ARMS:
            three.loc[mask, 'de_method_name'] = \
                'de_full_matrix (019, baseline×3seed mean)'
            three.loc[mask, 'de_status'] = 'ok (019 rerun)'
            note = ('DE multiplicative NP = single-renorm combined factor is identical to '
                    '005 sequential stacking (verified against the 019 anchor value)'
                    if arm == 'GNNpostNP' else '')
        elif arm in DEGENERATE_ARMS:
            three.loc[mask, 'de_method_name'] = 'de_full_matrix (019)'
            three.loc[mask, 'de_status'] = 'ok (019, structural degeneration)'
            note = 'DE Uniform base is uniform per Gemeinde -> additive degenerates exactly to Uni (asserted by 019)'
        else:
            three.loc[mask, 'de_method_name'] = 'de_full_matrix (019)'
            three.loc[mask, 'de_status'] = 'ok (019 rerun)'
            note = ''

        if note:
            old = three.loc[mask, 'note'].fillna('').iloc[0]
            three.loc[mask, 'note'] = (old + '；' + note) if old else note
        n_filled += 1

    remaining = three['de_status'].astype(str).str.contains('待补').sum()
    assert remaining == 0, f'{remaining} rows still have unfilled German fields'

    three.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    print(f'Backfilled {n_filled} German field rows -> {OUT_CSV}')
    print(three[['arm', 'uk_rmse_mean_16regions', 'de_rmse_boerde',
                 'au_rmse_mean_12regions', 'de_status']].to_string(index=False))


if __name__ == '__main__':
    main()
