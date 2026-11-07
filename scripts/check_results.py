"""Check the small reference results without loading the original data or fitting models."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
SUBJECTS = ['sub-01', 'sub-03', 'sub-05', 'sub-06', 'sub-07']


@lru_cache(None)
def load_null(mode, model_set):
    with np.load(RESULTS / 'generalization_nulls' / mode / f'{model_set}.npz') as f:
        return {key: f[key] for key in f.files}


def main():
    ids = {name: (ROOT / 'metadata/stimulus_ids' / f'{name}.txt').read_text().splitlines()
           for name in ['unique', 'shared', 'ood']}
    assert [len(ids[k]) for k in ids] == [23560, 1121, 371]
    assert len(set(sum(ids.values(), []))) == 25052
    splits = pd.read_csv(ROOT / 'generalization/stimulus_splits.csv').fillna('')
    for mode in ['shared', 'ood']:
        assert splits.loc[splits['mode'].eq(mode), 'stimulus_id'].tolist() == ids[mode]
    shared = splits[splits['mode'].eq('shared')]
    assert (shared[[f'cluster_k5_{i}' for i in range(5)]].eq('test').sum(axis=1) == 1).all()
    for row in pd.read_csv(ROOT / 'generalization/split_counts.csv').itertuples():
        assert splits[row.split].eq('train').sum() == row.n_train
        assert splits[row.split].eq('test').sum() == row.n_test
    layers = pd.read_csv(ROOT / 'metadata/model_layers.csv')
    assert len(layers) == 1018 and layers.model.nunique() == 67

    main_stats = pd.read_csv(RESULTS / 'main/figure2_statistics.csv').set_index('model_set').sort_index()
    participant_stats = pd.read_csv(RESULTS / 'report/finding1_statistics.csv')
    average = participant_stats[participant_stats.target.eq('average')].set_index('model_set').sort_index()
    np.testing.assert_allclose(average[main_stats.columns], main_stats, atol=1e-14, rtol=0)
    assert participant_stats.target.str.startswith('sub-').sum() == 20
    assert participant_stats.target.eq('original_tolerance').sum() == 4
    rsa = pd.read_csv(RESULTS / 'main/figure5_rsa.csv')
    network = rsa.groupby(['model_set', 'model', 'condition']).rsa.mean().unstack('condition')
    for row in pd.read_csv(RESULTS / 'report/figure5_summary.csv').itertuples():
        group = network.loc[row.model_set]
        for column, condition in [('intact', 'all'), ('top10', 'within_basis_ui.n_top=10'),
                                  ('top5', 'within_basis_ui.n_top=5')]:
            np.testing.assert_allclose(group[condition].mean(), getattr(row, column), atol=1e-14)
        assert (group['within_basis_ui.n_top=10'] > group['all']).sum() == row.n_improved_top10

    checked = 0
    for name, n_rows in [('participant_pvalues_all_7_splits.csv', 140),
                         ('participant_pvalues_report_3_tests.csv', 60),
                         ('participant_pvalues_clip_cluster_average.csv', 20),
                         ('mean_participant_effect_pvalues.csv', 32)]:
        frame = pd.read_csv(RESULTS / 'generalization' / name)
        assert len(frame) == n_rows
        assert not frame.duplicated(['mode', 'model_set', 'subject']).any()
        for prefix, key, obs_key in [('spearman', 'spearman', 'observed'),
                                     ('partial_spearman', 'partial_spearman', 'observed_partial')]:
            np.testing.assert_allclose(frame[f'{prefix}_q_bh'],
                                       false_discovery_control(frame[f'{prefix}_p_permutation']), atol=1e-14)
            for row in frame.to_dict('records'):
                saved = load_null(row['mode'], row['model_set'])
                null, observed = saved[key], saved[obs_key]
                assert null.shape == (5000, 5) and np.isfinite(null).all()
                if row['subject'] == 'mean_across_participants':
                    null, observed = null.mean(axis=1), observed.mean()
                else:
                    index = SUBJECTS.index(row['subject'])
                    null, observed = null[:, index], observed[index]
                metric = 'spearman_rho' if prefix == 'spearman' else 'partial_spearman_rho_control_pc'
                np.testing.assert_allclose(observed, row[metric], atol=1e-14)
                exceed = np.count_nonzero(null >= observed)
                assert exceed == row[f'{prefix}_exceedances']
                np.testing.assert_allclose((exceed + 1) / 5001, row[f'{prefix}_p_permutation'], atol=1e-15)
                checked += 1
    print(f'Passed: image IDs and splits, model inventory, reference summaries, and {checked} saved permutation p-values.')


if __name__ == '__main__':
    main()
