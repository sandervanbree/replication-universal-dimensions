#!/usr/bin/env python3
"""Audit completed tables and write a compact, report-ready results summary."""
from pathlib import Path
import hashlib
import json
import os
import sys

for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '2'

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.replication.config import load_config
CONFIG = Path(os.environ.get('REPLICATION_CONFIG', REPO / 'config/analysis.json')).resolve()
ROOT = load_config(CONFIG).output_root / 'generalization'


def p_description(values):
    values = np.asarray(values)
    if np.all(values == values[0]):
        return f'all p = {values[0]:.4f}'
    return f'p = {values.min():.4f} to {values.max():.4f}'


def main():
    completed = json.loads((ROOT/'diagnostics/completion.json').read_text())
    assert completed['state'] == 'complete'
    config = json.loads((ROOT/'configuration.json').read_text())
    b = config['n_permutations']
    environment = json.loads((ROOT/'diagnostics/environment.json').read_text())
    assert hashlib.sha256(Path(__file__).with_name('run.py').read_bytes()).hexdigest() == environment['runner_sha256']
    for split in json.loads((ROOT/'split_manifest.json').read_text()):
        path = ROOT/'permutations'/f"{split['mode']}.npy"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == split['permutation_sha256']
        permutations = np.load(path)
        inverse = np.load(ROOT/'permutations'/f"{split['mode']}.inverse.npy")
        assert permutations.shape == inverse.shape == (b, split['n_test'])
        assert np.all(np.take_along_axis(permutations, inverse, axis=1) == np.arange(split['n_test']))
    subjects = ['sub-01', 'sub-03', 'sub-05', 'sub-06', 'sub-07']
    names = {
        'resnet18_classification_imagenet1k_varied_seeds': 'Different initializations',
        'classification_imagenet1k_varied_architectures': 'Different architectures',
        'resnet50_imagenet1k_varied_tasks': 'Different tasks',
        'untrained_resnet18_varied_seeds': 'Untrained',
    }
    modes = {
        'generalization_tau': 'Tau',
        'generalization_clip_cluster_average': 'CLIP clusters (five-fold average)',
        'generalization_ood': 'OOD',
    }
    filenames = {
        'participant_pvalues_all_7_splits.csv': 140,
        'participant_pvalues_report_3_tests.csv': 60,
        'participant_pvalues_clip_cluster_average.csv': 20,
        'mean_participant_effect_pvalues.csv': 32,
    }
    frames = {}
    checked = 0
    for filename, n_rows in filenames.items():
        frame = pd.read_csv(ROOT/'tables'/filename)
        assert len(frame) == n_rows
        assert not frame.duplicated(['mode', 'model_set', 'subject']).any()
        assert np.isfinite(frame.select_dtypes(include='number').to_numpy()).all()
        assert (frame.n_permutations == b).all()
        assert (frame.alternative == 'greater').all()
        for prefix, null_key, obs_key in [
            ('spearman', 'spearman', 'observed'),
            ('partial_spearman', 'partial_spearman', 'observed_partial'),
        ]:
            p = frame[f'{prefix}_p_permutation'].to_numpy()
            np.testing.assert_allclose(frame[f'{prefix}_q_bh'],
                                       false_discovery_control(p, method='bh'), atol=1e-14)
            assert ((p >= 1/(b+1)-1e-15) & (p <= 1)).all()
            for row in frame.to_dict('records'):
                with np.load(ROOT/'null_statistics'/row['mode']/f"{row['model_set']}.npz") as f:
                    null = f[null_key]
                    obs = f[obs_key]
                assert null.shape == (b, 5)
                assert np.isfinite(null).all() and np.isfinite(obs).all()
                if row['subject'] == 'mean_across_participants':
                    null = null.mean(axis=1)
                    obs = obs.mean()
                else:
                    index = subjects.index(row['subject'])
                    null = null[:, index]
                    obs = obs[index]
                count = int(np.count_nonzero(null >= obs))
                assert count == row[f'{prefix}_exceedances']
                np.testing.assert_allclose(row[f'{prefix}_p_permutation'], (count+1)/(b+1), atol=1e-15)
                checked += 1
        frames[filename] = frame

    main_table = frames['participant_pvalues_report_3_tests.csv']
    all_splits = frames['participant_pvalues_all_7_splits.csv']
    assert (all_splits.spearman_rho > 0).all()
    assert (all_splits.partial_spearman_rho_control_pc > 0).all()
    compact = []
    lines = []
    for model_set, name in names.items():
        for mode, label in modes.items():
            group = main_table[(main_table.model_set == model_set) & (main_table['mode'] == mode)]
            assert len(group) == 5
            compact.append({
                'model_set': name, 'test': label, 'n_participants': 5,
                'mean_spearman_rho': group.spearman_rho.mean(),
                'mean_partial_spearman_rho': group.partial_spearman_rho_control_pc.mean(),
                'largest_participant_spearman_p': group.spearman_p_permutation.max(),
                'largest_participant_partial_p': group.partial_spearman_p_permutation.max(),
                'largest_participant_spearman_q_bh': group.spearman_q_bh.max(),
                'largest_participant_partial_q_bh': group.partial_spearman_q_bh.max(),
            })
            lines.append(f"{name}; {label}: mean rho = {group.spearman_rho.mean():.3f}, "
                         f"mean partial rho = {group.partial_spearman_rho_control_pc.mean():.3f}; "
                         f"largest participant p = {group.spearman_p_permutation.max():.8f} "
                         f"(partial: {group.partial_spearman_p_permutation.max():.8f}).")
    pd.DataFrame(compact).to_csv(ROOT/'tables/report_summary.csv', index=False)

    raw = all_splits.spearman_p_permutation.to_numpy()
    partial = all_splits.partial_spearman_p_permutation.to_numpy()
    all_significant = (raw < .05).all() and (partial < .05).all()
    claim = 'positive and significant' if all_significant else 'positive'
    sentence = (
        f"All 140 participant-level correlations across the seven generalization splits "
        f"were {claim} (Spearman rho = {all_splits.spearman_rho.min():.2f} to "
        f"{all_splits.spearman_rho.max():.2f}; {p_description(raw)}). "
        f"The partial correlations controlling for PC rank were also {claim} "
        f"(minimum partial rho = {all_splits.partial_spearman_rho_control_pc.min():.2f}; "
        f"{p_description(partial)}). P-values were obtained from {b:,} one-sided "
        "permutations of held-out DNN-to-DNN predictions, following the original "
        "study's image-shuffling procedure."
    )
    clusters = frames['participant_pvalues_clip_cluster_average.csv']
    report_insert = (
        f"Significance was assessed with {b:,} one-sided permutations of held-out "
        "DNN-to-DNN predictions, following the original study's image-shuffling "
        f"procedure. Across all 140 participant-level tests, {p_description(raw)} "
        f"for Spearman correlation and {p_description(partial)} after controlling "
        "for PC rank. For correlations averaged across the five CLIP-cluster "
        f"folds, {p_description(clusters.spearman_p_permutation)} for Spearman "
        f"correlation and {p_description(clusters.partial_spearman_p_permutation)} "
        "for partial correlation."
    )
    report_p_values = np.concatenate([
        raw, partial, clusters.spearman_p_permutation, clusters.partial_spearman_p_permutation,
    ])
    if all_significant and np.all(report_p_values == report_p_values[0]):
        report_p = f'{report_p_values[0]:.4f}'.lstrip('0')
        report_insert = (
            "All 140 participant-level correlations across the seven generalization "
            "splits were significant, including after controlling for PC rank "
            f"(all p = {report_p}; {b:,} one-sided permutations of held-out "
            "DNN-to-DNN predictions). The correlations averaged across the five "
            f"CLIP-cluster folds were also significant (all p = {report_p} for both metrics)."
        )
    (ROOT/'report_insert.txt').write_text(report_insert+'\n')
    q_max = max(all_splits.spearman_q_bh.max(), all_splits.partial_spearman_q_bh.max())
    correction = (f"Largest BH-adjusted q across the 140 tests per correlation metric: {q_max:.8f}. "
                  "The report's 60 participant effects are also supplied with their own BH correction.")
    text = (
        "GENERALIZATION P-VALUES — COMPLETED ANALYSIS\n\n"
        f"{b:,} permutations per split; seed {config['seed']}; alternative: greater.\n"
        f"Spearman: {p_description(raw)} across all 140 individual split tests.\n"
        f"Partial Spearman: {p_description(partial)} across all 140 individual split tests.\n"
        f"{correction}\n\n"
        "SUGGESTED TEXT FOR THE MARKED REPORT SECTION\n\n"
        f"{sentence}\n\n"
        "METHOD DETAIL\n\n"
        "Within each split, the same shuffled image order is used for every predictor "
        "network and target PC. Brain-similarity scores stay fixed. For the five-fold "
        "CLIP aggregate, the mean observed correlation is compared with the corresponding "
        "mean null correlation; p-values themselves are not averaged. P-values use "
        f"(exceedances + 1)/({b} + 1), so the minimum is {1/(b+1):.10f}, never zero. "
        "These tests do not test differences between generalization and the primary replication.\n\n"
        "REPORT TABLE SUMMARY\n\n" + '\n'.join(lines) + '\n\n'
        "FILES\n\n"
        "tables/participant_pvalues_report_3_tests.csv: 60 participant effects for tau, "
        "CLIP average, and OOD, with both correlation metrics and p-values.\n"
        "tables/participant_pvalues_all_7_splits.csv: all 140 individual split effects.\n"
        "tables/report_summary.csv: 12 compact model-set/test summaries.\n"
        "report_insert.txt: concise replacement for the marked p-value placeholder.\n"
        "The original report and September outputs have not been edited.\n"
    )
    (ROOT/'RESULTS.txt').write_text(text)
    audit = {'state': 'passed', 'p_values_recomputed_from_saved_nulls': checked,
             'all_null_statistics_finite': True, 'bh_matches_scipy': True,
             'runner_hash_unchanged': True, 'permutation_hashes_and_inverses_valid': True,
             'result_table_sha256': {name: hashlib.sha256((ROOT/'tables'/name).read_bytes()).hexdigest()
                                     for name in [*filenames, 'report_summary.csv']}}
    (ROOT/'diagnostics/final_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(text)


if __name__ == '__main__':
    main()
