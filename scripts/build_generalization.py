#!/usr/bin/env python
"""Export official shared-pool splits and empirical voxelwise reliability.

Large intermediate single-trial ROI arrays stay in --cache-root, outside the
deliverable. Dataset inputs and the existing feature/beta handoffs are read-only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.metadata
import json
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
import xarray as xr
from laion_fmri.subject import Subject
from laion_fmri.splits import get_train_test_ids
from laion_fmri._paths import r2mean_path
from laion_fmri.io import load_nifti_mask

SUBJECTS = ('sub-01', 'sub-03', 'sub-05', 'sub-06', 'sub-07')
ROI = 'laion-general'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def ids_from_handoff(root, mode):
    return (root / 'stimulus_ids' / f'{mode}.txt').read_text().splitlines()


def export_splits(args):
    shared = ids_from_handoff(args.pca_root, 'shared')
    ood = ids_from_handoff(args.pca_root, 'ood')
    assert len(shared) == len(set(shared)) == 1121
    assert len(ood) == len(set(ood)) == 371 and not set(shared) & set(ood)
    frame = pd.DataFrame({'stimulus_id': shared + ood,
                          'mode': ['shared'] * len(shared) + ['ood'] * len(ood),
                          'row_in_mode': list(range(len(shared))) + list(range(len(ood)))})
    counts = []
    for name in ['tau'] + [f'cluster_k5_{i}' for i in range(5)] + ['ood']:
        train, test = get_train_test_ids(name, pool='shared', variant_id=0)
        assert len(train) == len(set(train)) and len(test) == len(set(test))
        assert not set(train) & set(test)
        if name == 'ood':
            assert set(train) == set(shared) and set(test) == set(ood)
        else:
            assert set(train) | set(test) == set(shared)
        if name == 'tau':
            assert (len(train), len(test)) == (897, 224)
        mapping = {**dict.fromkeys(train, 'train'), **dict.fromkeys(test, 'test')}
        frame[name] = frame.stimulus_id.map(mapping).fillna('not_applicable')
        counts.append({'split': name, 'pool': 'shared', 'variant_id': 0,
                       'n_train': len(train), 'n_test': len(test)})
    cols = [f'cluster_k5_{i}' for i in range(5)]
    heldout = frame.loc[frame['mode'] == 'shared', cols].eq('test')
    assert (heldout.sum(axis=1) == 1).all()
    frame['clip_cluster'] = pd.array([int(x) for x in heldout.to_numpy().argmax(axis=1)]
                                     + [pd.NA] * len(ood), dtype='Int64')
    frame.to_csv(args.output_root / 'stimulus_splits.csv', index=False)
    pd.DataFrame(counts).to_csv(args.output_root / 'split_counts.csv', index=False)
    for subject in SUBJECTS:
        for mode, expected in [('shared', shared), ('ood', ood)]:
            with xr.open_dataarray(args.beta_root / f'mode={mode}' / f'subject={subject}.nc') as beta:
                assert beta.stimulus_id.values.astype(str).tolist() == expected
                assert beta.attrs['zscore_session'] == 1
    print('Official splits exported; IDs/order match both PCA handoff and all ten beta caches.', flush=True)


def session_job(task):
    args, subject_id, session = task
    target = args.cache_root / subject_id / f'{session}.npz'
    if target.exists():
        return subject_id, session, 'cached'
    subject = Subject(subject_id, str(args.data_root))
    roi = subject.get_roi_mask(ROI)
    brain = load_nifti_mask(r2mean_path(str(args.data_root), subject_id))
    directory = args.data_root / 'derivatives' / 'glmsingle-tedana' / subject_id / session / 'func'
    prefix = f'{subject_id}_{session}_task-images_space-T1w'
    source = directory / f'{prefix}_desc-GLMsingle_model.h5'
    trials = subject.get_trial_info(session)
    labels = trials['label'].astype(str).to_numpy()
    shared = ids_from_handoff(args.pca_root, 'shared')
    ood = ids_from_handoff(args.pca_root, 'ood')
    keep = np.isin(labels, shared + ood)
    if source.exists():
        with h5py.File(source, 'r') as handle:
            ds = handle['betasmd']
            assert ds.shape == (len(roi), 1, 1, len(trials)), (source, ds.shape)
            # Verify HDF5 voxel order against the distributed session R2 map.
            r2 = np.asarray(nib.load(directory / f'{prefix}_stat-rsquare_desc-R2_statmap.nii.gz').dataobj).ravel()[brain]
            np.testing.assert_allclose(handle['R2'][:].ravel(), r2, rtol=0, atol=0, equal_nan=True)
            # Read contiguous slabs; avoid expensive large fancy selections.
            pieces = []
            for start in range(0, len(roi), 4096):
                mask = roi[start:start + 4096]
                if mask.any():
                    pieces.append(ds[start:start + 4096, 0, 0, :][mask])
            values = np.ascontiguousarray(np.concatenate(pieces, axis=0).T, dtype=np.float32)
    else:
        # The final three sessions per subject have only the distributed NIfTI.
        source = directory / f'{prefix}_stat-effect_desc-SingletrialBetas_statmap.nii.gz'
        values = np.ascontiguousarray(subject.get_betas(session, roi=ROI, streaming=True), dtype=np.float32)
        assert values.shape == (len(trials), int(roi.sum()))
    assert np.isfinite(values).all(), (subject_id, session, 'nonfinite betas')
    # Match scripts/00_cache_laion_roi_betas.py: all session trials, ddof=0,
    # then select the desired stimulus rows. Do not z-score only the shared set.
    mu = values.mean(axis=0, keepdims=True)
    sd = values.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    values = ((values - mu) / sd)[keep]
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix('.tmp.npz')
    np.savez(temp, betas=values, stimulus_id=np.asarray(labels[keep], dtype=str),
             trial_index=np.flatnonzero(keep), source=str(source),
             session=session, subject=subject_id)
    temp.replace(target)
    return subject_id, session, f'{len(values)} trials, {values.shape[1]} voxels'


def extract(args):
    tasks = []
    for subject_id in SUBJECTS:
        with xr.open_dataarray(args.beta_root / 'mode=shared' / f'subject={subject_id}.nc') as beta:
            sessions = beta.attrs['sessions'].split(',')
        with xr.open_dataarray(args.beta_root / 'mode=ood' / f'subject={subject_id}.nc') as beta:
            assert beta.attrs['sessions'].split(',') == sessions
        tasks.extend((args, subject_id, session) for session in sessions)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(session_job, task) for task in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
            except Exception as error:
                print(f'extract FAILED: {error!r}', flush=True)
                raise
            print(f'extract {i}/{len(tasks)}: {result}', flush=True)


def voxel_pearson(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a -= a.mean(axis=0)
    b -= b.mean(axis=0)
    denominator = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
    return np.divide((a * b).sum(axis=0), denominator,
                     out=np.full(a.shape[1], np.nan), where=denominator > 0)


def subject_reliability(task):
    args, subject_id = task
    ids = ids_from_handoff(args.pca_root, 'shared') + ids_from_handoff(args.pca_root, 'ood')
    index = {label: i for i, label in enumerate(ids)}
    with xr.open_dataarray(args.beta_root / 'mode=shared' / f'subject={subject_id}.nc') as beta:
        sessions = beta.attrs['sessions'].split(',')
        n_voxels = beta.sizes['neuroid']
    labels, values, session_counts = [], [], []
    for session in sessions:
        with np.load(args.cache_root / subject_id / f'{session}.npz', allow_pickle=False) as z:
            labels.extend(z['stimulus_id'].astype(str).tolist())
            values.append(z['betas'])
            session_counts.append({'subject': subject_id, 'session': session,
                                   'n_selected_trials': len(z['betas']),
                                   'source': str(z['source'].item())})
    values = np.concatenate(values)
    indices = np.array([index[label] for label in labels])
    counts = np.bincount(indices, minlength=len(ids))
    assert counts.min() >= 2, (subject_id, counts.min())
    assert values.shape[1] == n_voxels and np.isfinite(values).all()
    sums = np.zeros((len(ids), n_voxels), dtype=np.float64)
    np.add.at(sums, indices, values)
    averaged = sums / counts[:, None]
    validation = []
    for mode, sl in [('shared', slice(0, 1121)), ('ood', slice(1121, 1492))]:
        with xr.open_dataarray(args.beta_root / f'mode={mode}' / f'subject={subject_id}.nc') as beta:
            cached = beta.values
            assert beta.stimulus_id.values.astype(str).tolist() == ids[sl]
            assert int(counts[sl].sum()) == beta.attrs['n_trials_kept']
            difference = np.abs(averaged[sl] - cached)
            np.testing.assert_allclose(averaged[sl], cached, atol=2e-5, rtol=2e-5)
            validation.append({'subject': subject_id, 'mode': mode,
                               'n_images': len(cached), 'n_voxels': n_voxels,
                               'n_trials': int(counts[sl].sum()),
                               'max_abs_difference_from_supplied_betas': float(difference.max()),
                               'mean_abs_difference_from_supplied_betas': float(difference.mean())})
    # Padded repetition axis supports the actual (occasionally incomplete)
    # trial counts. Split each image independently; no image is discarded.
    repetitions = np.zeros((len(ids), int(counts.max()), n_voxels), dtype=np.float32)
    offsets = np.zeros(len(ids), dtype=int)
    for trial, image in enumerate(indices):
        repetitions[image, offsets[image]] = values[trial]
        offsets[image] += 1
    del values
    groups = {'shared': np.arange(1121), 'ood': np.arange(1121, 1492),
              'shared_12rep': np.array([i for i, s in enumerate(ids) if s.startswith('shared_12rep_')]),
              'shared_4rep': np.array([i for i, s in enumerate(ids[:1121]) if s.startswith('shared_4rep_')])}
    correlations = {key: np.empty((args.n_splits, n_voxels)) for key in groups}
    rng = np.random.default_rng(args.seed + SUBJECTS.index(subject_id))
    half_n = counts // 2
    for iteration in range(args.n_splits):
        half_a = np.zeros_like(sums)
        for n in np.unique(counts):
            which = np.flatnonzero(counts == n)
            selected = np.argsort(rng.random((len(which), int(n))), axis=1)[:, :int(n) // 2]
            # Limit temporaries for the 12-repeat images.
            for start in range(0, len(which), 64):
                rows = which[start:start + 64]
                reps = selected[start:start + 64]
                half_a[rows] = repetitions[rows[:, None], reps].sum(axis=1, dtype=np.float64)
        half_b = (sums - half_a) / (counts - half_n)[:, None]
        half_a /= half_n[:, None]
        for key, which in groups.items():
            correlations[key][iteration] = voxel_pearson(half_a[which], half_b[which])
        if (iteration + 1) % 20 == 0:
            print(f'{subject_id}: reliability split {iteration + 1}/{args.n_splits}', flush=True)
    summaries, split_rows = [], []
    folder = args.output_root / 'voxelwise'
    folder.mkdir(exist_ok=True)
    for key, which in groups.items():
        r = correlations[key]
        per_split = np.nanmedian(r, axis=1)
        voxel_mean = np.nanmean(r, axis=0)
        pd.DataFrame({'neuroid': np.arange(n_voxels),
                      'mean_split_half_r': voxel_mean,
                      'sd_split_half_r': np.nanstd(r, axis=0, ddof=1),
                      'n_valid_splits': np.isfinite(r).sum(axis=0)}).to_csv(
                          folder / f'{subject_id}_{key}.csv.gz', index=False)
        summaries.append({'subject': subject_id, 'roi': ROI, 'evaluation_set': key,
                          'n_images': len(which), 'n_voxels': n_voxels,
                          'n_random_splits': args.n_splits,
                          'min_actual_repetitions': int(counts[which].min()),
                          'max_actual_repetitions': int(counts[which].max()),
                          'mean_actual_repetitions': float(counts[which].mean()),
                          'mean_over_splits_of_median_voxel_r': float(per_split.mean()),
                          'split_variability_p025': float(np.quantile(per_split, .025)),
                          'split_variability_p975': float(np.quantile(per_split, .975)),
                          'median_over_voxels_of_mean_split_r': float(np.nanmedian(voxel_mean)),
                          'mean_over_voxels_of_mean_split_r': float(np.nanmean(voxel_mean)),
                          'n_voxels_with_any_undefined_r': int((~np.isfinite(r)).any(axis=0).sum())})
        split_rows.extend({'subject': subject_id, 'evaluation_set': key, 'split_index': i,
                           'median_voxel_r': float(per_split[i]),
                           'mean_voxel_r': float(np.nanmean(r[i]))} for i in range(args.n_splits))
    pd.DataFrame({'subject': subject_id, 'stimulus_id': ids,
                  'n_repetitions': counts, 'half_a_n': half_n,
                  'half_b_n': counts - half_n}).to_csv(
                      args.output_root / f'repetition_counts_{subject_id}.csv', index=False)
    return summaries, split_rows, validation, session_counts


def reliability(args):
    summaries, split_rows, validations, sessions = [], [], [], []
    with ProcessPoolExecutor(max_workers=min(args.workers, 5)) as executor:
        futures = [executor.submit(subject_reliability, (args, s)) for s in SUBJECTS]
        for future in as_completed(futures):
            a, b, c, d = future.result()
            summaries.extend(a); split_rows.extend(b); validations.extend(c); sessions.extend(d)
    for name, rows, keys in [
        ('reliability_summary.csv', summaries, ['subject', 'evaluation_set']),
        ('reliability_random_splits.csv', split_rows, ['subject', 'evaluation_set', 'split_index']),
        ('beta_reconstruction_validation.csv', validations, ['subject', 'mode']),
        ('sessions_used.csv', sessions, ['subject', 'session']),
    ]:
        pd.DataFrame(rows).sort_values(keys).to_csv(args.output_root / name, index=False)
    write_json(args.output_root / 'provenance.json', {
        'data_root': str(args.data_root), 'pca_root': str(args.pca_root),
        'beta_root': str(args.beta_root), 'split_pool': 'shared', 'split_variant_id': 0,
        'seed_base': args.seed, 'n_random_splits': args.n_splits,
        'subject_order': SUBJECTS, 'subject_seed': 'seed_base + index_in_subject_order',
        'packages': {name: importlib.metadata.version(name) for name in
                     ['numpy', 'pandas', 'xarray', 'h5py', 'nibabel', 'laion-fmri']},
        'beta_validation': 'All ten reconstructed arrays passed atol=2e-5, rtol=2e-5.',
    })



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['splits', 'extract', 'reliability', 'all'])
    parser.add_argument('--data-root', type=Path, default=Path('data/laion_fmri'))
    parser.add_argument('--pca-root', type=Path, default=Path('data/pca'))
    parser.add_argument('--beta-root', type=Path, default=Path('data/betas/roi=laion-general'))
    parser.add_argument('--cache-root', type=Path, default=Path('data/cache/generalization_single_trials'))
    parser.add_argument('--output-root', type=Path, default=Path('outputs/reliability'))
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--n-splits', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20260912)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    assert args.n_splits >= 2 and args.workers >= 1
    for name, func in [('splits', export_splits), ('extract', extract),
                       ('reliability', reliability)]:
        if args.stage in [name, 'all']:
            func(args)


if __name__ == '__main__':
    main()
