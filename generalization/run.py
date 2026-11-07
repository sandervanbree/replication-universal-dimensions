#!/usr/bin/env python3
"""Add held-out-prediction permutation inference without changing the primary run."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import fcntl
from functools import lru_cache
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.replication.config import load_config
CONFIG = Path(os.environ.get('REPLICATION_CONFIG', REPO / 'config/analysis.json')).resolve()
ROOT = load_config(CONFIG).output_root / 'generalization'
THREADS = int(os.environ.get('GENERALIZATION_THREADS', '2'))
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = str(THREADS)
os.environ.update(PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR=str(ROOT/'runtime/matplotlib'),
                  BONNER_CACHING_HOME=str(ROOT/'runtime/bonner'),
                  BONNER_DATASETS_HOME=str(ROOT/'runtime/datasets'),
                  BONNER_MODELS_HOME=str(ROOT/'runtime/models'),
                  CUDA_VISIBLE_DEVICES='')
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import torch
from src.replication.config import load_config
from src.replication.data import Inventory, load_evaluation_split, load_model_predictor
from src.replication.scoring import load_regression_module, universality_path


def now():
    return datetime.now(timezone.utc).isoformat()


def key_for(*values):
    return hashlib.sha256('\0'.join(map(str, values)).encode()).hexdigest()[:24]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def npy_write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:
        np.save(f, value, allow_pickle=False)
    tmp.replace(path)


def npz_write(path, **values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:
        np.savez(f, **values)
    tmp.replace(path)


@lru_cache(None)
def context():
    cfg = load_config(CONFIG)
    inv = Inventory(cfg.pca_root)
    ids = {m: np.asarray((cfg.pca_root/'stimulus_ids'/f'{m}.txt').read_text().splitlines())
           for m in ['shared', 'ood']}
    return cfg, inv, ids


@lru_cache(None)
def settings():
    return json.loads((ROOT/'configuration.json').read_text())


@lru_cache(None)
def split_indices(mode):
    cfg, inv, ids = context()
    split = load_evaluation_split(cfg, mode)
    def positions(source, selected):
        lookup = {v: i for i, v in enumerate(ids[source])}
        return np.asarray([lookup[v] for v in selected], dtype=int)
    assert not set(split.train_ids) & set(split.test_ids)
    return split, positions(split.train_source, split.train_ids), positions(split.test_source, split.test_ids)


def feature_cache(model, source):
    return ROOT/'cache/features'/f'{key_for(model)}.{source}.npy'


def decomposition_cache(model, mode):
    return ROOT/'cache/decompositions'/mode/f'{key_for(model)}.npz'


def task_paths(task):
    mode, model_set, model, node = task
    stem = key_for(model, node)
    return {
        'predictions': ROOT/'cache/predictions'/mode/model_set/f'{stem}.npy',
        'prediction_validation': ROOT/'diagnostics/predictions'/mode/model_set/f'{stem}.json',
        'null': ROOT/'null_universality'/mode/model_set/f'{stem}.npy',
        'done': ROOT/'checkpoints'/mode/model_set/f'{stem}.json',
    }


def worker_init():
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)


def feature_job(model):
    cfg, inv, ids = context()
    for source in ['shared', 'ood']:
        path = feature_cache(model, source)
        if path.exists():
            continue
        x = load_model_predictor(cfg, inv, model, source, ids[source])
        npy_write(path, np.asarray(x.values, dtype=np.float32))
    return {'model': model}


def decomposition_job(task):
    mode, model = task
    path = decomposition_cache(model, mode)
    if path.exists():
        return {'cached': True}
    cfg, inv, ids = context()
    split, train_idx, test_idx = split_indices(mode)
    x = np.array(np.load(feature_cache(model, split.train_source), mmap_mode='r')[train_idx], copy=True)
    module = load_regression_module()
    helper = module.RidgeGCVRegression(alphas=list(cfg.alphas), device='cpu')
    with torch.no_grad():
        xt = torch.from_numpy(x)
        xt -= xt.mean(dim=0)
        x_mean, eigvals, q, _ = helper._eigen_decompose_gram(
            xt, torch.zeros((len(xt), 1)), torch.ones(len(xt)))
    npz_write(path, x_mean=x_mean.numpy(), eigvals=eigvals.numpy(), q=q.numpy())
    return {'model': model, 'mode': mode, 'n_train': len(train_idx)}


def target_arrays(model, node, mode):
    cfg, inv, ids = context()
    split, train_idx, test_idx = split_indices(mode)
    with np.load(cfg.pca_root/'layers'/model/f'node={node}'/'pca.npz', allow_pickle=False) as f:
        train = f[f'{split.train_source}_scores'][train_idx]
        test = f[f'{split.test_source}_scores'][test_idx]
    return np.array(train, copy=True), np.array(test, copy=True)


def predictor_arrays(model, mode):
    split, train_idx, test_idx = split_indices(mode)
    train = np.array(np.load(feature_cache(model, split.train_source), mmap_mode='r')[train_idx], copy=True)
    test = np.array(np.load(feature_cache(model, split.test_source), mmap_mode='r')[test_idx], copy=True)
    return train, test


def predict(model, mode, y_train, *, cached=True):
    cfg, inv, ids = context()
    module = load_regression_module()
    x_train, x_test = predictor_arrays(model, mode)
    cls = module.RidgeGCVRegression
    if cached:
        with np.load(decomposition_cache(model, mode), allow_pickle=False) as f:
            # eigh returns column-major eigenvectors. Preserve that layout so
            # BLAS uses the same reduction order as the uncached estimator.
            x_mean, eigvals, q = [torch.from_numpy(f[k].copy(order='K')) for k in ['x_mean', 'eigvals', 'q']]
        class CachedRidge(cls):
            def _eigen_decompose_gram(self, x, y, sqrt_sw):
                return x_mean, eigvals, q, torch.matmul(q.T, y)
        cls = CachedRidge
    estimator = cls(alphas=list(cfg.alphas), device='cpu')
    with torch.no_grad():
        estimator.fit(torch.from_numpy(x_train), torch.from_numpy(y_train.copy()))
        predicted = estimator.predict(torch.from_numpy(x_test)).numpy()
    return predicted, estimator.alpha_.numpy()


def correlation(x, y):
    # Match the observed transfer scorer's float32 reductions.
    x = np.array(x, dtype=np.float32, copy=True); y = np.array(y, dtype=np.float32, copy=True)
    x -= x.mean(axis=0, keepdims=True); y -= y.mean(axis=0, keepdims=True)
    return (x*y).sum(axis=0) / np.sqrt((x*x).sum(axis=0)*(y*y).sum(axis=0))


def unit_columns(x):
    # Permuting sample rows preserves means/norms. Normalize once in double
    # precision, then use float32 batched matrix products for the null scores.
    x = np.array(x, dtype=np.float64, copy=True)
    x -= x.mean(axis=0, keepdims=True)
    x /= np.sqrt((x*x).sum(axis=0, keepdims=True))
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite normalized predictions/targets')
    return x.astype(np.float32)


def null_batch(predictions_dpn, target_dn, inverse_permutations):
    # sum_j prediction[permutation[j]] * target[j] equals
    # sum_j prediction[j] * target[inverse_permutation[j]].
    # take keeps each PC's samples together. The transpose is a BLAS-compatible
    # view, avoiding a second large shuffled-data allocation and copy.
    target = np.take(target_dn, inverse_permutations, axis=1).transpose(0, 2, 1)
    scores = predictions_dpn @ target
    return np.median(scores, axis=1).T.copy()


def layer_job(task):
    started = time.perf_counter()
    cfg, inv, ids = context()
    mode, model_set, model, node = task
    paths = task_paths(task)
    b = settings()['n_permutations']
    if paths['done'].exists() and paths['null'].exists():
        meta = json.loads(paths['done'].read_text())
        assert meta['n_permutations'] == b
        return {**meta, 'reused': True}
    record = inv.layer(model, node)
    y_train, y_test = target_arrays(model, node, mode)
    predictors = [m for m in inv.models(model_set) if m != model]
    if paths['predictions'].exists() and paths['prediction_validation'].exists():
        predictions = np.load(paths['predictions'])
        validation = json.loads(paths['prediction_validation'].read_text())
    else:
        predictions = np.empty((record.rank, len(predictors), len(y_test)), dtype=np.float32)
        checks = []
        for k, predictor in enumerate(predictors):
            yp, alphas = predict(predictor, mode, y_train)
            observed = correlation(yp, y_test)
            old_path = universality_path(cfg, mode, model_set, record, predictor)
            with np.load(old_path, allow_pickle=False) as f:
                expected = f['correlation'][0]
            error = float(np.max(np.abs(observed-expected)))
            if not np.isfinite(observed).all() or error > 2e-5:
                raise ValueError(f'Observed-score mismatch {task} predictor={predictor}: max_abs={error}')
            predictions[:, k, :] = unit_columns(yp).T
            checks.append({'predictor': predictor, 'max_abs_observed_difference': error})
        validation = {'task': list(task), 'checks': checks,
                      'max_abs_observed_difference': max(c['max_abs_observed_difference'] for c in checks)}
        npy_write(paths['predictions'], predictions)
        json_write(paths['prediction_validation'], validation)
    assert predictions.shape == (record.rank, len(predictors), len(y_test))
    predictions = np.ascontiguousarray(predictions)
    yn = np.ascontiguousarray(unit_columns(y_test).T)
    inverse = np.load(ROOT/'permutations'/f'{mode}.inverse.npy', mmap_mode='r')
    paths['null'].parent.mkdir(parents=True, exist_ok=True)
    tmp = paths['null'].with_name(paths['null'].name+f'.{os.getpid()}.tmp')
    result = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.float32, shape=(b, record.rank))
    batch_size = min(128, max(4, int(128*1024**2/(4*record.rank*len(y_test)))))
    for start in range(0, b, batch_size):
        stop = min(start+batch_size, b)
        values = null_batch(predictions, yn, inverse[start:stop])
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite null scores {task}')
        result[start:stop] = values
    result.flush(); del result
    tmp.replace(paths['null'])
    meta = {'task': list(task), 'n_permutations': b, 'rank': record.rank,
            'n_test': len(y_test), 'n_predictors': len(predictors),
            'seconds': time.perf_counter()-started,
            'max_abs_observed_difference': validation['max_abs_observed_difference'], 'completed': now()}
    json_write(paths['done'], meta)
    return meta


def score_ranks(x, y_rank, pc_rank):
    xr = rankdata(x, axis=1).astype(np.float64)
    xr -= xr.mean(axis=1, keepdims=True)
    xr /= np.linalg.norm(xr, axis=1, keepdims=True)
    r = xr @ y_rank
    xc = xr @ pc_rank
    yc = pc_rank @ y_rank
    partial = (r-xc[:, None]*yc[None, :]) / np.sqrt((1-xc[:, None]**2)*(1-yc[None, :]**2))
    return r, partial


def p_value(observed, null):
    exceed = (null >= observed).sum(axis=0)
    return exceed, (exceed+1)/(len(null)+1)


def metric_rows(mode, model_set, observed, partial, null, null_partial, subjects):
    rows = []
    for i, subject in enumerate(subjects):
        exceed, p = p_value(observed[i], null[:, i])
        exceed_p, pp = p_value(partial[i], null_partial[:, i])
        rows.append({'mode': mode, 'model_set': model_set, 'subject': subject,
                     'spearman_rho': float(observed[i]), 'partial_spearman_rho_control_pc': float(partial[i]),
                     'spearman_exceedances': int(exceed), 'spearman_p_permutation': float(p),
                     'partial_spearman_exceedances': int(exceed_p), 'partial_spearman_p_permutation': float(pp),
                     'n_permutations': len(null), 'alternative': 'greater'})
    return rows


def summary_job(task):
    mode, model_set = task
    cfg, inv, ids = context()
    path = cfg.analysis_dir/'summaries'/f'mode={mode}'/f'model_set={model_set}'/'dimensions.csv'
    frame = pd.read_csv(path)
    columns = [f'brain_similarity_{s}' for s in cfg.subjects]
    y = rankdata(frame[columns].to_numpy(), axis=0)
    y -= y.mean(axis=0); y /= np.linalg.norm(y, axis=0)
    pc = rankdata(frame.pc.to_numpy()); pc -= pc.mean(); pc /= np.linalg.norm(pc)
    obs, part = score_ranks(frame.universality.to_numpy()[None, :], y, pc)
    blocks = []
    expected_order = []
    for (model, node), group in frame.groupby(['model', 'node'], sort=False):
        indices = group.pc.to_numpy(dtype=int)-1
        expected_order.extend(group.index.tolist())
        arr = np.load(task_paths((mode, model_set, model, node))['null'], mmap_mode='r')
        assert arr.shape == (settings()['n_permutations'], len(group))
        blocks.append((arr, indices))
    assert expected_order == list(range(len(frame))), 'Observed/null PC order differs'
    b = settings()['n_permutations']
    null = np.empty((b, len(cfg.subjects)), dtype=float); nullp = np.empty_like(null)
    for start in range(0, b, 32):
        stop = min(start+32, b)
        values = np.concatenate([a[start:stop, idx] for a,idx in blocks], axis=1)
        null[start:stop], nullp[start:stop] = score_ranks(values, y, pc)
    original = pd.read_csv(cfg.tables_dir/'generalization_correlations_by_split.csv')
    original = original[(original['mode']==mode)&(original.model_set==model_set)].set_index('subject').loc[list(cfg.subjects)]
    np.testing.assert_allclose(obs[0], original.spearman_rho, atol=2e-12, rtol=0)
    np.testing.assert_allclose(part[0], original.partial_spearman_rho_control_pc, atol=2e-12, rtol=0)
    out = ROOT/'null_statistics'/mode/f'{model_set}.npz'
    npz_write(out, spearman=null, partial_spearman=nullp, observed=obs[0], observed_partial=part[0])
    rows = metric_rows(mode, model_set, obs[0], part[0], null, nullp, cfg.subjects)
    json_write(ROOT/'diagnostics/statistics'/mode/f'{model_set}.json', rows)
    return {'mode': mode, 'model_set': model_set, 'n_dimensions': len(frame)}


def bh(values):
    values=np.asarray(values); order=np.argsort(values); ranked=values[order]
    adjusted=np.minimum.accumulate((ranked*len(ranked)/np.arange(1,len(ranked)+1))[::-1])[::-1]
    result=np.empty_like(adjusted); result[order]=np.minimum(adjusted,1)
    return result


def add_q(frame):
    for prefix in ['spearman', 'partial_spearman']:
        frame[f'{prefix}_q_bh'] = bh(frame[f'{prefix}_p_permutation'].to_numpy())
    return frame


def finish():
    cfg, inv, ids = context()
    rows=[]; cluster_rows=[]; mean_rows=[]
    for model_set in cfg.model_sets:
        cluster=[]
        for mode in cfg.transfer_modes:
            rows.extend(json.loads((ROOT/'diagnostics/statistics'/mode/f'{model_set}.json').read_text()))
            with np.load(ROOT/'null_statistics'/mode/f'{model_set}.npz') as f:
                a={k:f[k].copy() for k in f.files}
            mean_rows.extend(metric_rows(mode,model_set,np.array([a['observed'].mean()]),
                np.array([a['observed_partial'].mean()]),a['spearman'].mean(axis=1,keepdims=True),
                a['partial_spearman'].mean(axis=1,keepdims=True),['mean_across_participants']))
            if 'cluster_k5' in mode: cluster.append(a)
        assert len(cluster)==5
        averaged={k:np.mean([a[k] for a in cluster],axis=0) for k in cluster[0]}
        cluster_rows.extend(metric_rows('generalization_clip_cluster_average',model_set,
            averaged['observed'],averaged['observed_partial'],averaged['spearman'],averaged['partial_spearman'],cfg.subjects))
        npz_write(ROOT/'null_statistics/generalization_clip_cluster_average'/f'{model_set}.npz',**averaged)
        mean_rows.extend(metric_rows('generalization_clip_cluster_average',model_set,
            np.array([averaged['observed'].mean()]),np.array([averaged['observed_partial'].mean()]),
            averaged['spearman'].mean(axis=1,keepdims=True),averaged['partial_spearman'].mean(axis=1,keepdims=True),
            ['mean_across_participants']))
    out=ROOT/'tables';out.mkdir(exist_ok=True)
    all_splits=add_q(pd.DataFrame(rows)); assert len(all_splits)==140
    all_splits.to_csv(out/'participant_pvalues_all_7_splits.csv',index=False)
    clusters=add_q(pd.DataFrame(cluster_rows)); assert len(clusters)==20
    clusters.to_csv(out/'participant_pvalues_clip_cluster_average.csv',index=False)
    main=pd.concat([all_splits[all_splits['mode'].isin(['generalization_tau','generalization_ood'])],clusters],ignore_index=True)
    main=add_q(main);assert len(main)==60
    main.to_csv(out/'participant_pvalues_report_3_tests.csv',index=False)
    add_q(pd.DataFrame(mean_rows)).to_csv(out/'mean_participant_effect_pvalues.csv',index=False)
    checks=[json.loads(task_paths(t)['done'].read_text()) for t in all_layer_tasks()]
    diagnostics={'state':'complete','finished':now(),'n_layer_split_tests':len(checks),
                 'n_participant_split_tests':len(all_splits),'n_participant_report_tests':len(main),
                 'n_permutations':settings()['n_permutations'],
                 'max_abs_observed_score_difference':max(c['max_abs_observed_difference'] for c in checks),
                 'max_spearman_p_all_splits':float(all_splits.spearman_p_permutation.max()),
                 'max_partial_p_all_splits':float(all_splits.partial_spearman_p_permutation.max()),
                 'max_spearman_p_report_tests':float(main.spearman_p_permutation.max()),
                 'max_partial_p_report_tests':float(main.partial_spearman_p_permutation.max())}
    for item in settings()['source_hashes']:
        assert sha256(item['path'])==item['sha256'],f'Source changed: {item["path"]}'
    json_write(ROOT/'diagnostics/completion.json',diagnostics)
    print(json.dumps(diagnostics,indent=2),flush=True)


def all_layer_tasks():
    cfg,inv,ids=context()
    tasks=[(mode,ms,r.model,r.node) for mode in cfg.transfer_modes for ms in cfg.model_sets for r in inv.records(ms)]
    return sorted(tasks,key=lambda t:inv.layer(t[2],t[3]).rank*(len(inv.models(t[1]))-1)*len(split_indices(t[0])[2]),reverse=True)


def parallel_stage(name, function, tasks, workers):
    started=time.perf_counter();finished=0;pending={};iterator=iter(tasks)
    json_write(ROOT/'status.json',{'state':'running','stage':name,'completed':0,'total':len(tasks),'started':now()})
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=worker_init) as pool:
        for _ in range(min(len(tasks),workers*2)):
            task=next(iterator);pending[pool.submit(function,task)]=task
        while pending:
            done,_=wait(pending,timeout=15,return_when=FIRST_COMPLETED)
            for future in done:
                task=pending.pop(future)
                try: result=future.result()
                except BaseException:
                    json_write(ROOT/'diagnostics/failure.json',{'stage':name,'task':task,'traceback':traceback.format_exc(),'time':now()})
                    for f in pending:f.cancel()
                    raise
                finished+=1
                if finished%25==0 or finished==len(tasks):
                    print(f'{now()} {name} {finished}/{len(tasks)} elapsed={time.perf_counter()-started:.1f}s',flush=True)
                try: new=next(iterator)
                except StopIteration: pass
                else: pending[pool.submit(function,new)]=new
            json_write(ROOT/'status.json',{'state':'running','stage':name,'completed':finished,'total':len(tasks),
                       'elapsed_seconds':time.perf_counter()-started,'updated':now()})


def prepare(n_permutations, seed, workers):
    cfg,inv,ids=context()
    for folder in ['runtime','cache','diagnostics','logs','tables','permutations','checkpoints','null_statistics']:
        (ROOT/folder).mkdir(exist_ok=True)
    sources=[CONFIG,REPO/'orig_codebase/src/lib/scorers/_regression.py',
             REPO/'src/replication/scoring.py',cfg.generalization_splits_path,
             cfg.tables_dir/'generalization_correlations_by_split.csv',
             cfg.tables_dir/'generalization_cluster_summary.csv',cfg.pca_root/'layer_manifest.csv']
    configuration={'n_permutations':n_permutations,'seed':seed,'threads_per_worker':THREADS,
        'source_run':str(cfg.output_root),'resolved_pca_root':str(cfg.pca_root),
        'alternative':'greater','null':'shuffle held-out DNN predictions across images; fixed observed brain similarity',
        'cluster_summary':'arithmetic mean of five fold correlations, with corresponding mean null per permutation',
        'source_hashes':[{'path':str(p),'sha256':sha256(p)} for p in sources]}
    path=ROOT/'configuration.json'
    if path.exists():assert json.loads(path.read_text())==configuration,'Configuration differs; use a new folder'
    else:json_write(path,configuration)
    json_write(ROOT/'diagnostics/environment.json', {
        'runner_sha256': sha256(Path(__file__)),
        'python': sys.version,
        'packages': {'numpy': np.__version__, 'pandas': pd.__version__, 'torch': torch.__version__},
    })
    metadata=[]
    for mode_index,mode in enumerate(cfg.transfer_modes):
        split,train_idx,test_idx=split_indices(mode)
        # Independent random streams by split; shared permutations across every
        # predictor, target layer and participant inside a split.
        rng=np.random.default_rng(np.random.SeedSequence([seed,mode_index]))
        permutations=np.stack([rng.permutation(len(test_idx)) for _ in range(n_permutations)])
        path=ROOT/'permutations'/f'{mode}.npy'
        if path.exists():np.testing.assert_array_equal(np.load(path),permutations)
        else:npy_write(path,permutations)
        npy_write(ROOT/'permutations'/f'{mode}.inverse.npy',np.argsort(permutations,axis=1))
        metadata.append({'mode':mode,'n_train':len(train_idx),'n_test':len(test_idx),
                         'train_ids':list(split.train_ids),'test_ids':list(split.test_ids),
                         'permutation_sha256':sha256(path)})
    json_write(ROOT/'split_manifest.json',metadata)
    models=list(dict.fromkeys(r.model for r in inv.layers))
    parallel_stage('cache_features',feature_job,models, min(workers,24))
    parallel_stage('cache_decompositions',decomposition_job,[(mode,m) for mode in cfg.transfer_modes for m in models],workers)
    json_write(ROOT/'status.json',{'state':'prepared','n_permutations':n_permutations,'time':now()})


def smoke():
    cfg,inv,ids=context();module=load_regression_module();records=[]
    samples=[('resnet18_classification_imagenet1k_varied_seeds',64),
             ('classification_imagenet1k_varied_architectures',3072),
             ('resnet50_imagenet1k_varied_tasks',2048),('untrained_resnet18_varied_seeds',512)]
    for i,(ms,rank) in enumerate(samples):
        rec=min(inv.records(ms),key=lambda r:abs(r.rank-rank))
        mode=cfg.transfer_modes[-1] if i%2 else cfg.transfer_modes[0]
        predictor=next(m for m in inv.models(ms) if m!=rec.model)
        yt,ye=target_arrays(rec.model,rec.node,mode)
        start=time.perf_counter()
        predicted,alpha=predict(predictor,mode,yt,cached=True)
        ref,ref_alpha=predict(predictor,mode,yt,cached=False)
        np.testing.assert_allclose(predicted,ref,atol=1e-5,rtol=2e-5)
        np.testing.assert_array_equal(alpha,ref_alpha)
        observed=correlation(predicted,ye)
        with np.load(universality_path(cfg,mode,ms,rec,predictor)) as f:old=f['correlation'][0]
        np.testing.assert_allclose(observed,old,atol=2e-5,rtol=0)
        # Reference routine permutes predictions, resets RNG, and shares indices
        # across target columns, just as the new vectorized calculation does.
        seed=41; rng=np.random.default_rng(seed)
        p=np.stack([rng.permutation(len(ye)) for _ in range(31)])
        ncol=min(32,rec.rank)
        original=module._compute_metric('correlation',[torch.from_numpy(ye[:,:ncol].copy())],
            [torch.from_numpy(predicted[:,:ncol].copy())],n_permutations=31,seed=seed).numpy()[:,0,:]
        fast=null_batch(np.ascontiguousarray(unit_columns(predicted[:,:ncol]).T[:,None,:]),
                        np.ascontiguousarray(unit_columns(ye[:,:ncol]).T),np.argsort(p,axis=1))
        np.testing.assert_allclose(fast,original,atol=2e-6,rtol=2e-5)
        # Two distinct predictors exercise the even-count median as well.
        pred2=np.stack([unit_columns(predicted[:,:ncol]),unit_columns(ref[:,:ncol][::-1])])
        naive=np.stack([np.median(np.stack([correlation(a[perm],ye[:,:ncol]) for a in pred2]),axis=0) for perm in p])
        fast2=null_batch(np.ascontiguousarray(pred2.transpose(2,0,1)),
                         np.ascontiguousarray(unit_columns(ye[:,:ncol]).T),np.argsort(p,axis=1))
        np.testing.assert_allclose(fast2,naive,atol=2e-6,rtol=2e-5)
        row={'mode':mode,'model_set':ms,'target_model':rec.model,'target_node':rec.node,
             'rank':rec.rank,'predictor':predictor,'max_original_score_difference':float(np.max(abs(observed-old))),
             'max_reference_null_difference':float(np.max(abs(fast-original))),'seconds':time.perf_counter()-start}
        records.append(row);print(row,flush=True)
    # Validate partial correlations including tied PC ranks against residualization.
    rng=np.random.default_rng(1);x=rng.normal(size=(7,100));y=rng.normal(size=(100,5));pc=np.tile(np.arange(20),5)
    yr=rankdata(y,axis=0);yr-=yr.mean(axis=0);yr/=np.linalg.norm(yr,axis=0)
    cr=rankdata(pc);cr-=cr.mean();cr/=np.linalg.norm(cr)
    rr,rp=score_ranks(x,yr,cr)
    from src.replication.summary import _partial_spearman
    for i in range(7):
        for j in range(5):np.testing.assert_allclose(rp[i,j],_partial_spearman(x[i],y[:,j],pc),atol=2e-14)
    json_write(ROOT/'diagnostics/smoke.json',{'state':'passed','checks':records,'time':now()})
    json_write(ROOT/'status.json',{'state':'validated','n_permutations':settings()['n_permutations'],'time':now()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','smoke','run','summarize'])
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--permutations',type=int,default=5000)
    parser.add_argument('--seed',type=int,default=20261022)
    args=parser.parse_args()
    assert args.workers*THREADS<=len(os.sched_getaffinity(0))
    worker_init()
    ROOT.mkdir(parents=True, exist_ok=True)
    lock=(ROOT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        if args.command=='prepare':prepare(args.permutations,args.seed,args.workers)
        elif args.command=='smoke':smoke()
        else:
            assert json.loads((ROOT/'diagnostics/smoke.json').read_text())['state']=='passed'
            if args.command=='run':
                parallel_stage('layer_nulls',layer_job,all_layer_tasks(),args.workers)
            cfg,inv,ids=context()
            parallel_stage('correlation_nulls',summary_job,[(mode,ms) for mode in cfg.transfer_modes for ms in cfg.model_sets],min(args.workers,28))
            finish()
            json_write(ROOT/'status.json',{'state':'complete','finished':now()})
    except BaseException:
        json_write(ROOT/'status.json',{'state':'failed','time':now(),'traceback':traceback.format_exc()})
        raise


if __name__=='__main__':
    main()
