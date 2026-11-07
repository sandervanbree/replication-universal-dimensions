from __future__ import annotations

from dataclasses import asdict
import gc
import importlib
from functools import cache
from pathlib import Path
import platform
import sys
import types
from typing import Iterator

import numpy as np
import xarray as xr

from .config import REPO_ROOT, RunConfig
from .data import (
    Inventory,
    LayerRecord,
    align_rows,
    load_beta,
    load_evaluation_split,
    load_model_predictor,
    load_target_scores,
    save_npy,
    save_npz,
    write_json,
)


def transfer_definition(config: RunConfig, mode: str):
    split = load_evaluation_split(config, mode)
    if split is None:
        return config.transfer_train_mode, None, mode, None
    return split.train_source, split.train_ids, split.test_source, split.test_ids


@cache
def load_regression_module():
    if int(np.__version__.split(".", 1)[0]) >= 2:
        raise RuntimeError(
            "The full scorer requires numpy<2 with torch 2.3. Install requirements.txt."
        )
    paths = [
        REPO_ROOT / "orig_codebase" / "src",
    ]
    for path in reversed(paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    package = types.ModuleType("lib.scorers")
    package.__path__ = [str(REPO_ROOT / "orig_codebase" / "src" / "lib" / "scorers")]
    sys.modules.setdefault("lib.scorers", package)
    return importlib.import_module("lib.scorers._regression")


def load_regression_scorer():
    return load_regression_module().RegressionScorer


def make_scorer(config: RunConfig, *, permutations: bool):
    scorer_class = load_regression_scorer()
    return scorer_class(
        regression=config.regression,
        n_folds=config.n_folds,
        metric="correlation",
        shuffle=True,
        seed=config.seed,
        device=config.device,
        alphas=list(config.alphas),
        n_permutations=config.n_permutations if permutations else None,
    )


def score_arrays(
    scorer, predictor: xr.DataArray, target: xr.DataArray
) -> dict[str, np.ndarray]:
    target_ids = target.stimulus_id.values.astype(str)
    predictor = xr.DataArray(
        align_rows(predictor.stimulus_id.values, predictor.values, target_ids),
        dims=("presentation", "neuroid"),
        coords={
            "stimulus_id": ("presentation", target_ids),
            "neuroid": predictor.neuroid.values,
        },
        name=predictor.name,
    )
    result = scorer(predictor=predictor, target=target)
    arrays = {"correlation": np.asarray(result["correlation"].values, dtype=np.float32)}
    null_name = f"null.correlation.n_permutations={scorer.n_permutations}"
    if scorer.n_permutations is not None:
        arrays["null_correlation"] = np.asarray(
            result[null_name].values, dtype=np.float32
        )
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Scorer returned non-finite values")
    return arrays


def score_transfer_arrays(
    config: RunConfig,
    train_predictor: xr.DataArray,
    train_target: xr.DataArray,
    test_predictor: xr.DataArray,
    test_target: xr.DataArray,
) -> dict[str, np.ndarray]:
    module = load_regression_module()
    torch = module.torch
    train_ids = train_target.stimulus_id.values.astype(str)
    test_ids = test_target.stimulus_id.values.astype(str)
    x_train = align_rows(
        train_predictor.stimulus_id.values, train_predictor.values, train_ids
    )
    x_test = align_rows(
        test_predictor.stimulus_id.values, test_predictor.values, test_ids
    )
    model = module.RidgeGCVRegression(alphas=list(config.alphas), device=config.device)
    with torch.no_grad():
        model.fit(
            torch.as_tensor(x_train, dtype=torch.float32, device=config.device),
            torch.as_tensor(
                train_target.values, dtype=torch.float32, device=config.device
            ),
        )
        predicted = (
            model.predict(
                torch.as_tensor(x_test, dtype=torch.float32, device=config.device)
            )
            .cpu()
            .numpy()
        )
    expected = np.array(test_target.values, dtype=np.float32, copy=True)
    predicted -= predicted.mean(axis=0, keepdims=True)
    expected -= expected.mean(axis=0, keepdims=True)
    denominator = np.sqrt(
        np.sum(predicted * predicted, axis=0) * np.sum(expected * expected, axis=0)
    )
    correlation = np.divide(
        np.sum(predicted * expected, axis=0),
        denominator,
        out=np.full(expected.shape[1], np.nan, dtype=np.float32),
        where=denominator > 0,
    )
    if not np.isfinite(correlation).all():
        raise ValueError("Transfer scorer returned non-finite values")
    return {"correlation": correlation[None, :]}


def brain_path(
    config: RunConfig, mode: str, model_set: str, record: LayerRecord, subject: str
) -> Path:
    return (
        config.analysis_dir
        / "brain_similarity"
        / f"mode={mode}"
        / f"model_set={model_set}"
        / f"model={record.model}"
        / f"node={record.node}"
        / f"subject={subject}.npz"
    )


def universality_path(
    config: RunConfig,
    mode: str,
    model_set: str,
    record: LayerRecord,
    predictor: str,
) -> Path:
    return (
        config.analysis_dir
        / "universality"
        / f"mode={mode}"
        / f"model_set={model_set}"
        / f"target={record.model}"
        / f"node={record.node}"
        / f"predictor={predictor}.npz"
    )


def universality_null_path(
    config: RunConfig, mode: str, model_set: str, record: LayerRecord
) -> Path:
    return (
        config.analysis_dir
        / "universality"
        / f"mode={mode}"
        / f"model_set={model_set}"
        / f"target={record.model}"
        / f"node={record.node}"
        / "null_median_across_predictors.npy"
    )


def target_tasks(
    config: RunConfig, inventory: Inventory
) -> list[tuple[str, str, LayerRecord]]:
    return [
        (mode, model_set, record)
        for mode in config.modes
        for model_set in config.model_sets
        for record in inventory.records(model_set)
    ]


def shard(items: list, index: int, count: int) -> Iterator[tuple[int, object]]:
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"Invalid shard {index}/{count}")
    for task_index, item in enumerate(items):
        if task_index % count == index:
            yield task_index, item


def run_brain(
    config: RunConfig,
    inventory: Inventory,
    *,
    shard_index: int,
    shard_count: int,
    overwrite: bool,
) -> None:
    scorer = make_scorer(config, permutations=False)
    beta_cache: dict[tuple[str, str], xr.DataArray] = {}
    tasks = target_tasks(config, inventory)
    for task_index, task in shard(tasks, shard_index, shard_count):
        mode, model_set, record = task
        pending = [
            subject
            for subject in config.subjects
            if overwrite
            or not brain_path(config, mode, model_set, record, subject).is_file()
        ]
        if not pending:
            continue
        if mode in config.transfer_modes:
            train_source, train_ids, test_source, test_ids = transfer_definition(
                config, mode
            )
            train_target = load_target_scores(
                config, record, train_source, train_ids
            )
            target = load_target_scores(config, record, test_source, test_ids)
        else:
            train_source = test_source = mode
            train_target = None
            target = load_target_scores(config, record, mode)
        for subject in pending:
            key = (subject, test_source)
            if key not in beta_cache:
                beta_cache[key] = load_beta(config, subject, test_source)
            if train_target is None:
                arrays = score_arrays(scorer, beta_cache[key], target)
            else:
                train_key = (subject, train_source)
                if train_key not in beta_cache:
                    beta_cache[train_key] = load_beta(config, subject, train_source)
                arrays = score_transfer_arrays(
                    config,
                    beta_cache[train_key],
                    train_target,
                    beta_cache[key],
                    target,
                )
            save_npz(
                brain_path(config, mode, model_set, record, subject),
                **arrays,
                pc=np.arange(1, record.rank + 1),
            )
        print(
            f"brain {task_index + 1}/{len(tasks)} {mode} {model_set} "
            f"{record.model} {record.node}",
            flush=True,
        )


def run_universality(
    config: RunConfig,
    inventory: Inventory,
    *,
    shard_index: int,
    shard_count: int,
    overwrite: bool,
) -> None:
    scorer = make_scorer(config, permutations=True)
    tasks = target_tasks(config, inventory)
    for task_index, task in shard(tasks, shard_index, shard_count):
        mode, model_set, record = task
        predictors = [
            model for model in inventory.models(model_set) if model != record.model
        ]
        observed_paths = [
            universality_path(config, mode, model_set, record, model)
            for model in predictors
        ]
        if mode in config.transfer_modes:
            pending = [
                (model, path)
                for model, path in zip(predictors, observed_paths, strict=True)
                if overwrite or not path.is_file()
            ]
            if not pending:
                continue
            train_source, train_ids, test_source, test_ids = transfer_definition(
                config, mode
            )
            train_target = load_target_scores(
                config, record, train_source, train_ids
            )
            test_target = load_target_scores(config, record, test_source, test_ids)
            for predictor_model, observed_path in pending:
                train_predictor = load_model_predictor(
                    config, inventory, predictor_model, train_source
                )
                test_predictor = (
                    train_predictor
                    if test_source == train_source
                    else load_model_predictor(
                        config, inventory, predictor_model, test_source
                    )
                )
                arrays = score_transfer_arrays(
                    config,
                    train_predictor,
                    train_target,
                    test_predictor,
                    test_target,
                )
                save_npz(
                    observed_path,
                    correlation=arrays["correlation"],
                    pc=np.arange(1, record.rank + 1),
                )
                del train_predictor, test_predictor, arrays
                gc.collect()
            print(
                f"universality {task_index + 1}/{len(tasks)} {mode} {model_set} "
                f"{record.model} {record.node}",
                flush=True,
            )
            continue
        aggregate_path = universality_null_path(config, mode, model_set, record)
        if (
            not overwrite
            and aggregate_path.is_file()
            and all(path.is_file() for path in observed_paths)
        ):
            continue
        target = load_target_scores(config, record, mode)
        null_scores = []
        for predictor_model, observed_path in zip(
            predictors, observed_paths, strict=True
        ):
            predictor = load_model_predictor(config, inventory, predictor_model, mode)
            arrays = score_arrays(scorer, predictor, target)
            save_npz(
                observed_path,
                correlation=arrays["correlation"],
                pc=np.arange(1, record.rank + 1),
            )
            null_scores.append(arrays["null_correlation"].mean(axis=1))
            del predictor, arrays
            gc.collect()
        median_null = np.median(np.stack(null_scores), axis=0).astype(np.float32)
        save_npy(aggregate_path, median_null)
        print(
            f"universality {task_index + 1}/{len(tasks)} {mode} {model_set} "
            f"{record.model} {record.node}",
            flush=True,
        )


def write_run_manifest(config: RunConfig, inventory: Inventory) -> None:
    values = asdict(config)
    values = {
        key: (
            str(value)
            if isinstance(value, Path)
            else list(value) if isinstance(value, tuple) else value
        )
        for key, value in values.items()
    }
    manifest = {
        "configuration": values,
        "original_codebase_commit": "4da32500b9def393f2bea1ae15d9323fc15555dc",
        "pca_archive_sha256": "8763D2CFC801821993F669F1E95792A3A1B6B90A37A6F753DF16F1E21C23E805",
        "pca_fit_split": "unique",
        "pca_fit_presentations": 23560,
        "layers": len(inventory.layers),
        "physical_models": len({record.model for record in inventory.layers}),
        "scorer": "orig_codebase/src/lib/scorers/_regression.py::RegressionScorer",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "command_line": sys.argv,
    }
    write_json(config.analysis_dir / "run_manifest.json", manifest)
