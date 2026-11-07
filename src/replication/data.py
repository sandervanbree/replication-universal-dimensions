from __future__ import annotations

from dataclasses import dataclass
import csv
from functools import cache
import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
import xarray as xr

from src.laion.pca_handoff import PCAHandoff
from .config import REPO_ROOT, RunConfig


EXPECTED_MODEL_COUNTS = {
    "resnet18_classification_imagenet1k_varied_seeds": 20,
    "classification_imagenet1k_varied_architectures": 19,
    "resnet50_imagenet1k_varied_tasks": 9,
    "untrained_resnet18_varied_seeds": 20,
}


@dataclass(frozen=True)
class LayerRecord:
    model: str
    node: str
    hook: str
    model_sets: tuple[str, ...]
    rank: int
    input_features: int

    @property
    def architecture(self) -> str:
        return self.model.split(".weights=", maxsplit=1)[0]


@dataclass(frozen=True)
class EvaluationSplit:
    name: str
    train_source: str
    test_source: str
    train_ids: tuple[str, ...]
    test_ids: tuple[str, ...]


class Inventory:
    def __init__(self, pca_root: Path):
        self.pca_root = Path(pca_root)
        manifest = self.pca_root / "layer_manifest.csv"
        with manifest.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        self.layers = tuple(
            LayerRecord(
                model=row["model"],
                node=row["node"],
                hook=row["hook"],
                model_sets=tuple(row["model_sets"].split(";")),
                rank=int(row["rank"]),
                input_features=int(row["input_features"]),
            )
            for row in rows
            if row["status"] == "written"
        )
        if len(self.layers) != 1018:
            raise ValueError(f"Expected 1,018 PCA layers, found {len(self.layers)}")
        self._layer_map = {(item.model, item.node): item for item in self.layers}

    def records(self, model_set: str | None = None) -> list[LayerRecord]:
        if model_set is None:
            return list(self.layers)
        return [item for item in self.layers if model_set in item.model_sets]

    def models(self, model_set: str) -> list[str]:
        return list(dict.fromkeys(item.model for item in self.records(model_set)))

    def nodes(self, model: str) -> list[str]:
        return [item.node for item in self.layers if item.model == model]

    def layer(self, model: str, node: str) -> LayerRecord:
        return self._layer_map[(model, node)]


def source_mode(mode: str) -> str:
    if mode in {"shared_train", "shared_4rep"}:
        return "shared"
    if mode not in {"shared", "ood"}:
        raise ValueError(f"Unknown evaluation mode: {mode}")
    return mode


def generalization_column(mode: str) -> str | None:
    prefix = "generalization_"
    return mode.removeprefix(prefix) if mode.startswith(prefix) else None


@cache
def load_evaluation_split(config: RunConfig, mode: str) -> EvaluationSplit | None:
    column = generalization_column(mode)
    if column is None:
        return None
    if config.generalization_splits_path is None:
        raise ValueError(f"{mode} requires generalization_splits_path")
    frame = pd.read_csv(config.generalization_splits_path, dtype=str).fillna("")
    if column not in frame.columns:
        raise ValueError(f"Split column {column!r} is missing from {config.generalization_splits_path}")
    train_source = "shared"
    test_source = "ood" if column == "ood" else "shared"
    train_ids = tuple(
        frame.loc[(frame["mode"] == train_source) & (frame[column] == "train"), "stimulus_id"]
    )
    test_ids = tuple(
        frame.loc[(frame["mode"] == test_source) & (frame[column] == "test"), "stimulus_id"]
    )
    if not train_ids or not test_ids:
        raise ValueError(f"Split {column!r} has an empty train or test set")
    return EvaluationSplit(column, train_source, test_source, train_ids, test_ids)


def select_mode(
    ids: np.ndarray, values: np.ndarray, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    ids = ids.astype(str)
    if mode == "shared_train":
        keep = np.char.startswith(ids, "shared_12rep_")
        return ids[keep], values[keep]
    if mode == "shared_4rep":
        keep = np.char.startswith(ids, "shared_4rep_") & ~np.char.startswith(
            ids, "shared_4rep_OOD_"
        )
        return ids[keep], values[keep]
    if mode in {"shared", "ood"}:
        return ids, values
    raise ValueError(f"Unknown evaluation mode: {mode}")


def decode_ids(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            (
                value.decode("utf-8")
                if isinstance(value, (bytes, np.bytes_))
                else str(value)
            )
            for value in np.asarray(values).reshape(-1)
        ],
        dtype=str,
    )


def data_variable(file: h5py.File, preferred: str | None = None) -> str:
    if preferred and preferred in file:
        return preferred
    excluded = {"presentation", "stimulus_id", "channel", "neuroid", "subject", "roi"}
    candidates = [
        key
        for key, value in file.items()
        if key not in excluded and isinstance(value, h5py.Dataset) and value.ndim == 2
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Could not identify data variable in {file.filename}: {candidates}"
        )
    return candidates[0]


def read_array(
    path: Path, preferred: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as file:
        ids = decode_ids(file["stimulus_id"][:])
        values = np.asarray(file[data_variable(file, preferred)][:], dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != len(ids):
        raise ValueError(
            f"Unexpected data shape in {path}: {values.shape}, ids={len(ids)}"
        )
    if len(ids) != len(set(ids.tolist())) or not np.isfinite(values).all():
        raise ValueError(f"Duplicate IDs or non-finite values in {path}")
    return ids, values


def align_rows(
    ids: np.ndarray, values: np.ndarray, ordered_ids: Iterable[str]
) -> np.ndarray:
    ordered_ids = np.asarray(list(ordered_ids), dtype=str)
    positions = {value: index for index, value in enumerate(ids.astype(str))}
    missing = [value for value in ordered_ids if value not in positions]
    if missing:
        raise ValueError(f"Missing {len(missing)} stimulus IDs; first={missing[0]}")
    return values[np.asarray([positions[value] for value in ordered_ids])]


def feature_path(config: RunConfig, record: LayerRecord, mode: str) -> Path:
    directory = (
        config.feature_root
        / record.model
        / f"datapipe=laion_fmri.{source_mode(mode)}"
        / f"node={record.node}"
    )
    files = sorted(directory.glob("hook=*.nc"))
    if len(files) != 1:
        raise FileNotFoundError(
            f"Expected one feature file in {directory}, found {len(files)}"
        )
    if files[0].stem.removeprefix("hook=") != record.hook:
        raise ValueError(
            f"Hook mismatch for {record.model}/{record.node}: {files[0].name}"
        )
    return files[0]


def load_layer_features(
    config: RunConfig,
    record: LayerRecord,
    mode: str,
    stimulus_ids: Iterable[str] | None = None,
) -> xr.DataArray:
    ids, values = read_array(feature_path(config, record, mode))
    if stimulus_ids is None:
        ids, values = select_mode(ids, values, mode)
    else:
        ordered_ids = np.asarray(list(stimulus_ids), dtype=str)
        values = align_rows(ids, values, ordered_ids)
        ids = ordered_ids
    return xr.DataArray(
        values,
        dims=("presentation", "neuroid"),
        coords={
            "stimulus_id": ("presentation", ids),
            "neuroid": np.arange(values.shape[1]),
        },
        name=f"features.mode={mode}.model={record.model}.node={record.node}",
    )


def load_model_predictor(
    config: RunConfig,
    inventory: Inventory,
    model: str,
    mode: str,
    stimulus_ids: Iterable[str] | None = None,
) -> xr.DataArray:
    arrays = [
        load_layer_features(config, inventory.layer(model, node), mode, stimulus_ids)
        for node in inventory.nodes(model)
    ]
    ordered_ids = arrays[0].stimulus_id.values.astype(str)
    values = [
        align_rows(array.stimulus_id.values, array.values, ordered_ids)
        for array in arrays
    ]
    combined = np.concatenate(values, axis=1)
    return xr.DataArray(
        combined,
        dims=("presentation", "neuroid"),
        coords={
            "stimulus_id": ("presentation", ordered_ids),
            "neuroid": np.arange(combined.shape[1]),
        },
        name=f"predictor.mode={mode}.model={model}.nodes=all",
    )


def load_target_scores(
    config: RunConfig,
    record: LayerRecord,
    mode: str,
    stimulus_ids: Iterable[str] | None = None,
) -> xr.DataArray:
    state = PCAHandoff.load(config.pca_root, model=record.model, node=record.node)
    scores = state.scores(source_mode(mode))
    if stimulus_ids is not None:
        ordered_ids = np.asarray(list(stimulus_ids), dtype=str)
        values = align_rows(scores.stimulus_id.values, scores.values, ordered_ids)
        scores = xr.DataArray(
            values,
            dims=("presentation", "neuroid"),
            coords={
                "stimulus_id": ("presentation", ordered_ids),
                "neuroid": scores.neuroid.values,
                "pc": ("neuroid", scores.pc.values),
            },
            name=scores.name,
            attrs=scores.attrs,
        )
    elif mode in {"shared_train", "shared_4rep"}:
        ids, values = select_mode(
            scores.stimulus_id.values.astype(str), scores.values, mode
        )
        scores = xr.DataArray(
            values,
            dims=("presentation", "neuroid"),
            coords={
                "stimulus_id": ("presentation", ids),
                "neuroid": scores.neuroid.values,
                "pc": ("neuroid", scores.pc.values),
            },
            name=scores.name,
            attrs=scores.attrs,
        )
    return scores


def beta_path(config: RunConfig, subject: str, mode: str) -> Path:
    return config.beta_root / f"mode={source_mode(mode)}" / f"subject={subject}.nc"


def load_beta(
    config: RunConfig,
    subject: str,
    mode: str,
    stimulus_ids: Iterable[str] | None = None,
) -> xr.DataArray:
    path = beta_path(config, subject, mode)
    ids, values = read_array(path, preferred="beta")
    if stimulus_ids is None:
        ids, values = select_mode(ids, values, mode)
    else:
        ordered_ids = np.asarray(list(stimulus_ids), dtype=str)
        values = align_rows(ids, values, ordered_ids)
        ids = ordered_ids
    return xr.DataArray(
        values,
        dims=("presentation", "neuroid"),
        coords={
            "stimulus_id": ("presentation", ids),
            "neuroid": np.arange(values.shape[1]),
        },
        name=f"beta.roi={config.roi}.mode={mode}.subject={subject}",
    )


def load_node_depths() -> dict[tuple[str, str], float]:
    path = REPO_ROOT / "orig_codebase" / "src" / "lib" / "models" / "model_layers.csv"
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    grouped: dict[str, list[str]] = {}
    for row in rows:
        grouped.setdefault(row["Architecture"], []).append(row["Layer"])
    return {
        (architecture, node): index / max(len(nodes) - 1, 1)
        for architecture, nodes in grouped.items()
        for index, node in enumerate(nodes)
    }


def validate_inputs(config: RunConfig, inventory: Inventory) -> dict[str, object]:
    errors: list[str] = []
    if config.main_mode not in config.modes:
        errors.append(f"main_mode is not in modes: {config.main_mode}")
    if not set(config.transfer_modes).issubset(config.modes):
        errors.append("transfer_modes must be a subset of modes")
    if config.main_mode in config.transfer_modes:
        errors.append("main_mode cannot also be a transfer mode")
    for path in (config.pca_root, config.feature_root, config.beta_root):
        if not path.is_dir():
            errors.append(f"missing directory: {path}")
    if config.generalization_splits_path is not None and not config.generalization_splits_path.is_file():
        errors.append(f"missing split file: {config.generalization_splits_path}")
    if config.reliability_summary_path is not None and not config.reliability_summary_path.is_file():
        errors.append(f"missing reliability summary: {config.reliability_summary_path}")

    splits: dict[str, EvaluationSplit] = {}
    source_modes: set[str] = set()
    for mode in config.modes:
        try:
            split = load_evaluation_split(config, mode)
            if split is None:
                source_modes.add(source_mode(mode))
            else:
                splits[mode] = split
                source_modes.update((split.train_source, split.test_source))
        except (FileNotFoundError, KeyError, ValueError) as error:
            errors.append(str(error))
    if any(mode in config.transfer_modes and mode not in splits for mode in config.modes):
        source_modes.add(source_mode(config.transfer_train_mode))
    set_counts = {name: len(inventory.models(name)) for name in config.model_sets}
    for name, expected in EXPECTED_MODEL_COUNTS.items():
        if name in config.model_sets and set_counts.get(name) != expected:
            errors.append(
                f"{name}: expected {expected} models, found {set_counts.get(name)}"
            )
    missing_pca = []
    missing_features = []
    for record in inventory.layers:
        if not (
            config.pca_root
            / "layers"
            / record.model
            / f"node={record.node}"
            / "pca.npz"
        ).is_file():
            missing_pca.append(f"{record.model}/{record.node}")
        for mode in source_modes:
            directory = (
                config.feature_root
                / record.model
                / f"datapipe=laion_fmri.{mode}"
                / f"node={record.node}"
            )
            if len(list(directory.glob("hook=*.nc"))) != 1:
                missing_features.append(f"{mode}/{record.model}/{record.node}")
    missing_betas = [
        str(beta_path(config, subject, mode))
        for subject in config.subjects
        for mode in source_modes
        if not beta_path(config, subject, mode).is_file()
    ]
    if missing_pca:
        errors.append(f"missing PCA artifacts: {len(missing_pca)}")
    if missing_features:
        errors.append(f"missing feature files: {len(missing_features)}")
    if missing_betas:
        errors.append(f"missing beta files: {len(missing_betas)}")
    first = inventory.layers[0]
    state = PCAHandoff.load(config.pca_root, model=first.model, node=first.node)
    shared_ids = state.stimulus_ids("shared")
    ood_ids = state.stimulus_ids("ood")
    stimulus_counts = {
        "shared": len(shared_ids),
        "ood": len(ood_ids),
    }
    expected_counts = {"shared": 1121, "ood": 371}
    if stimulus_counts != expected_counts:
        errors.append(
            f"unexpected stimulus counts: expected {expected_counts}, found {stimulus_counts}"
        )
    split_counts: dict[str, dict[str, int]] = {}
    shared_set = set(shared_ids.astype(str))
    ood_set = set(ood_ids.astype(str))
    for mode, split in splits.items():
        train_set = set(split.train_ids)
        test_set = set(split.test_ids)
        split_counts[mode] = {"train": len(split.train_ids), "test": len(split.test_ids)}
        expected_train = shared_set
        expected_test = ood_set if split.test_source == "ood" else shared_set
        if not train_set.issubset(expected_train) or not test_set.issubset(expected_test):
            errors.append(f"{mode} contains stimulus IDs outside its declared source")
        if train_set & test_set:
            errors.append(f"{mode} train and test sets overlap")
        if split.name == "ood":
            if train_set != shared_set or test_set != ood_set:
                errors.append("generalization_ood must use all 1,121 shared images for training and all 371 OOD images for testing")
        elif train_set | test_set != shared_set:
            errors.append(f"{mode} does not partition all shared stimuli")

    expected_split_counts = {
        "generalization_tau": {"train": 897, "test": 224},
        "generalization_cluster_k5_0": {"train": 847, "test": 274},
        "generalization_cluster_k5_1": {"train": 873, "test": 248},
        "generalization_cluster_k5_2": {"train": 875, "test": 246},
        "generalization_cluster_k5_3": {"train": 896, "test": 225},
        "generalization_cluster_k5_4": {"train": 993, "test": 128},
        "generalization_ood": {"train": 1121, "test": 371},
    }
    for mode, expected in expected_split_counts.items():
        if mode in config.modes and split_counts.get(mode) != expected:
            errors.append(f"{mode}: expected {expected}, found {split_counts.get(mode)}")

    cluster_modes = [f"generalization_cluster_k5_{index}" for index in range(5)]
    if set(cluster_modes).issubset(splits):
        cluster_tests = [set(splits[mode].test_ids) for mode in cluster_modes]
        if set().union(*cluster_tests) != shared_set or sum(map(len, cluster_tests)) != len(shared_set):
            errors.append("the five CLIP test folds must partition the 1,121 shared stimuli exactly once")

    report = {
        "status": "valid" if not errors else "invalid",
        "layers": len(inventory.layers),
        "physical_models": len({item.model for item in inventory.layers}),
        "model_counts": set_counts,
        "subjects": list(config.subjects),
        "modes": list(config.modes),
        "transfer_train_mode": config.transfer_train_mode,
        "transfer_modes": list(config.transfer_modes),
        "stimulus_counts": stimulus_counts,
        "generalization_split_counts": split_counts,
        "errors": errors,
        "examples": {
            "missing_pca": missing_pca[:10],
            "missing_features": missing_features[:10],
            "missing_betas": missing_betas[:10],
        },
    }
    return report


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, **arrays)
    temporary.replace(path)


def save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        np.save(file, values, allow_pickle=False)
    temporary.replace(path)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
