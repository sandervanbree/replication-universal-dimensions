from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.laion.pca_handoff import PCAHandoff
from .config import RunConfig
from .data import Inventory, LayerRecord, align_rows, load_beta, load_layer_features
from .scoring import shard


CONDITIONS = ("all", "within_basis_ui.n_top=10", "within_basis_ui.n_top=5")


def _rdm(values: np.ndarray) -> np.ndarray:
    matrix = 1 - np.corrcoef(values)
    if not np.isfinite(matrix).all():
        raise ValueError("RDM contains non-finite values")
    return matrix


def _rsa(model_rdm: np.ndarray, brain_rdm: np.ndarray) -> float:
    lower = np.tril_indices(model_rdm.shape[0], k=-1)
    x = rankdata(model_rdm[lower])
    y = rankdata(brain_rdm[lower])
    return float(np.corrcoef(x, y)[0, 1])


def _split_ids(
    config: RunConfig, inventory: Inventory
) -> tuple[np.ndarray, np.ndarray]:
    first = inventory.layers[0]
    ids = PCAHandoff.load(
        config.pca_root, model=first.model, node=first.node
    ).stimulus_ids("shared")
    ids = np.sort(ids.astype(str))
    return ids[::2], ids[1::2]


def _brain_rdms(
    config: RunConfig, ids: np.ndarray, cache: dict[tuple[str, ...], np.ndarray]
) -> dict[str, np.ndarray]:
    output = {}
    split_key = tuple(ids.tolist())
    for subject in config.subjects:
        key = (subject, *split_key)
        if key not in cache:
            beta = load_beta(config, subject, "shared")
            values = align_rows(beta.stimulus_id.values, beta.values, ids)
            cache[key] = _rdm(values)
        output[subject] = cache[key]
    return output


def _node_scores(
    config: RunConfig,
    inventory: Inventory,
    model: str,
    ids: np.ndarray,
    brain_rdms: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for node in inventory.nodes(model):
        record = inventory.layer(model, node)
        features = load_layer_features(config, record, "shared")
        values = align_rows(features.stimulus_id.values, features.values, ids)
        model_rdm = _rdm(values)
        for subject, brain_rdm in brain_rdms.items():
            rows.append(
                {
                    "model": model,
                    "node": node,
                    "subject": subject,
                    "rsa": _rsa(model_rdm, brain_rdm),
                }
            )
    return pd.DataFrame(rows)


def _condition_values(
    config: RunConfig,
    record: LayerRecord,
    ids: np.ndarray,
    universality: pd.DataFrame,
    condition: str,
) -> np.ndarray:
    features = load_layer_features(config, record, "shared")
    if condition == "all":
        return align_rows(features.stimulus_id.values, features.values, ids)
    n_top = int(condition.rsplit("=", 1)[-1])
    state = PCAHandoff.load(config.pca_root, model=record.model, node=record.node)
    scores = state.scores("shared")
    values = align_rows(scores.stimulus_id.values, scores.values, ids)
    layer = universality[
        (universality["model"] == record.model) & (universality["node"] == record.node)
    ].sort_values("pc")
    if len(layer) != record.rank:
        raise ValueError(f"Universality rows missing for {record.model}/{record.node}")
    keep = np.argsort(layer["universality"].to_numpy())[-min(n_top, record.rank) :]
    reduced = np.zeros_like(values)
    reduced[:, keep] = values[:, keep]
    return state.inverse_transform(reduced)


def rsa_path(config: RunConfig, model_set: str, model: str) -> Path:
    return config.analysis_dir / "rsa" / f"model_set={model_set}" / f"model={model}.csv"


def run_rsa(
    config: RunConfig,
    inventory: Inventory,
    *,
    shard_index: int,
    shard_count: int,
    overwrite: bool,
) -> None:
    dimensions_path = config.tables_dir / "figure2_dimensions.csv"
    if not dimensions_path.is_file():
        raise FileNotFoundError(f"Run summarize first: {dimensions_path}")
    dimensions = pd.read_csv(dimensions_path)
    train_ids, test_ids = _split_ids(config, inventory)
    brain_cache: dict[tuple[str, ...], np.ndarray] = {}
    train_rdms = _brain_rdms(config, train_ids, brain_cache)
    test_rdms = _brain_rdms(config, test_ids, brain_cache)
    tasks = [
        (model_set, model)
        for model_set in config.model_sets
        for model in inventory.models(model_set)
    ]
    for task_index, task in shard(tasks, shard_index, shard_count):
        model_set, model = task
        output = rsa_path(config, model_set, model)
        if output.is_file() and not overwrite:
            continue
        train_scores = _node_scores(config, inventory, model, train_ids, train_rdms)
        best_node = (
            train_scores.groupby("node", as_index=False)["rsa"]
            .mean()
            .sort_values("rsa", ascending=False)
            .iloc[0]["node"]
        )
        record = inventory.layer(model, str(best_node))
        set_dimensions = dimensions[dimensions["model_set"] == model_set]
        rows = []
        for condition in CONDITIONS:
            model_rdm = _rdm(
                _condition_values(config, record, test_ids, set_dimensions, condition)
            )
            for subject, brain_rdm in test_rdms.items():
                rows.append(
                    {
                        "model_set": model_set,
                        "model": model,
                        "best_node": best_node,
                        "subject": subject,
                        "condition": condition,
                        "rsa": _rsa(model_rdm, brain_rdm),
                    }
                )
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output, index=False)
        train_scores.to_csv(
            output.with_name(output.stem + ".train_nodes.csv"), index=False
        )
        print(f"rsa {task_index + 1}/{len(tasks)} {model_set} {model}", flush=True)


def collect_rsa(config: RunConfig, inventory: Inventory) -> pd.DataFrame:
    paths = [
        rsa_path(config, model_set, model)
        for model_set in config.model_sets
        for model in inventory.models(model_set)
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} RSA outputs; first={missing[0]}"
        )
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    frame.to_csv(config.tables_dir / "figure5_rsa.csv", index=False)
    return frame
