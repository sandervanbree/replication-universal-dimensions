from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from src.laion.pca_handoff import PCAHandoff
from .config import RunConfig
from .data import Inventory, LayerRecord
from .scoring import brain_path, universality_null_path, universality_path


def summary_dir(config: RunConfig, mode: str, model_set: str) -> Path:
    return config.analysis_dir / "summaries" / f"mode={mode}" / f"model_set={model_set}"


def null_path(
    config: RunConfig, mode: str, model_set: str, record: LayerRecord
) -> Path:
    return universality_null_path(config, mode, model_set, record)


def _load_score(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def summarize_layer(
    config: RunConfig,
    inventory: Inventory,
    mode: str,
    model_set: str,
    record: LayerRecord,
    depths: dict[tuple[str, str], float],
) -> list[dict[str, object]]:
    subject_scores = []
    expected_splits = config.n_folds if mode == config.main_mode else 1
    for subject in config.subjects:
        values = _load_score(brain_path(config, mode, model_set, record, subject))[
            "correlation"
        ]
        if values.shape != (expected_splits, record.rank):
            raise ValueError(
                f"Bad brain score shape for {record.model}/{record.node}/{subject}: {values.shape}"
            )
        subject_scores.append(values.mean(axis=0))
    subject_scores = np.stack(subject_scores)

    observed = []
    for predictor in inventory.models(model_set):
        if predictor == record.model:
            continue
        values = _load_score(
            universality_path(config, mode, model_set, record, predictor)
        )
        if values["correlation"].shape != (expected_splits, record.rank):
            raise ValueError(
                f"Bad universality score shape for {record.model}/{record.node}/"
                f"{predictor}: {values['correlation'].shape}"
            )
        observed.append(values["correlation"].mean(axis=0))
    universality = np.median(np.stack(observed), axis=0)
    if mode == config.main_mode:
        null = np.load(null_path(config, mode, model_set, record), mmap_mode="r")
        if null.shape != (config.n_permutations, record.rank):
            raise ValueError(
                f"Bad universality null shape for {record.model}/{record.node}: "
                f"{null.shape}"
            )

    state = PCAHandoff.load(config.pca_root, model=record.model, node=record.node)
    variance_fraction = state.eigenvalues / state.eigenvalues.sum()
    depth_key = (record.architecture, record.node)
    if depth_key not in depths:
        raise KeyError(f"Missing original operation depth for {depth_key}")
    rows = []
    for pc in range(record.rank):
        row = {
            "model_set": model_set,
            "mode": mode,
            "model": record.model,
            "architecture": record.architecture,
            "node": record.node,
            "node_depth": depths[depth_key],
            "pc": pc + 1,
            "rank": record.rank,
            "eigenvalue": float(state.eigenvalues[pc]),
            "variance_fraction": float(variance_fraction[pc]),
            "brain_similarity": float(subject_scores[:, pc].mean()),
            "universality": float(universality[pc]),
        }
        row.update(
            {
                f"brain_similarity_{subject}": float(subject_scores[index, pc])
                for index, subject in enumerate(config.subjects)
            }
        )
        rows.append(row)
    return rows


def load_depths(config: RunConfig) -> dict[tuple[str, str], float]:
    path = config.tables_dir / "node_depths.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Run the depths stage first: {path}")
    frame = pd.read_csv(path)
    return {
        (str(row.architecture), str(row.node)): float(row.node_depth)
        for row in frame.itertuples(index=False)
    }


def run_summaries(config: RunConfig, inventory: Inventory) -> None:
    depths = load_depths(config)
    all_frames = []
    for mode in config.modes:
        for model_set in config.model_sets:
            rows = []
            for record in inventory.records(model_set):
                layer_rows = summarize_layer(
                    config, inventory, mode, model_set, record, depths
                )
                rows.extend(layer_rows)
            frame = pd.DataFrame(rows)
            path = summary_dir(config, mode, model_set) / "dimensions.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False)
            all_frames.append(frame)
    dimensions = pd.concat(all_frames, ignore_index=True)
    config.tables_dir.mkdir(parents=True, exist_ok=True)
    dimensions.to_csv(config.tables_dir / "dimensions_all_modes.csv", index=False)
    dimensions[dimensions["mode"] == config.main_mode].to_csv(
        config.tables_dir / "figure2_dimensions.csv", index=False
    )
    write_generalization_results(config, dimensions)


def _partial_spearman(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    design = np.column_stack([np.ones(len(control)), rankdata(control)])
    x_residual = rankdata(x) - design @ np.linalg.lstsq(design, rankdata(x), rcond=None)[0]
    y_residual = rankdata(y) - design @ np.linalg.lstsq(design, rankdata(y), rcond=None)[0]
    return float(np.corrcoef(x_residual, y_residual)[0, 1])


def _split_metadata(mode: str) -> tuple[str, int | None]:
    split = mode.removeprefix("generalization_")
    if split.startswith("cluster_k5_"):
        return "clip_cluster", int(split.rsplit("_", maxsplit=1)[1])
    return split, None


def write_generalization_results(
    config: RunConfig, dimensions: pd.DataFrame
) -> None:
    generalization = dimensions[
        dimensions["mode"].str.startswith("generalization_")
    ].copy()
    if generalization.empty:
        return
    generalization.to_csv(
        config.tables_dir / "generalization_dimensions.csv", index=False
    )

    rows = []
    for (mode, model_set), group in generalization.groupby(
        ["mode", "model_set"], sort=False
    ):
        method, fold = _split_metadata(str(mode))
        for subject in config.subjects:
            brain = group[f"brain_similarity_{subject}"].to_numpy(dtype=float)
            universality = group["universality"].to_numpy(dtype=float)
            pc = group["pc"].to_numpy(dtype=float)
            rows.append(
                {
                    "mode": mode,
                    "method": method,
                    "fold": fold,
                    "model_set": model_set,
                    "subject": subject,
                    "n_dimensions": len(group),
                    "spearman_rho": float(spearmanr(universality, brain).statistic),
                    "partial_spearman_rho_control_pc": _partial_spearman(
                        universality, brain, pc
                    ),
                }
            )
    correlations = pd.DataFrame(rows)
    correlations.to_csv(
        config.tables_dir / "generalization_correlations_by_split.csv", index=False
    )

    cluster = correlations[correlations["method"] == "clip_cluster"].copy()
    summary_rows = []
    for (model_set, subject), group in cluster.groupby(
        ["model_set", "subject"], sort=False
    ):
        row = {
            "model_set": model_set,
            "subject": subject,
            "n_folds": len(group),
        }
        for metric in ("spearman_rho", "partial_spearman_rho_control_pc"):
            values = group[metric].to_numpy(dtype=float)
            clipped = np.clip(values, -1 + 1e-7, 1 - 1e-7)
            row[f"mean_{metric}"] = float(values.mean())
            row[f"sd_{metric}"] = float(values.std(ddof=1))
            row[f"fisher_z_mean_{metric}"] = float(np.tanh(np.arctanh(clipped).mean()))
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(
        config.tables_dir / "generalization_cluster_summary.csv", index=False
    )

    if config.reliability_summary_path is not None:
        reliability = pd.read_csv(config.reliability_summary_path)
        reliability.to_csv(
            config.tables_dir / "generalization_reliability.csv", index=False
        )
