from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from .config import RunConfig
from .data import Inventory
from .rsa import collect_rsa
from .summary import null_path


SET_LABELS = {
    "resnet18_classification_imagenet1k_varied_seeds": "Different initializations",
    "classification_imagenet1k_varied_architectures": "Different architectures",
    "resnet50_imagenet1k_varied_tasks": "Different tasks",
    "untrained_resnet18_varied_seeds": "Untrained",
}


def _correlation_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean()
    denominator = np.sqrt(np.sum(x * x, axis=1) * np.sum(y * y))
    return np.divide(
        x @ y, denominator, out=np.full(len(x), np.nan), where=denominator > 0
    )


def _residualize_rows(values: np.ndarray, control: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(control)), control])
    projection = np.linalg.inv(design.T @ design) @ design.T
    return values - (values @ design @ projection)


def partial_spearman(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    ranked = np.vstack([rankdata(x), rankdata(y)])
    residuals = _residualize_rows(ranked, rankdata(control))
    return float(_correlation_rows(residuals[:1], residuals[1])[0])


def _null_blocks(
    config: RunConfig, inventory: Inventory, group: pd.DataFrame
) -> list[tuple[np.ndarray, np.ndarray]]:
    blocks = []
    for (model_set, mode, model, node), rows in group.groupby(
        ["model_set", "mode", "model", "node"], sort=False
    ):
        record = inventory.layer(str(model), str(node))
        values = np.load(
            null_path(config, str(mode), str(model_set), record), mmap_mode="r"
        )
        blocks.append((values, rows["pc"].to_numpy(dtype=int) - 1))
    return blocks


def permutation_statistics(
    config: RunConfig,
    inventory: Inventory,
    group: pd.DataFrame,
    *,
    batch_size: int = 50,
) -> dict[str, float]:
    group = group.reset_index(drop=True)
    observed = float(
        spearmanr(group["universality"], group["brain_similarity"]).statistic
    )
    partial = partial_spearman(
        group["universality"].to_numpy(),
        group["brain_similarity"].to_numpy(),
        group["pc"].to_numpy(),
    )
    y_rank = rankdata(group["brain_similarity"].to_numpy())
    control_rank = rankdata(group["pc"].to_numpy())
    y_partial = _residualize_rows(y_rank[None, :], control_rank)[0]
    blocks = _null_blocks(config, inventory, group)
    null_observed = []
    null_partial = []
    for start in range(0, config.n_permutations, batch_size):
        stop = min(start + batch_size, config.n_permutations)
        values = np.concatenate(
            [array[start:stop, indices] for array, indices in blocks], axis=1
        )
        ranked = rankdata(values, axis=1)
        null_observed.append(_correlation_rows(ranked, y_rank))
        null_partial.append(
            _correlation_rows(_residualize_rows(ranked, control_rank), y_partial)
        )
    null_observed = np.concatenate(null_observed)
    null_partial = np.concatenate(null_partial)
    return {
        "n_dimensions": len(group),
        "spearman_rho": observed,
        "spearman_p_permutation": float(
            (1 + np.sum(null_observed >= observed)) / (len(null_observed) + 1)
        ),
        "partial_spearman_rho_control_pc": partial,
        "partial_spearman_p_permutation": float(
            (1 + np.sum(null_partial >= partial)) / (len(null_partial) + 1)
        ),
    }


def add_depth_bins(frame: pd.DataFrame, n_bins: int = 20) -> pd.DataFrame:
    parts = []
    for model_set, group in frame.groupby("model_set", sort=False):
        group = group.copy()
        edges = np.linspace(
            group["node_depth"].min(), group["node_depth"].max(), n_bins + 1
        )
        group["depth_bin"] = pd.cut(
            group["node_depth"],
            bins=edges,
            labels=False,
            include_lowest=True,
            duplicates="drop",
        )
        group["depth_bin"] = group["depth_bin"].astype(int)
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def write_statistics(
    config: RunConfig, inventory: Inventory, frame: pd.DataFrame
) -> pd.DataFrame:
    figure2_rows = []
    for model_set, group in frame.groupby("model_set", sort=False):
        figure2_rows.append(
            {"model_set": model_set, **permutation_statistics(config, inventory, group)}
        )
    figure2 = pd.DataFrame(figure2_rows)
    figure2.to_csv(config.tables_dir / "figure2_statistics.csv", index=False)

    depth_frame = add_depth_bins(frame)
    depth_frame.to_csv(config.tables_dir / "figure3_dimensions.csv", index=False)
    figure3_rows = []
    for (model_set, depth_bin), group in depth_frame.groupby(
        ["model_set", "depth_bin"], sort=False
    ):
        if len(group) < 4:
            continue
        figure3_rows.append(
            {
                "model_set": model_set,
                "depth_bin": depth_bin,
                "mean_node_depth": group["node_depth"].mean(),
                **permutation_statistics(config, inventory, group),
            }
        )
    pd.DataFrame(figure3_rows).to_csv(
        config.tables_dir / "figure3_statistics.csv", index=False
    )
    return depth_frame


def _quantile_line(group: pd.DataFrame, bins: int = 100) -> pd.DataFrame:
    group = group.sort_values("universality").copy()
    group["bin"] = np.arange(len(group)) // max(1, math.ceil(len(group) / bins))
    return group.groupby("bin", as_index=False)[
        ["universality", "brain_similarity"]
    ].mean()


def _save(fig: plt.Figure, config: RunConfig, stem: str) -> None:
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    for extension in config.figure_formats:
        fig.savefig(
            config.figures_dir / f"{stem}.{extension}", dpi=300, bbox_inches="tight"
        )
    plt.close(fig)


def plot_figure2(config: RunConfig, frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    for axis in list(axes.flat)[len(config.model_sets):]:
        axis.set_visible(False)
    for axis, model_set in zip(list(axes.flat)[:len(config.model_sets)], config.model_sets, strict=True):
        group = frame[frame["model_set"] == model_set]
        axis.hexbin(
            group["universality"],
            group["brain_similarity"],
            gridsize=55,
            bins="log",
            mincnt=1,
            cmap="Greys",
        )
        line = _quantile_line(group)
        axis.plot(
            line["universality"], line["brain_similarity"], color="#d95f02", linewidth=2
        )
        axis.set_title(SET_LABELS[model_set])
        axis.set_xlabel("Universality")
        axis.set_ylabel("Brain similarity")
    fig.tight_layout()
    _save(fig, config, "figure2_universality_brain_similarity")


def plot_figure3(config: RunConfig, frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    cmap = plt.get_cmap("viridis")
    for axis in list(axes.flat)[len(config.model_sets):]:
        axis.set_visible(False)
    for axis, model_set in zip(list(axes.flat)[:len(config.model_sets)], config.model_sets, strict=True):
        subset = frame[frame["model_set"] == model_set]
        for depth_bin, group in subset.groupby("depth_bin", sort=True):
            line = _quantile_line(group, bins=20)
            color = cmap(float(depth_bin) / max(subset["depth_bin"].max(), 1))
            axis.plot(
                line["universality"], line["brain_similarity"], color=color, alpha=0.8
            )
        axis.set_title(SET_LABELS[model_set])
        axis.set_xlabel("Universality")
        axis.set_ylabel("Brain similarity")
    legend_axis = axes[0, 0].inset_axes([0.08, 0.81, 0.30, 0.035])
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=cmap),
        cax=legend_axis, orientation="horizontal", ticks=[0, 1], alpha=0.8,
    )
    legend_axis.set_title("Network depth", fontsize=9, pad=5)
    colorbar.set_ticklabels(["Early", "Late"])
    legend_axis.tick_params(length=0, labelsize=8)
    fig.tight_layout()
    _save(fig, config, "figure3_depth_resolved_relationship")


def plot_figure5(config: RunConfig, rsa: pd.DataFrame) -> None:
    per_model = rsa.groupby(["model_set", "model", "condition"], as_index=False)[
        "rsa"
    ].mean()
    summary = (
        per_model.groupby(["model_set", "condition"])["rsa"]
        .agg(["mean", "sem"])
        .reset_index()
    )
    order = ["all", "within_basis_ui.n_top=10", "within_basis_ui.n_top=5"]
    x = np.arange(len(config.model_sets))
    width = 0.24
    fig, axis = plt.subplots(figsize=(9, 4.5))
    for index, condition in enumerate(order):
        group = (
            summary[summary["condition"] == condition]
            .set_index("model_set")
            .reindex(config.model_sets)
        )
        axis.bar(
            x + (index - 1) * width,
            group["mean"],
            width,
            yerr=group["sem"],
            label=condition,
        )
    axis.set_xticks(
        x, [SET_LABELS[item] for item in config.model_sets], rotation=15, ha="right"
    )
    axis.set_ylabel("Held-out RSA")
    axis.legend(frameon=False)
    fig.tight_layout()
    _save(fig, config, "figure5_rsa_universal_subspaces")


def plot_generalization(config: RunConfig) -> None:
    correlations = pd.read_csv(
        config.tables_dir / "generalization_correlations_by_split.csv"
    )
    correlations["test"] = correlations["method"]
    correlations.loc[
        correlations["method"] == "clip_cluster", "test"
    ] = "CLIP clusters"
    correlations.loc[correlations["method"] == "tau", "test"] = "Tau split"
    correlations.loc[correlations["method"] == "ood", "test"] = "OOD"
    collapsed = correlations.groupby(
        ["test", "model_set", "subject"], as_index=False
    )[["spearman_rho", "partial_spearman_rho_control_pc"]].mean()
    order = ["Tau split", "CLIP clusters", "OOD"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for axis, metric in zip(
        axes,
        ["spearman_rho", "partial_spearman_rho_control_pc"],
        strict=True,
    ):
        for model_set in config.model_sets:
            group = collapsed[collapsed["model_set"] == model_set]
            means = group.groupby("test")[metric].mean().reindex(order)
            axis.plot(order, means, marker="o", label=SET_LABELS[model_set])
        axis.axhline(0, color="0.75", linewidth=1)
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.tick_params(axis="x", rotation=15)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, config, "generalization_tests")

    if config.reliability_summary_path is None:
        return
    reliability = pd.read_csv(config.reliability_summary_path)
    reliability = reliability[
        reliability["evaluation_set"].isin(["shared", "ood"])
    ]
    metric = "mean_over_splits_of_median_voxel_r"
    fig, axis = plt.subplots(figsize=(5.5, 4.5))
    for subject, group in reliability.groupby("subject", sort=False):
        values = group.set_index("evaluation_set")[metric].reindex(["shared", "ood"])
        axis.plot(["Shared", "OOD"], values, marker="o", alpha=0.8, label=subject)
    axis.set_ylabel("Median voxel split-half reliability")
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, config, "generalization_reliability")


def run_figures(config: RunConfig, inventory: Inventory) -> None:
    frame = pd.read_csv(config.tables_dir / "figure2_dimensions.csv")
    depth_frame = write_statistics(config, inventory, frame)
    rsa = collect_rsa(config, inventory)
    plot_figure2(config, frame)
    plot_figure3(config, depth_frame)
    plot_figure5(config, rsa)
    plot_generalization(config)
