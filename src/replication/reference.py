from __future__ import annotations

from dataclasses import dataclass, replace
import gc
import json
import math
from pathlib import Path
from typing import Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from src.laion.pca_handoff import PCAHandoff
from .config import REPO_ROOT, RunConfig, load_config
from .data import (
    Inventory,
    LayerRecord,
    align_rows,
    feature_path,
    load_layer_features,
    load_model_predictor,
    load_node_depths,
    load_target_scores,
    save_npy,
    save_npz,
    source_mode,
    write_json,
)
from .figures import SET_LABELS
from .rsa import _brain_rdms, _node_scores, _rdm, _rsa, _split_ids
from .scoring import (
    brain_path,
    make_scorer,
    score_arrays,
    score_transfer_arrays,
    universality_null_path,
    universality_path,
)


@dataclass(frozen=True)
class ReferenceConfig:
    source_path: Path
    base: RunConfig
    brain_output_root: Path
    output_root: Path
    target_models: dict[str, tuple[str, ...] | None]
    nodes: tuple[str, ...] | None
    modes: tuple[str, ...]
    n_pcs: int
    n_permutations: int
    device: str
    figure_formats: tuple[str, ...]

    @property
    def brain_config(self) -> RunConfig:
        return replace(self.base, output_root=self.brain_output_root)

    def scoring_config(self, device: str | None = None) -> RunConfig:
        return replace(
            self.base,
            output_root=self.output_root,
            model_sets=tuple(self.target_models),
            modes=self.modes,
            n_permutations=self.n_permutations,
            device=device or self.device,
            figure_formats=self.figure_formats,
        )


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def load_reference_config(path: Path | str) -> ReferenceConfig:
    source_path = Path(path).resolve()
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    raw_targets = raw["target_models"]
    targets = {
        str(model_set): (
            None
            if models == "all"
            else tuple(str(model) for model in models)
        )
        for model_set, models in raw_targets.items()
    }
    raw_nodes = raw.get("nodes", "all")
    nodes = None if raw_nodes == "all" else tuple(str(node) for node in raw_nodes)
    return ReferenceConfig(
        source_path=source_path,
        base=load_config(_path(raw["base_config"])),
        brain_output_root=_path(raw["brain_output_root"]),
        output_root=_path(raw["output_root"]),
        target_models=targets,
        nodes=nodes,
        modes=tuple(str(mode) for mode in raw["modes"]),
        n_pcs=int(raw["n_pcs"]),
        n_permutations=int(raw["n_permutations"]),
        device=str(raw.get("device", "cpu")),
        figure_formats=tuple(raw.get("figure_formats", ["png", "pdf"])),
    )


def selected_models(
    reference: ReferenceConfig, inventory: Inventory, model_set: str
) -> list[str]:
    available = inventory.models(model_set)
    configured = reference.target_models[model_set]
    if configured is None:
        return available
    missing = [model for model in configured if model not in available]
    if missing:
        raise ValueError(f"Target models are unavailable in {model_set}: {missing}")
    if len(configured) != len(set(configured)):
        raise ValueError(f"Target models must be unique in {model_set}")
    return list(configured)


def selected_records(
    reference: ReferenceConfig, inventory: Inventory, model_set: str
) -> list[LayerRecord]:
    models = set(selected_models(reference, inventory, model_set))
    records = [record for record in inventory.records(model_set) if record.model in models]
    if reference.nodes is not None:
        allowed = set(reference.nodes)
        records = [record for record in records if record.node in allowed]
        found = {record.node for record in records}
        missing = [node for node in reference.nodes if node not in found]
        if missing:
            raise ValueError(f"Target nodes are unavailable in {model_set}: {missing}")
    return records


def n_components(reference: ReferenceConfig, record: LayerRecord) -> int:
    return min(reference.n_pcs, record.rank)


def predictor_models(
    reference: ReferenceConfig,
    inventory: Inventory,
    model_set: str,
    target_model: str,
) -> list[str]:
    del reference
    return [model for model in inventory.models(model_set) if model != target_model]


def reference_tasks(
    reference: ReferenceConfig, inventory: Inventory
) -> list[tuple[str, str, LayerRecord]]:
    return [
        (mode, model_set, record)
        for mode in reference.modes
        for model_set in reference.target_models
        for record in selected_records(reference, inventory, model_set)
    ]


def _shard(items: list, index: int, count: int) -> Iterator[tuple[int, object]]:
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"Invalid shard {index}/{count}")
    for task_index, item in enumerate(items):
        if task_index % count == index:
            yield task_index, item


def validate_reference(
    reference: ReferenceConfig, inventory: Inventory
) -> dict[str, object]:
    errors: list[str] = []
    allowed_modes = {reference.base.main_mode, *reference.base.transfer_modes}
    if not set(reference.modes).issubset(allowed_modes):
        errors.append(f"unsupported modes: {sorted(set(reference.modes) - allowed_modes)}")
    if reference.base.main_mode not in reference.modes:
        errors.append("the main shared mode is required")
    if reference.n_pcs < 1:
        errors.append("n_pcs must be positive")
    if reference.n_permutations < 1:
        errors.append("n_permutations must be positive")

    target_entries = 0
    main_pairs = 0
    target_dimensions = 0
    required_features: set[tuple[str, str]] = set()
    brain_config = reference.brain_config
    for model_set in reference.target_models:
        try:
            records = selected_records(reference, inventory, model_set)
        except ValueError as error:
            errors.append(str(error))
            continue
        target_entries += len(records)
        target_dimensions += sum(n_components(reference, record) for record in records)
        for record in records:
            predictors = predictor_models(
                reference, inventory, model_set, record.model
            )
            main_pairs += len(predictors)
            for mode in reference.modes:
                for subject in reference.base.subjects:
                    path = brain_path(brain_config, mode, model_set, record, subject)
                    if not path.is_file():
                        errors.append(f"missing brain score: {path}")
                required_modes = (
                    (reference.base.transfer_train_mode, mode)
                    if mode in reference.base.transfer_modes
                    else (mode,)
                )
                for predictor in predictors:
                    for required_mode in required_modes:
                        required_features.add((predictor, source_mode(required_mode)))

    for model, mode in sorted(required_features):
        for node in inventory.nodes(model):
            try:
                feature_path(reference.base, inventory.layer(model, node), mode)
            except (FileNotFoundError, ValueError) as error:
                errors.append(str(error))

    first = inventory.layers[0]
    state = PCAHandoff.load(
        reference.base.pca_root, model=first.model, node=first.node
    )
    stimulus_counts = {
        "shared": len(state.stimulus_ids("shared")),
        "ood": len(state.stimulus_ids("ood")),
    }
    n_modes = len(reference.modes)
    report = {
        "status": "valid" if not errors else "invalid",
        "scope": "complete reduced local reference",
        "formal_result": False,
        "pca_fit_split": "unique",
        "pca_fit_presentations": 23560,
        "stimulus_counts": stimulus_counts,
        "modes": list(reference.modes),
        "target_models": {
            model_set: selected_models(reference, inventory, model_set)
            for model_set in reference.target_models
        },
        "target_layer_entries": target_entries,
        "target_dimensions_main_mode": target_dimensions,
        "main_mode_predictor_target_pairs": main_pairs,
        "all_mode_predictor_target_pairs": main_pairs * n_modes,
        "n_pcs_maximum_per_layer": reference.n_pcs,
        "n_permutations_main_mode": reference.n_permutations,
        "brain_scores_reused_from": str(reference.brain_output_root),
        "output_root": str(reference.output_root),
        "errors": errors[:20],
        "error_count": len(errors),
    }
    return report


def write_reference_plan(
    reference: ReferenceConfig, inventory: Inventory
) -> Path:
    report = validate_reference(reference, inventory)
    path = reference.output_root / "analysis" / "reference_plan.json"
    write_json(path, report)
    if report["status"] != "valid":
        raise RuntimeError(f"Reference inputs are invalid; see {path}")
    return path


def run_reference_universality(
    reference: ReferenceConfig,
    inventory: Inventory,
    *,
    shard_index: int,
    shard_count: int,
    overwrite: bool,
    device: str | None = None,
) -> None:
    config = reference.scoring_config(device)
    config.make_output_dirs()
    main_scorer = make_scorer(config, permutations=True)
    tasks = reference_tasks(reference, inventory)

    for task_index, item in _shard(tasks, shard_index, shard_count):
        mode, model_set, record = item
        count = n_components(reference, record)
        predictors = predictor_models(reference, inventory, model_set, record.model)
        paths = [
            universality_path(config, mode, model_set, record, predictor)
            for predictor in predictors
        ]

        if mode in config.transfer_modes:
            pending = [
                (predictor, path)
                for predictor, path in zip(predictors, paths, strict=True)
                if overwrite or not path.is_file()
            ]
            if not pending:
                continue
            train_target = load_target_scores(
                config, record, config.transfer_train_mode
            ).isel(neuroid=slice(0, count))
            test_target = load_target_scores(config, record, mode).isel(
                neuroid=slice(0, count)
            )
            for predictor, path in pending:
                train_predictor = load_model_predictor(
                    config, inventory, predictor, config.transfer_train_mode
                )
                test_predictor = load_model_predictor(
                    config, inventory, predictor, mode
                )
                arrays = score_transfer_arrays(
                    config,
                    train_predictor,
                    train_target,
                    test_predictor,
                    test_target,
                )
                save_npz(
                    path,
                    correlation=arrays["correlation"],
                    pc=np.arange(1, count + 1),
                )
                del train_predictor, test_predictor, arrays
                gc.collect()
            print(
                f"reference universality {task_index + 1}/{len(tasks)} "
                f"{mode} {model_set} {record.model} {record.node}",
                flush=True,
            )
            continue

        aggregate_path = universality_null_path(config, mode, model_set, record)
        if (
            not overwrite
            and aggregate_path.is_file()
            and all(path.is_file() for path in paths)
        ):
            continue
        target = load_target_scores(config, record, mode).isel(
            neuroid=slice(0, count)
        )
        for predictor, path in zip(predictors, paths, strict=True):
            if not overwrite and path.is_file():
                continue
            values = load_model_predictor(config, inventory, predictor, mode)
            arrays = score_arrays(main_scorer, values, target)
            save_npz(
                path,
                correlation=arrays["correlation"],
                null_correlation=arrays["null_correlation"],
                pc=np.arange(1, count + 1),
            )
            del values, arrays
            gc.collect()

        null_scores = []
        for path in paths:
            with np.load(path, allow_pickle=False) as archive:
                values = np.asarray(archive["null_correlation"], dtype=np.float32)
            expected = (reference.n_permutations, config.n_folds, count)
            if values.shape != expected:
                raise ValueError(f"Unexpected null score shape in {path}: {values.shape}")
            null_scores.append(values.mean(axis=1))
        save_npy(
            aggregate_path,
            np.median(np.stack(null_scores), axis=0).astype(np.float32),
        )
        print(
            f"reference universality {task_index + 1}/{len(tasks)} "
            f"{mode} {model_set} {record.model} {record.node}",
            flush=True,
        )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _residualize(values: np.ndarray, control: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(control)), control])
    return values - values @ design @ np.linalg.pinv(design)


def _row_correlations(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    values = values - values.mean(axis=1, keepdims=True)
    target = target - target.mean()
    denominator = np.sqrt(np.sum(values * values, axis=1) * np.sum(target * target))
    return np.divide(
        values @ target,
        denominator,
        out=np.full(len(values), np.nan),
        where=denominator > 0,
    )


def _statistics(
    frame: pd.DataFrame, null_blocks: list[np.ndarray], n_permutations: int
) -> dict[str, float]:
    universality_rank = rankdata(frame["universality"].to_numpy())
    brain_rank = rankdata(frame["brain_similarity"].to_numpy())
    pc_rank = rankdata(frame["pc"].to_numpy())
    observed = float(spearmanr(universality_rank, brain_rank).statistic)
    observed_partial = float(
        _row_correlations(
            _residualize(universality_rank[None, :], pc_rank),
            _residualize(brain_rank[None, :], pc_rank)[0],
        )[0]
    )
    null_rank = rankdata(np.concatenate(null_blocks, axis=1), axis=1)
    null_observed = _row_correlations(null_rank, brain_rank)
    null_partial = _row_correlations(
        _residualize(null_rank, pc_rank),
        _residualize(brain_rank[None, :], pc_rank)[0],
    )
    return {
        "n_dimensions": len(frame),
        "spearman_rho": observed,
        "spearman_p_permutation": float(
            (1 + np.sum(null_observed >= observed)) / (n_permutations + 1)
        ),
        "partial_spearman_rho_control_pc": observed_partial,
        "partial_spearman_p_permutation": float(
            (1 + np.sum(null_partial >= observed_partial))
            / (n_permutations + 1)
        ),
    }


def summarize_reference(
    reference: ReferenceConfig, inventory: Inventory
) -> tuple[Path, Path]:
    config = reference.scoring_config()
    brain_config = reference.brain_config
    depths = load_node_depths()
    rows: list[dict[str, object]] = []

    for mode, model_set, record in reference_tasks(reference, inventory):
        count = n_components(reference, record)
        expected_splits = config.n_folds if mode == config.main_mode else 1
        subjects = []
        for subject in config.subjects:
            path = brain_path(brain_config, mode, model_set, record, subject)
            values = _load_npz(path)["correlation"]
            if values.shape[0] != expected_splits or values.shape[1] < count:
                raise ValueError(f"Unexpected brain score shape in {path}: {values.shape}")
            subjects.append(values[:, :count].mean(axis=0))
        brain_similarity = np.stack(subjects).mean(axis=0)

        observed = []
        for predictor in predictor_models(reference, inventory, model_set, record.model):
            path = universality_path(config, mode, model_set, record, predictor)
            values = _load_npz(path)["correlation"]
            if values.shape != (expected_splits, count):
                raise ValueError(f"Unexpected universality shape in {path}: {values.shape}")
            observed.append(values.mean(axis=0))
        universality = np.median(np.stack(observed), axis=0)

        state = PCAHandoff.load(config.pca_root, model=record.model, node=record.node)
        variance_fraction = state.eigenvalues / state.eigenvalues.sum()
        depth = depths[(record.architecture, record.node)]
        for index in range(count):
            row = {
                "scope": "complete_reduced_local_reference",
                "model_set": model_set,
                "mode": mode,
                "model": record.model,
                "architecture": record.architecture,
                "node": record.node,
                "node_depth": depth,
                "pc": index + 1,
                "full_rank": record.rank,
                "eigenvalue": float(state.eigenvalues[index]),
                "variance_fraction": float(variance_fraction[index]),
                "brain_similarity": float(brain_similarity[index]),
                "universality": float(universality[index]),
            }
            row.update(
                {
                    f"brain_similarity_{subject}": float(scores[index])
                    for subject, scores in zip(config.subjects, subjects, strict=True)
                }
            )
            rows.append(row)

    dimensions = pd.DataFrame(rows)
    tables = reference.output_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    dimensions_path = tables / "reference_dimensions_all_modes.csv"
    dimensions.to_csv(dimensions_path, index=False)
    main = dimensions[dimensions["mode"] == config.main_mode].copy()
    main.to_csv(tables / "reference_figure2_dimensions.csv", index=False)

    stats_rows = []
    for model_set, group in main.groupby("model_set", sort=False):
        null_blocks = []
        for (model, node), layer in group.groupby(["model", "node"], sort=False):
            record = inventory.layer(str(model), str(node))
            values = np.load(
                universality_null_path(
                    config, config.main_mode, str(model_set), record
                ),
                allow_pickle=False,
            )
            indices = layer["pc"].to_numpy(dtype=int) - 1
            null_blocks.append(np.asarray(values)[:, indices])
        stats_rows.append(
            {
                "scope": "complete_reduced_local_reference",
                "model_set": model_set,
                **_statistics(group, null_blocks, reference.n_permutations),
            }
        )
    statistics_path = tables / "reference_figure2_statistics.csv"
    pd.DataFrame(stats_rows).to_csv(statistics_path, index=False)

    depth_parts = []
    for model_set, group in main.groupby("model_set", sort=False):
        group = group.copy()
        unique_depths = group["node_depth"].nunique()
        bins = min(20, unique_depths)
        if bins == 1:
            group["depth_bin"] = 0
        else:
            group["depth_bin"] = pd.cut(
                group["node_depth"],
                bins=np.linspace(
                    group["node_depth"].min(), group["node_depth"].max(), bins + 1
                ),
                labels=False,
                include_lowest=True,
                duplicates="drop",
            ).astype(int)
        depth_parts.append(group)
    pd.concat(depth_parts, ignore_index=True).to_csv(
        tables / "reference_figure3_dimensions.csv", index=False
    )

    return dimensions_path, statistics_path


def _reference_rsa_path(
    reference: ReferenceConfig, model_set: str, model: str
) -> Path:
    return (
        reference.output_root
        / "analysis"
        / "rsa"
        / f"model_set={model_set}"
        / f"model={model}.csv"
    )


def _condition_values(
    reference: ReferenceConfig,
    record: LayerRecord,
    ids: np.ndarray,
    dimensions: pd.DataFrame,
    n_top: int | None,
) -> np.ndarray:
    features = load_layer_features(reference.base, record, "shared")
    if n_top is None:
        return align_rows(features.stimulus_id.values, features.values, ids)
    state = PCAHandoff.load(
        reference.base.pca_root, model=record.model, node=record.node
    )
    scores = state.scores("shared")
    values = align_rows(scores.stimulus_id.values, scores.values, ids)
    layer = dimensions[
        (dimensions["model"] == record.model)
        & (dimensions["node"] == record.node)
    ].sort_values("pc")
    expected = n_components(reference, record)
    if len(layer) != expected:
        raise ValueError(f"Universality rows missing for {record.model}/{record.node}")
    keep_local = np.argsort(layer["universality"].to_numpy())[-min(n_top, expected):]
    reduced = np.zeros_like(values)
    reduced[:, keep_local] = values[:, keep_local]
    return state.inverse_transform(reduced)


def run_reference_rsa(
    reference: ReferenceConfig,
    inventory: Inventory,
    *,
    shard_index: int,
    shard_count: int,
    overwrite: bool,
) -> None:
    dimensions_path = reference.output_root / "tables" / "reference_figure2_dimensions.csv"
    if not dimensions_path.is_file():
        raise FileNotFoundError(f"Run summarize first: {dimensions_path}")
    dimensions = pd.read_csv(dimensions_path)
    train_ids, test_ids = _split_ids(reference.base, inventory)
    brain_cache: dict[tuple[str, ...], np.ndarray] = {}
    train_rdms = _brain_rdms(reference.base, train_ids, brain_cache)
    test_rdms = _brain_rdms(reference.base, test_ids, brain_cache)
    tasks = [
        (model_set, model)
        for model_set in reference.target_models
        for model in selected_models(reference, inventory, model_set)
    ]

    for task_index, item in _shard(tasks, shard_index, shard_count):
        model_set, model = item
        output = _reference_rsa_path(reference, model_set, model)
        if output.is_file() and not overwrite:
            continue
        train_scores = _node_scores(
            reference.base, inventory, model, train_ids, train_rdms
        )
        best_node = str(
            train_scores.groupby("node", as_index=False)["rsa"]
            .mean()
            .sort_values("rsa", ascending=False)
            .iloc[0]["node"]
        )
        record = inventory.layer(model, best_node)
        set_dimensions = dimensions[dimensions["model_set"] == model_set]
        rows = []
        for condition, n_top in (("all", None), ("top_10", 10), ("top_5", 5)):
            model_rdm = _rdm(
                _condition_values(
                    reference, record, test_ids, set_dimensions, n_top
                )
            )
            for subject, brain_rdm in test_rdms.items():
                rows.append(
                    {
                        "scope": "complete_reduced_local_reference",
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
        train_scores.to_csv(output.with_name(output.stem + ".train_nodes.csv"), index=False)
        print(f"reference rsa {task_index + 1}/{len(tasks)} {model_set} {model}", flush=True)


def collect_reference_rsa(
    reference: ReferenceConfig, inventory: Inventory
) -> pd.DataFrame:
    paths = [
        _reference_rsa_path(reference, model_set, model)
        for model_set in reference.target_models
        for model in selected_models(reference, inventory, model_set)
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} RSA outputs; first={missing[0]}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    frame.to_csv(reference.output_root / "tables" / "reference_figure5_rsa.csv", index=False)
    return frame


def _quantile_line(group: pd.DataFrame, bins: int = 50) -> pd.DataFrame:
    group = group.sort_values("universality").copy()
    group["bin"] = np.arange(len(group)) // max(1, math.ceil(len(group) / bins))
    return group.groupby("bin", as_index=False)[["universality", "brain_similarity"]].mean()


def _save(figure: plt.Figure, reference: ReferenceConfig, stem: str) -> list[Path]:
    directory = reference.output_root / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in reference.figure_formats:
        path = directory / f"{stem}.{extension}"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(path)
    plt.close(figure)
    return outputs


def make_reference_figures(
    reference: ReferenceConfig, inventory: Inventory
) -> list[Path]:
    tables = reference.output_root / "tables"
    main = pd.read_csv(tables / "reference_figure2_dimensions.csv")
    depth = pd.read_csv(tables / "reference_figure3_dimensions.csv")
    all_modes = pd.read_csv(tables / "reference_dimensions_all_modes.csv")
    rsa = collect_reference_rsa(reference, inventory)
    outputs: list[Path] = []

    figure, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    for axis, model_set in zip(axes.flat, reference.target_models, strict=True):
        group = main[main["model_set"] == model_set]
        axis.hexbin(group["universality"], group["brain_similarity"], gridsize=45, mincnt=1, bins="log", cmap="Greys")
        line = _quantile_line(group)
        axis.plot(line["universality"], line["brain_similarity"], color="#d95f02", linewidth=2)
        axis.set_title(SET_LABELS[model_set])
        axis.set_xlabel("Universality")
        axis.set_ylabel("Brain similarity")
    figure.suptitle(f"Reduced local reference: first {reference.n_pcs} PCs")
    figure.tight_layout()
    outputs.extend(_save(figure, reference, "reference_figure2_universality_brain_similarity"))

    figure, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    cmap = plt.get_cmap("viridis")
    for axis, model_set in zip(axes.flat, reference.target_models, strict=True):
        subset = depth[depth["model_set"] == model_set]
        maximum = max(int(subset["depth_bin"].max()), 1)
        for depth_bin, group in subset.groupby("depth_bin", sort=True):
            line = _quantile_line(group, bins=20)
            axis.plot(line["universality"], line["brain_similarity"], color=cmap(float(depth_bin) / maximum), alpha=0.8)
        axis.set_title(SET_LABELS[model_set])
        axis.set_xlabel("Universality")
        axis.set_ylabel("Brain similarity")
    figure.suptitle(f"Reduced local depth reference: first {reference.n_pcs} PCs")
    figure.tight_layout()
    outputs.extend(_save(figure, reference, "reference_figure3_depth_resolved_relationship"))

    per_model = rsa.groupby(["model_set", "model", "condition"], as_index=False)["rsa"].mean()
    summary = per_model.groupby(["model_set", "condition"])["rsa"].agg(["mean", "sem"]).reset_index()
    order = ["all", "top_10", "top_5"]
    x = np.arange(len(reference.target_models))
    width = 0.24
    figure, axis = plt.subplots(figsize=(9, 4.5))
    for index, condition in enumerate(order):
        group = summary[summary["condition"] == condition].set_index("model_set").reindex(reference.target_models)
        errors = group["sem"].fillna(0).to_numpy()
        axis.bar(x + (index - 1) * width, group["mean"], width, yerr=errors, label=condition)
    axis.set_xticks(x, [SET_LABELS[item] for item in reference.target_models], rotation=15, ha="right")
    axis.set_ylabel("Held-out RSA")
    axis.legend(frameon=False)
    figure.suptitle(f"Reduced local RSA reference: top PCs selected among first {reference.n_pcs}")
    figure.tight_layout()
    outputs.extend(_save(figure, reference, "reference_figure5_rsa_universal_subspaces"))
    return outputs
