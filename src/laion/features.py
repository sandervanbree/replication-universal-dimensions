"""Load cached globally max-pooled features from BONNER_MODELS_HOME."""
from __future__ import annotations

from pathlib import Path

import xarray as xr
from bonner.models.features import concatenate_features, flatten_features


def features_root(bonner_models_home: Path) -> Path:
    return Path(bonner_models_home) / "features"


def list_models(root: Path) -> list[str]:
    return sorted(path.name for path in Path(root).iterdir() if path.is_dir())


def list_modes(root: Path, model_identifier: str) -> list[str]:
    model_dir = Path(root) / model_identifier
    prefix = "datapipe=laion_fmri."
    return sorted(
        path.name.removeprefix(prefix)
        for path in model_dir.iterdir()
        if path.is_dir() and path.name.startswith(prefix)
    )


def list_nodes(root: Path, model_identifier: str, mode: str) -> list[str]:
    mode_dir = Path(root) / model_identifier / f"datapipe=laion_fmri.{mode}"
    return sorted(
        path.name.removeprefix("node=")
        for path in mode_dir.iterdir()
        if path.is_dir() and path.name.startswith("node=")
    )


def node_file(root: Path, model_identifier: str, mode: str, node: str) -> Path:
    node_dir = Path(root) / model_identifier / f"datapipe=laion_fmri.{mode}" / f"node={node}"
    files = sorted(node_dir.glob("hook=*.nc"))
    if not files:
        raise FileNotFoundError(node_dir)
    if len(files) > 1:
        raise RuntimeError(f"Multiple hook files in {node_dir}: {files}")
    return files[0]


def load_node(
    root: Path,
    model_identifier: str,
    mode: str,
    node: str,
) -> xr.DataArray:
    path = node_file(root, model_identifier, mode, node)
    hook = path.stem.removeprefix("hook=")
    with xr.open_dataset(path) as dataset:
        data = dataset[node].load()
        stimulus_ids = dataset["stimulus_id"].values.astype(str)
    return data.assign_coords(stimulus_id=("presentation", stimulus_ids)).rename(
        f"{model_identifier}.node={node}.hook={hook}.laion_fmri.{mode}"
    )


def load_features(
    root: Path,
    model_identifier: str,
    mode: str,
    nodes: list[str] | None = None,
) -> xr.DataArray:
    nodes = list_nodes(root, model_identifier, mode) if nodes is None else nodes
    features = {node: load_node(root, model_identifier, mode, node) for node in nodes}
    features = flatten_features({node: array.copy() for node, array in features.items()})
    return concatenate_features(features).rename(f"{model_identifier}.laion_fmri.{mode}")
