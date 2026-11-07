from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def replication_beta_path(replication_root: Path, roi: str, subject: str, mode: str = "shared") -> Path:
    return (
        replication_root
        / "cache"
        / "bonner-caching"
        / "universal_dimensions"
        / "laion_fmri"
        / "betas"
        / f"roi={roi}"
        / f"mode={mode}"
        / f"subject={subject}.nc"
    )


def project_beta_path(beta_root: Path, roi: str, subject: str, mode: str = "shared") -> Path:
    return beta_root / f"roi={roi}" / f"mode={mode}" / f"subject={subject}.nc"


def beta_path(paths: dict[str, str], roi: str, subject: str, mode: str = "shared") -> Path:
    if "beta_root" in paths:
        return project_beta_path(Path(paths["beta_root"]), roi, subject, mode)
    return replication_beta_path(Path(paths["replication_root"]), roi, subject, mode)


def replication_feature_path(
    replication_root: Path,
    model_identifier: str,
    stimulus_mode: str,
    node: str,
    hook: str,
) -> Path:
    return (
        replication_root
        / "cache"
        / "bonner-models"
        / "universal_dimensions"
        / "features"
        / model_identifier
        / f"datapipe=laion_fmri.{stimulus_mode}"
        / f"node={node}"
        / f"hook={hook}.nc"
    )


def project_feature_path(
    feature_root: Path,
    model_identifier: str,
    stimulus_mode: str,
    node: str,
    hook: str,
) -> Path:
    return (
        feature_root
        / model_identifier
        / f"datapipe=laion_fmri.{stimulus_mode}"
        / f"node={node}"
        / f"hook={hook}.nc"
    )


def feature_path(
    paths: dict[str, str],
    model_identifier: str,
    stimulus_mode: str,
    node: str,
    hook: str,
) -> Path:
    replication_path = replication_feature_path(
        Path(paths["replication_root"]),
        model_identifier,
        stimulus_mode,
        node,
        hook,
    )
    if "feature_root" not in paths:
        return replication_path

    local_path = project_feature_path(
        Path(paths["feature_root"]),
        model_identifier,
        stimulus_mode,
        node,
        hook,
    )
    if local_path.exists() or not replication_path.exists():
        return local_path
    return replication_path


def stimulus_ids_from_dataset(ds: xr.Dataset) -> np.ndarray:
    if "stimulus_id" not in ds:
        raise KeyError("Dataset has no stimulus_id coordinate or variable")
    return ds["stimulus_id"].load().values.astype(str)


def load_stimulus_ids(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    with xr.open_dataset(path) as ds:
        return stimulus_ids_from_dataset(ds)


def load_brain_matrix(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with xr.open_dataset(path) as ds:
        matrix = ds["beta"].load().values.astype(np.float32)
        stimulus_ids = stimulus_ids_from_dataset(ds)
    return matrix, stimulus_ids


def load_feature_matrix(path: Path, node: str) -> tuple[np.ndarray, np.ndarray, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    with xr.open_dataset(path) as ds:
        if node in ds.data_vars:
            data_var = node
        else:
            candidates = [name for name in ds.data_vars if name != "stimulus_id"]
            if not candidates:
                raise KeyError(f"No feature variable found in {path}")
            data_var = candidates[0]
        matrix = ds[data_var].load().values.astype(np.float32)
        stimulus_ids = stimulus_ids_from_dataset(ds)

    if matrix.ndim > 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    return matrix, stimulus_ids, data_var
