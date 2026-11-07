"""Load the validated unique-fit PCA handoff produced on the cluster."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

REQUIRED_KEYS = {
    "format_version", "model", "node", "hook", "model_sets", "fit_split",
    "projected_splits", "rank", "n_fit_presentations", "n_input_features",
    "mean", "std", "source_mean", "source_std", "eigenvectors",
    "eigenvalues", "shared_scores", "ood_scores",
}


def _scalar(value: np.ndarray) -> object:
    return np.asarray(value).item()


def _read_ids(path: Path) -> np.ndarray:
    values = np.asarray(
        [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()],
        dtype=str,
    )
    if len(values) != len(set(values.tolist())):
        raise ValueError(f"Duplicate stimulus IDs in {path}")
    return values


@dataclass(frozen=True)
class PCAHandoff:
    """One model/layer PCA state plus precomputed shared and OOD projections."""

    root: Path
    model: str
    node: str
    hook: str
    model_sets: str
    rank: int
    n_fit_presentations: int
    n_input_features: int
    mean: np.ndarray
    std: np.ndarray
    source_mean: np.ndarray
    source_std: np.ndarray
    eigenvectors: np.ndarray
    eigenvalues: np.ndarray
    shared_scores: np.ndarray
    ood_scores: np.ndarray

    @classmethod
    def load(cls, root: Path | str, *, model: str, node: str) -> "PCAHandoff":
        root = Path(root).resolve()
        path = root / "layers" / model / f"node={node}" / "pca.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as archive:
            missing = REQUIRED_KEYS - set(archive.files)
            if missing:
                raise ValueError(f"Missing keys in {path}: {sorted(missing)}")
            if str(_scalar(archive["model"])) != model:
                raise ValueError(f"Embedded model does not match {model!r}")
            if str(_scalar(archive["node"])) != node:
                raise ValueError(f"Embedded node does not match {node!r}")
            if str(_scalar(archive["fit_split"])) != "unique":
                raise ValueError("PCA handoff was not fitted on the unique split")
            projected = [str(item) for item in archive["projected_splits"].tolist()]
            if projected != ["shared", "ood"]:
                raise ValueError(f"Unexpected projected splits: {projected}")
            rank = int(_scalar(archive["rank"]))
            n_input = int(_scalar(archive["n_input_features"]))
            values = {
                key: np.asarray(archive[key]).copy()
                for key in (
                    "mean", "std", "source_mean", "source_std", "eigenvectors",
                    "eigenvalues", "shared_scores", "ood_scores",
                )
            }
            instance = cls(
                root=root, model=model, node=node,
                hook=str(_scalar(archive["hook"])),
                model_sets=str(_scalar(archive["model_sets"])),
                rank=rank,
                n_fit_presentations=int(_scalar(archive["n_fit_presentations"])),
                n_input_features=n_input,
                **values,
            )
        instance._validate_shapes()
        return instance

    def _validate_shapes(self) -> None:
        expected = {
            "mean": (1, self.n_input_features),
            "eigenvectors": (self.n_input_features, self.rank),
            "eigenvalues": (self.rank,),
            "shared_scores": (1121, self.rank),
            "ood_scores": (371, self.rank),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape:
                raise ValueError(f"{name} has shape {value.shape}; expected {shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
        if self.n_fit_presentations != 23560:
            raise ValueError(f"Expected 23,560 unique fit images, got {self.n_fit_presentations}")

    def stimulus_ids(self, split: str) -> np.ndarray:
        if split not in {"shared", "ood", "unique"}:
            raise ValueError("split must be 'shared', 'ood', or 'unique'")
        return _read_ids(self.root / "stimulus_ids" / f"{split}.txt")

    def scores(self, split: str = "shared") -> xr.DataArray:
        if split not in {"shared", "ood"}:
            raise ValueError("Projected scores are available only for 'shared' and 'ood'")
        data = self.shared_scores if split == "shared" else self.ood_scores
        ids = self.stimulus_ids(split)
        if data.shape[0] != len(ids):
            raise ValueError(f"{split} row count {data.shape[0]} does not match {len(ids)} IDs")
        return xr.DataArray(
            data,
            dims=("presentation", "neuroid"),
            coords={
                "stimulus_id": ("presentation", ids),
                "neuroid": np.arange(1, self.rank + 1),
                "pc": ("neuroid", np.arange(1, self.rank + 1)),
            },
            name=f"pca_scores.fit=unique.split={split}.model={self.model}.node={self.node}",
            attrs={
                "fit_split": "unique", "n_fit_presentations": self.n_fit_presentations,
                "model": self.model, "node": self.node, "hook": self.hook,
                "model_sets": self.model_sets, "rank": self.rank,
            },
        )

    def inverse_transform(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores)
        if scores.ndim != 2 or scores.shape[1] != self.rank:
            raise ValueError(f"Expected scores with shape (n, {self.rank}), got {scores.shape}")
        return scores @ self.eigenvectors.T + self.mean


def load_pca_scores(
    root: Path | str, *, model: str, node: str, split: str = "shared"
) -> xr.DataArray:
    """Return ordered PC scores as an xarray DataArray."""
    return PCAHandoff.load(root, model=model, node=node).scores(split)
