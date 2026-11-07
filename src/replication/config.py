from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _io_path(path: Path) -> Path:
    if os.name == "nt":
        return Path("\\\\?\\" + str(path.resolve()))
    return path


@dataclass(frozen=True)
class RunConfig:
    pca_root: Path
    feature_root: Path
    beta_root: Path
    output_root: Path
    model_sets: tuple[str, ...]
    subjects: tuple[str, ...]
    modes: tuple[str, ...]
    main_mode: str
    transfer_train_mode: str
    transfer_modes: tuple[str, ...]
    roi: str
    regression: str
    n_folds: int
    seed: int
    alphas: tuple[float, ...]
    n_permutations: int
    device: str
    figure_formats: tuple[str, ...]
    generalization_splits_path: Path | None = None
    reliability_summary_path: Path | None = None

    @property
    def analysis_dir(self) -> Path:
        return _io_path(self.output_root / "analysis")

    @property
    def tables_dir(self) -> Path:
        return self.output_root / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.output_root / "figures"

    def make_output_dirs(self) -> None:
        for path in (self.analysis_dir, self.tables_dir, self.figures_dir):
            path.mkdir(parents=True, exist_ok=True)


def _path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(path: Path | str) -> RunConfig:
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    base = REPO_ROOT
    return RunConfig(
        pca_root=_path(raw["pca_root"], base),
        feature_root=_path(raw["feature_root"], base),
        beta_root=_path(raw["beta_root"], base),
        output_root=_path(raw.get("output_root", "."), base),
        model_sets=tuple(raw["model_sets"]),
        subjects=tuple(raw["subjects"]),
        modes=tuple(raw["modes"]),
        main_mode=raw["main_mode"],
        transfer_train_mode=raw["transfer_train_mode"],
        transfer_modes=tuple(raw["transfer_modes"]),
        roi=raw["roi"],
        regression=raw["regression"],
        n_folds=int(raw["n_folds"]),
        seed=int(raw["seed"]),
        alphas=tuple(float(value) for value in raw["alphas"]),
        n_permutations=int(raw["n_permutations"]),
        device=raw["device"],
        figure_formats=tuple(raw.get("figure_formats", ["png", "pdf"])),
        generalization_splits_path=(
            _path(raw["generalization_splits_path"], base)
            if raw.get("generalization_splits_path")
            else None
        ),
        reliability_summary_path=(
            _path(raw["reliability_summary_path"], base)
            if raw.get("reliability_summary_path")
            else None
        ),
    )
