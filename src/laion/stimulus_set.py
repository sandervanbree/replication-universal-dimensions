"""LAION-fMRI stimulus set with the same interface Chen & Bonner use for NSD.

`Model.__call__` expects a MapDataPipe that maps stimulus_id -> PIL.Image,
plus `.identifier` and `.metadata` (index = image names).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from torchdata.datapipes.map import MapDataPipe

from laion_fmri import load_stimuli
from laion_fmri.config import dataset_initialize


MODES = ("shared", "unique", "ood", "shared-all", "all")


def is_ood(image_names: pd.Index) -> np.ndarray:
    return np.asarray(
        pd.Index(image_names).astype(str).str.startswith("shared_4rep_OOD_"),
        dtype=bool,
    )


def load_metadata(
    data_dir: Path,
    mode: str,
    n_stimuli: int | None = None,
    seed: int = 11,
) -> pd.DataFrame:
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}. Choose from {MODES}.")

    metadata = pd.read_csv(data_dir / "stimuli" / "task-images_metadata.csv")
    metadata = metadata.set_index("image_name", drop=False)

    match mode:
        case "shared":
            metadata = metadata[
                (metadata["unique_or_shared"] == "shared") & ~is_ood(metadata.index)
            ]
        case "shared-all":
            metadata = metadata[metadata["unique_or_shared"] == "shared"]
        case "unique":
            metadata = metadata[metadata["unique_or_shared"] == "unique"]
        case "ood":
            metadata = metadata[is_ood(metadata.index)]
        case "all":
            pass

    if n_stimuli is not None:
        metadata = metadata.sample(n=n_stimuli, random_state=seed, axis=0)
    return metadata.sort_index()


class LaionStimulusSet(MapDataPipe):
    def __init__(self, metadata: pd.DataFrame, identifier: str, data_dir: Path) -> None:
        self.metadata = metadata
        self.identifier = identifier
        self.data_dir = Path(data_dir)
        self._stimuli = None

    def __getitem__(self, stimulus_id: str):
        if self._stimuli is None:
            dataset_initialize(str(self.data_dir))
            self._stimuli = load_stimuli(self.data_dir)
        return self._stimuli.images.get(stimulus_id).convert("RGB")

    def __len__(self) -> int:
        return len(self.metadata)


def load_laion_stimulus_set(
    *,
    data_dir: Path,
    mode: str = "shared",
    n_stimuli: int | None = None,
    seed: int = 11,
) -> LaionStimulusSet:
    metadata = load_metadata(data_dir, mode, n_stimuli=n_stimuli, seed=seed)
    identifier = f"laion_fmri.{mode}"
    if n_stimuli is not None:
        identifier += f".n_stimuli={n_stimuli}.seed={seed}"
    return LaionStimulusSet(metadata=metadata, identifier=identifier, data_dir=data_dir)
