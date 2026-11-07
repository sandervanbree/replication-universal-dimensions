#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("XDG_CONFIG_HOME", str(PROJECT_ROOT / "data" / ".config"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "data" / ".matplotlib"))

import numpy as np
import pandas as pd
import xarray as xr
from laion_fmri import load_subject
from laion_fmri.config import dataset_initialize

sys.path.insert(0, str(PROJECT_ROOT))

from src.laion_cache import project_beta_path  # noqa: E402
from src.pipeline_config import load_profile, profile_from_args  # noqa: E402


SUBJECTS = ("sub-01", "sub-03", "sub-05", "sub-06", "sub-07")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache averaged LAION-fMRI betas for a configured ROI.")
    parser.add_argument("--profile", default=None, help="Profile name or JSON path. Defaults to default.")
    parser.add_argument("--debug", choices=["0", "1", "2"], default=None, help="Shortcut: 1=debug, 2=full_debug, 0=default.")
    parser.add_argument("--data-dir", type=Path, default=None, help="LAION-fMRI dataset root.")
    parser.add_argument("--beta-root", type=Path, default=None, help="Output root for averaged beta caches.")
    parser.add_argument("--roi", default=None, help="ROI label. Defaults to the profile ROI.")
    parser.add_argument("--mode", default=None, choices=("shared", "shared-all", "ood", "all"))
    parser.add_argument("--subjects", default=None, help="Comma-separated subjects, or all. Defaults to the profile subjects.")
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--all-stimuli", action="store_true", help="Ignore profile n_stimuli and cache every target image.")
    parser.add_argument(
        "--zscore-session",
        action="store_true",
        help="Z-score each session across trials before averaging repetitions (Chen & Bonner / proposal).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_subjects(value: str | None, profile_subjects: list[str]) -> list[str]:
    if value is None:
        return profile_subjects
    if value == "all":
        return list(SUBJECTS)
    return [item.strip() for item in value.split(",") if item.strip()]


def atomic_to_netcdf(data: xr.DataArray | xr.Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data.to_netcdf(tmp)
    tmp.replace(path)


def is_ood(labels: pd.Index | pd.Series | np.ndarray) -> np.ndarray:
    values = pd.Index(labels).astype(str)
    return np.asarray(values.str.startswith("shared_4rep_OOD_"), dtype=bool)


def load_laion_metadata(data_dir: Path, mode: str) -> pd.DataFrame:
    metadata = pd.read_csv(data_dir / "stimuli" / "task-images_metadata.csv")
    metadata = metadata.set_index("image_name", drop=False)
    match mode:
        case "shared":
            metadata = metadata[
                (metadata["unique_or_shared"] == "shared") & ~is_ood(metadata.index)
            ]
        case "shared-all":
            metadata = metadata[metadata["unique_or_shared"] == "shared"]
        case "ood":
            metadata = metadata[is_ood(metadata.index)]
        case "all":
            pass
        case _:
            raise ValueError(f"Unsupported LAION mode: {mode}")
    return metadata.sort_index()


def select_target_labels(labels: np.ndarray, n_stimuli: int | None, seed: int) -> np.ndarray:
    if n_stimuli is None or n_stimuli <= 0 or labels.size <= n_stimuli:
        return labels
    rng = np.random.default_rng(seed)
    selected = sorted(rng.choice(labels.size, size=int(n_stimuli), replace=False).tolist())
    return labels[np.asarray(selected, dtype=int)]


def cache_subject(
    *,
    data_dir: Path,
    beta_root: Path,
    subject_id: str,
    roi: str,
    mode: str,
    target_labels: np.ndarray,
    max_sessions: int | None,
    overwrite: bool,
    zscore_session: bool = False,
) -> None:
    path = project_beta_path(beta_root, roi, subject_id, mode)
    if path.exists() and not overwrite:
        print(f"exists subject={subject_id} roi={roi} mode={mode} path={path}", flush=True)
        return

    label_to_index = {label: idx for idx, label in enumerate(target_labels)}
    subject = load_subject(subject_id)
    sessions = subject.get_sessions()
    if max_sessions is not None:
        sessions = sessions[:max_sessions]

    sums: np.ndarray | None = None
    counts = np.zeros(len(target_labels), dtype=np.int32)
    n_trials_kept = 0

    for session in sessions:
        trials = subject.get_trial_info(session)
        labels = trials["label"].astype(str).to_numpy()
        indices = np.array([label_to_index.get(label, -1) for label in labels])
        keep = indices >= 0
        if not np.any(keep):
            continue

        betas = np.asarray(subject.get_betas(session, roi=roi, streaming=True), dtype=np.float32)
        if zscore_session:
            mu = betas.mean(axis=0, keepdims=True)
            sd = betas.std(axis=0, keepdims=True)
            sd[sd == 0] = 1.0
            betas = (betas - mu) / sd
        betas = betas[keep]
        if sums is None:
            sums = np.zeros((len(target_labels), betas.shape[1]), dtype=np.float64)

        np.add.at(sums, indices[keep], betas)
        np.add.at(counts, indices[keep], 1)
        n_trials_kept += int(keep.sum())
        print(f"subject={subject_id} session={session} kept={int(keep.sum())}", flush=True)

    if sums is None:
        raise RuntimeError(f"No matching trials for subject={subject_id} roi={roi} mode={mode}")

    present = counts > 0
    averaged = (sums[present] / counts[present, None]).astype(np.float32)
    labels = target_labels[present]
    da = xr.DataArray(
        averaged,
        dims=("presentation", "neuroid"),
        coords={
            "stimulus_id": ("presentation", labels.astype(str)),
            "neuroid": np.arange(averaged.shape[1]),
            "subject": subject_id,
            "roi": roi,
        },
        name="beta",
        attrs={
            "mode": mode,
            "sessions": ",".join(map(str, sessions)),
            "n_trials_kept": n_trials_kept,
            "zscore_session": int(bool(zscore_session)),
        },
    )
    atomic_to_netcdf(da, path)
    print(f"wrote subject={subject_id} shape={tuple(da.shape)} path={path}", flush=True)


def main() -> None:
    args = parse_args()
    cfg = load_profile(profile_from_args(args.profile, args.debug))
    paths = cfg["paths"]
    dataset = cfg["dataset"]

    data_dir = (args.data_dir or Path(paths["laion_data_root"])).resolve()
    beta_root = (args.beta_root or Path(paths["beta_root"])).resolve()
    roi = args.roi or dataset["roi"]
    mode = args.mode or dataset.get("stimulus_mode", "shared")
    subjects = parse_subjects(args.subjects, list(dataset["subjects"]))

    dataset_initialize(str(data_dir))
    metadata = load_laion_metadata(data_dir, mode)
    target_labels = metadata.index.astype(str).to_numpy()
    if not args.all_stimuli:
        target_labels = select_target_labels(
            target_labels,
            dataset.get("n_stimuli"),
            int(dataset.get("random_seed", 42)),
        )
    print(
        f"profile={cfg['profile']} roi={roi} mode={mode} subjects={len(subjects)} "
        f"target_stimuli={len(target_labels)} beta_root={beta_root}",
        flush=True,
    )

    for subject_id in subjects:
        cache_subject(
            data_dir=data_dir,
            beta_root=beta_root,
            subject_id=subject_id,
            roi=roi,
            mode=mode,
            target_labels=target_labels,
            max_sessions=args.max_sessions,
            overwrite=args.overwrite,
            zscore_session=args.zscore_session,
        )


if __name__ == "__main__":
    main()
