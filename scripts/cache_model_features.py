#!/usr/bin/env python
"""Extract globally max-pooled features with Chen & Bonner's model zoo."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
ORIG = ROOT / "orig_codebase"
load_dotenv(ROOT / ".env")

os.environ.setdefault("PROJECT_HOME", str(ROOT))
os.environ.setdefault("TORCH_HOME", str(ROOT / "data" / "cache" / "torch"))
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "cache" / "huggingface"))
os.environ.setdefault("BONNER_MODELS_HOME", str(ROOT / "data" / "cache" / "bonner-models"))
os.environ.setdefault("BONNER_CACHING_HOME", str(ROOT / "data" / "cache" / "bonner-caching"))
os.environ.setdefault("BONNER_DATASETS_HOME", str(ROOT / "data"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data" / "cache" / "matplotlib"))

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ORIG / "src"))

import torch

from lib.analyses.yield_models import load_yielder, yield_models
from src.laion.stimulus_set import MODES, load_laion_stimulus_set


YIELDERS = (
    "resnet18_classification_imagenet1k_varied_seeds",
    "classification_imagenet1k_varied_architectures",
    "resnet50_imagenet1k_varied_tasks",
    "untrained_resnet18_varied_seeds",
)

DEFAULT_DATA_DIR = Path(
    os.environ.get("LAION_STIMULI_DIR", str(ROOT / "data" / "laion_fmri"))
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache Chen & Bonner model features on LAION-fMRI images."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--modes",
        default="unique,shared,ood",
        help="Comma-separated LAION splits. Default: unique,shared,ood",
    )
    parser.add_argument(
        "--yielder",
        default="resnet50_imagenet1k_varied_tasks",
        help="Chen & Bonner yielder name, or 'demo' for one ResNet-50.",
    )
    parser.add_argument("--sidx", type=int, default=0)
    parser.add_argument("--nyield", type=int, default=9999)
    parser.add_argument("--n-stimuli", type=int, default=16, help="Debug subset size unless --full.")
    parser.add_argument("--full", action="store_true", help="Use every image in each mode.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    return parser.parse_args()


def resolve_yielder(name: str) -> str:
    if name == "demo":
        return "resnet50_imagenet1k_varied_tasks"
    if name not in YIELDERS:
        raise ValueError(f"Unknown yielder {name!r}. Choose from {YIELDERS} or 'demo'.")
    return name


def main() -> None:
    args = parse_args()
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    unknown = [mode for mode in modes if mode not in MODES]
    if unknown:
        raise ValueError(f"Unknown modes {unknown}. Choose from {MODES}.")
    if not args.data_dir.exists():
        raise FileNotFoundError(args.data_dir)

    n_stimuli = None if args.full else args.n_stimuli
    yielder_name = resolve_yielder(args.yielder)
    nyield = 1 if args.yielder == "demo" else args.nyield

    stimulus_sets = {
        mode: load_laion_stimulus_set(
            data_dir=args.data_dir,
            mode=mode,
            n_stimuli=n_stimuli,
            seed=args.seed,
        )
        for mode in modes
    }
    for mode, stimulus_set in stimulus_sets.items():
        print(f"stimulus_set={stimulus_set.identifier} n={len(stimulus_set)}", flush=True)

    print(f"yielder={yielder_name} sidx={args.sidx} nyield={nyield}", flush=True)
    print(f"torch_cuda_available={torch.cuda.is_available()}", flush=True)
    print(f"BONNER_MODELS_HOME={os.environ['BONNER_MODELS_HOME']}", flush=True)

    yielder = load_yielder(yielder_name)
    for model in yield_models(yielder, num_yield=nyield, start_idx=args.sidx):
        print(f"model={model.identifier} nodes={len(model.nodes)}", flush=True)
        if args.list_only:
            continue
        for stimulus_set in stimulus_sets.values():
            cache_dir = (
                Path(os.environ["BONNER_MODELS_HOME"])
                / "features"
                / model.identifier
                / f"datapipe={stimulus_set.identifier}"
            )
            if args.overwrite and cache_dir.exists():
                shutil.rmtree(cache_dir)
            print(
                f"extracting model={model.identifier} stimulus_set={stimulus_set.identifier}",
                flush=True,
            )
            features = model(stimulus_set, batch_size=args.batch_size)
            print(
                f"done model={model.identifier} stimulus_set={stimulus_set.identifier} "
                f"shape={tuple(features.shape)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
