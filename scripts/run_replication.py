#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys

import pandas as pd
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

from src.replication.config import load_config
from src.replication.data import Inventory, validate_inputs, write_json
from src.replication.depths import run_depths
from src.replication.figures import run_figures
from src.replication.rsa import run_rsa
from src.replication.scoring import (
    run_brain,
    run_universality,
    target_tasks,
    write_run_manifest,
)
from src.replication.summary import run_summaries


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the LAION-fMRI universality replication"
    )
    parser.add_argument(
        "stage",
        choices=(
            "plan",
            "validate",
            "depths",
            "brain",
            "universality",
            "summarize",
            "rsa",
            "figures",
            "all",
        ),
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config" / "analysis.json"
    )
    parser.add_argument("--pca-root", type=Path)
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--beta-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--modes", help="Comma-separated evaluation modes")
    parser.add_argument(
        "--shard-index", type=int, default=int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=int(os.getenv("REPLICATION_SHARD_COUNT", "1")),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def apply_overrides(config, args):
    values = {}
    for name in ("pca_root", "feature_root", "beta_root", "output_root"):
        value = getattr(args, name)
        if value is not None:
            values[name] = value.resolve()
    if args.device:
        values["device"] = args.device
    if args.modes:
        values["modes"] = tuple(
            item.strip() for item in args.modes.split(",") if item.strip()
        )
    return replace(config, **values)


def write_plan(config, inventory):
    rows = []
    for index, (mode, model_set, record) in enumerate(target_tasks(config, inventory)):
        rows.append(
            {
                "task_index": index,
                "mode": mode,
                "model_set": model_set,
                "model": record.model,
                "node": record.node,
                "rank": record.rank,
                "brain_outputs": len(config.subjects),
                "universality_outputs": len(inventory.models(model_set)) - 1,
            }
        )
    path = config.tables_dir / "task_plan.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"wrote {path} with {len(rows)} target-layer tasks")


def validate(config, inventory):
    report = validate_inputs(config, inventory)
    path = config.analysis_dir / "input_validation.json"
    write_json(path, report)
    if report["status"] != "valid":
        raise RuntimeError(f"Input validation failed; see {path}")
    print(f"input validation passed: {path}")


def main() -> None:
    args = arguments()
    config = apply_overrides(load_config(args.config), args)
    config.make_output_dirs()
    inventory = Inventory(config.pca_root)
    if args.stage in {"plan", "validate", "all"} and args.shard_index == 0:
        write_run_manifest(config, inventory)

    if args.stage == "plan":
        write_plan(config, inventory)
    elif args.stage == "validate":
        validate(config, inventory)
    elif args.stage == "depths":
        run_depths(config, inventory, overwrite=args.overwrite)
    elif args.stage == "brain":
        run_brain(
            config,
            inventory,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            overwrite=args.overwrite,
        )
    elif args.stage == "universality":
        run_universality(
            config,
            inventory,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            overwrite=args.overwrite,
        )
    elif args.stage == "summarize":
        run_summaries(config, inventory)
    elif args.stage == "rsa":
        run_rsa(
            config,
            inventory,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            overwrite=args.overwrite,
        )
    elif args.stage == "figures":
        run_figures(config, inventory)
    else:
        if args.shard_count != 1:
            raise ValueError("Use individual stages for a sharded cluster run")
        validate(config, inventory)
        write_plan(config, inventory)
        run_depths(config, inventory, overwrite=args.overwrite)
        run_brain(
            config, inventory, shard_index=0, shard_count=1, overwrite=args.overwrite
        )
        run_universality(
            config, inventory, shard_index=0, shard_count=1, overwrite=args.overwrite
        )
        run_summaries(config, inventory)
        run_rsa(
            config, inventory, shard_index=0, shard_count=1, overwrite=args.overwrite
        )
        run_figures(config, inventory)


if __name__ == "__main__":
    main()
