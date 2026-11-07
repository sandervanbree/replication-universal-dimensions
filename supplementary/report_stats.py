"""Numbers for the re:vision report that are not already in outputs/tables.

Read-only on the project: imports the run's own statistics code and writes CSVs
to the directory given by --output.

  A. Finding 1 per participant: Spearman / partial rho with the run's 5,000-permutation
     null (null universality vs. that participant's brain similarity).
  B. Rank-tolerance sensitivity: subject-averaged Finding 1 restricted to PCs that pass
     the original study's effective matrix_rank tolerance (72,128 fit images).
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config', type=Path, default=REPO / 'config/analysis.json')
parser.add_argument('--tolerance', type=Path, default=REPO / 'outputs/report/rank_tolerance.csv')
parser.add_argument('--output', type=Path, default=REPO / 'outputs/report')
parser.add_argument('--workers', type=int, default=2)
args = parser.parse_args()
RANK_SENS = args.tolerance
SUBJECTS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]

sys.path.insert(0, str(REPO))
from src.replication.config import load_config  # noqa: E402
from src.replication.data import Inventory  # noqa: E402
from src.replication.figures import permutation_statistics  # noqa: E402

config = load_config(args.config)
inventory = Inventory(config.pca_root)

frame = pd.read_csv(config.tables_dir / "figure2_dimensions.csv")
tolerance = pd.read_csv(RANK_SENS)[["model", "node", "k_orig_tol"]]
frame = frame.merge(tolerance, on=["model", "node"], how="left", validate="many_to_one")
assert frame["k_orig_tol"].notna().all()


def task(args: tuple[str, str]) -> dict[str, object]:
    model_set, target = args
    group = frame[frame["model_set"] == model_set]
    if target == "original_tolerance":
        group = group[group["pc"] <= group["k_orig_tol"]].copy()
    else:
        group = group.copy()
        if target != "average":
            group["brain_similarity"] = group[f"brain_similarity_{target}"]
    stats = permutation_statistics(config, inventory, group)
    return {"model_set": model_set, "target": target, **stats}


if __name__ == "__main__":
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    jobs = [
        (model_set, target)
        for model_set in config.model_sets
        for target in ["average", "original_tolerance", *SUBJECTS]
    ]
    with Pool(args.workers) as pool:
        rows = pool.map(task, jobs, chunksize=1)
    pd.DataFrame(rows).to_csv(out / "finding1_statistics.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string())
