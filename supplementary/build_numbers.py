"""Collect every number shown in the report draft into numbers.json (--output = out dir)."""
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tables', type=Path, default=REPO / 'outputs/tables')
parser.add_argument('--output', type=Path, default=REPO / 'outputs/report')
args = parser.parse_args()
T = args.tables
OUT = args.output
SETS = [
    ("resnet18_classification_imagenet1k_varied_seeds", "Different initializations"),
    ("classification_imagenet1k_varied_architectures", "Different architectures"),
    ("resnet50_imagenet1k_varied_tasks", "Different tasks"),
    ("untrained_resnet18_varied_seeds", "Untrained"),
]
SUBJECTS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
# Chen & Bonner (2025): PC counts (Materials and Methods), rho and partial rho (Results)
ORIGINAL = {
    "resnet18_classification_imagenet1k_varied_seeds": (36596, 0.61, 0.29),
    "classification_imagenet1k_varied_architectures": (149743, 0.65, 0.40),
    "resnet50_imagenet1k_varied_tasks": (43132, 0.68, 0.32),
    "untrained_resnet18_varied_seeds": (9413, 0.84, 0.50),
}

f2 = pd.read_csv(T / "figure2_statistics.csv").set_index("model_set")
f1 = pd.read_csv(OUT / "finding1_statistics.csv")
check = f1[f1.target == "average"].set_index("model_set")
for column in ["n_dimensions", "spearman_rho", "partial_spearman_rho_control_pc",
               "spearman_p_permutation", "partial_spearman_p_permutation"]:
    assert np.allclose(check.loc[f2.index, column], f2[column]), column

def stats(row):
    return {"dims": int(row.n_dimensions), "rho": row.spearman_rho, "p": row.spearman_p_permutation,
            "partial": row.partial_spearman_rho_control_pc, "partial_p": row.partial_spearman_p_permutation}

numbers = {"sets": [], "per_subject": []}
for key, label in SETS:
    tol = f1[(f1.model_set == key) & (f1.target == "original_tolerance")].iloc[0]
    numbers["sets"].append({
        "key": key, "label": label,
        "orig": dict(zip(["dims", "rho", "partial"], ORIGINAL[key])),
        "rep": stats(f2.loc[key]),
        "tol": stats(tol),
    })
    for subject in SUBJECTS:
        row = f1[(f1.model_set == key) & (f1.target == subject)].iloc[0]
        numbers["per_subject"].append({"set": label, "subject": subject, **stats(row)})

numbers["fig3"] = pd.read_csv(OUT / "figure3_summary.csv").iloc[0].to_dict()
fig5 = pd.read_csv(OUT / "figure5_summary.csv")
numbers["fig5"] = [{"label": dict(SETS)[r.model_set], **r._asdict()} for r in fig5.itertuples(index=False)]

rho = pd.read_csv(OUT / "generalization_vs_primary_rho.csv", header=[0, 1], index_col=0)
partial = pd.read_csv(OUT / "generalization_partial.csv", header=[0, 1], index_col=0)
numbers["generalization"] = []
for key, label in SETS:
    entry = {"label": label}
    for mode in ["primary", "tau", "clusters", "ood"]:
        entry[mode] = {"mean": rho.loc[key, ("mean", mode)], "min": rho.loc[key, ("min", mode)],
                       "max": rho.loc[key, ("max", mode)]}
        if mode != "primary":
            entry[mode]["partial"] = partial.loc[key, ("mean", mode)]
    numbers["generalization"].append(entry)
gen = pd.read_csv(T / "generalization_correlations_by_split.csv")
numbers["generalization_range"] = {"min": gen.spearman_rho.min(), "max": gen.spearman_rho.max(), "n": len(gen),
                                   "partial_min": gen.partial_spearman_rho_control_pc.min()}

perf = pd.read_csv(OUT / "generalization_performance.csv")
numbers["performance"] = []
for key, label in SETS:
    rows = perf[perf.model_set == key].set_index("mode")
    numbers["performance"].append({"label": label, **{m: rows.loc[m, "top1pct_brain_similarity"]
                                                       for m in ["primary", "tau", "clusters", "ood"]}})

rel = pd.read_csv(T / "generalization_reliability.csv")
wide = rel.pivot(index="subject", columns="evaluation_set", values="mean_over_splits_of_median_voxel_r")
numbers["reliability"] = [{"subject": s, **wide.loc[s].to_dict()} for s in SUBJECTS]
numbers["reliability_mean"] = wide.mean().to_dict()
numbers["voxels"] = {"min": int(rel.n_voxels.min()), "max": int(rel.n_voxels.max())}

(OUT / "numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
print(json.dumps(numbers["sets"], indent=1, default=float)[:3000])
