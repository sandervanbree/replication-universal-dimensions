"""Descriptive report numbers computable directly from outputs/tables (no nulls)."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tables', type=Path, default=REPO / 'outputs/tables')
parser.add_argument('--output', type=Path, default=REPO / 'outputs/report')
args = parser.parse_args()
T = args.tables
OUT = args.output
OUT.mkdir(parents=True, exist_ok=True)
SETS = [
    "resnet18_classification_imagenet1k_varied_seeds",
    "classification_imagenet1k_varied_architectures",
    "resnet50_imagenet1k_varied_tasks",
    "untrained_resnet18_varied_seeds",
]
SUBJECTS = ["sub-01", "sub-03", "sub-05", "sub-06", "sub-07"]
pd.set_option("display.width", 220)

# Figure 3: depth-bin summary (mean/SD of rho and partial rho across bins)
f3 = pd.read_csv(T / "figure3_statistics.csv")
fig3 = pd.DataFrame({
    "rho_mean": [f3.spearman_rho.mean()], "rho_sd": [f3.spearman_rho.std()],
    "partial_mean": [f3.partial_spearman_rho_control_pc.mean()],
    "partial_sd": [f3.partial_spearman_rho_control_pc.std()],
    "n_bins": [len(f3)], "max_p": [f3.spearman_p_permutation.max()],
    "max_partial_p": [f3.partial_spearman_p_permutation.max()],
    "min_rho": [f3.spearman_rho.min()], "min_partial": [f3.partial_spearman_rho_control_pc.min()],
})
fig3.to_csv(OUT / "figure3_summary.csv", index=False)
print(fig3.round(3).to_string())

# Figure 5: subject-averaged RSA per network, then set means and networks improved
f5 = pd.read_csv(T / "figure5_rsa.csv")
net = f5.groupby(["model_set", "model", "condition"]).rsa.mean().unstack("condition")
net.columns = ["intact", "top10", "top5"]
rows = []
for s in SETS:
    g = net.loc[s]
    rows.append({
        "model_set": s, "n_networks": len(g),
        "intact": g.intact.mean(), "top10": g.top10.mean(), "top5": g.top5.mean(),
        "delta10": (g.top10 - g.intact).mean(), "delta5": (g.top5 - g.intact).mean(),
        "n_improved_top10": int((g.top10 > g.intact).sum()),
        "n_improved_top5": int((g.top5 > g.intact).sum()),
    })
fig5 = pd.DataFrame(rows)
fig5.to_csv(OUT / "figure5_summary.csv", index=False)
print(fig5.round(3).to_string())

# Figure 5 per subject (template asks per-subject where applicable; descriptive only)
f5s = f5.groupby(["model_set", "subject", "condition"]).rsa.mean().unstack("condition")
f5s.columns = ["intact", "top10", "top5"]
f5s.reset_index().to_csv(OUT / "figure5_per_subject.csv", index=False)

# Generalization vs primary analysis (per participant), effect sizes
dims = pd.read_csv(T / "figure2_dimensions.csv",
                   usecols=["model_set", "universality", *[f"brain_similarity_{s}" for s in SUBJECTS], "brain_similarity"])
primary = []
for s in SETS:
    g = dims[dims.model_set == s]
    for sub in SUBJECTS:
        primary.append({"model_set": s, "subject": sub, "mode": "primary",
                         "spearman_rho": spearmanr(g.universality, g[f"brain_similarity_{sub}"]).statistic})
primary = pd.DataFrame(primary)
gen = pd.read_csv(T / "generalization_correlations_by_split.csv")
gen["mode"] = np.where(gen["mode"].str.startswith("generalization_cluster"), "clusters", gen["mode"].str.replace("generalization_", ""))
gsum = gen.groupby(["model_set", "subject", "mode"]).spearman_rho.mean().reset_index()
allrho = pd.concat([primary, gsum])
wide = allrho.groupby(["model_set", "mode"]).spearman_rho.agg(["mean", "min", "max"]).unstack("mode")
wide.to_csv(OUT / "generalization_vs_primary_rho.csv")
print(wide.round(3).to_string())
gpart = gen.groupby(["model_set", "mode"]).partial_spearman_rho_control_pc.agg(["mean", "min", "max"]).unstack("mode")
gpart.to_csv(OUT / "generalization_partial.csv")
print(gpart.round(3).to_string())
print("all generalization rho > 0:", bool((gen.spearman_rho > 0).all()), "min", gen.spearman_rho.min().round(3))

# Performance change: brain similarity / universality of the most universal dims (top 1% per set)
perf = []
def top_stats(g, mode):
    cut = g.universality.quantile(0.99)
    top = g[g.universality >= cut]
    return {"mode": mode, "median_universality": g.universality.median(),
            "top1pct_universality": top.universality.mean(),
            "top1pct_brain_similarity": top[[f"brain_similarity_{s}" for s in SUBJECTS]].mean(axis=1).mean(),
            "mean_brain_similarity_all": g[[f"brain_similarity_{s}" for s in SUBJECTS]].mean(axis=1).mean()}
for s in SETS:
    perf.append({"model_set": s, **top_stats(dims[dims.model_set == s], "primary")})
reader = pd.read_csv(T / "generalization_dimensions.csv", chunksize=2_000_000,
                     usecols=["model_set", "mode", "universality", *[f"brain_similarity_{s}" for s in SUBJECTS]])
gd = pd.concat(reader)
gd["mode"] = np.where(gd["mode"].str.startswith("generalization_cluster"), "clusters", gd["mode"].str.replace("generalization_", ""))
for (s, mode), g in gd.groupby(["model_set", "mode"]):
    perf.append({"model_set": s, **top_stats(g, mode)})
perf = pd.DataFrame(perf)
perf.to_csv(OUT / "generalization_performance.csv", index=False)
print(perf.round(3).to_string())

# Reliability summary (median voxel split-half r), mean and range across participants
rel = pd.read_csv(T / "generalization_reliability.csv")
r = rel.groupby("evaluation_set").mean_over_splits_of_median_voxel_r.agg(["mean", "min", "max"])
r.to_csv(OUT / "reliability_summary.csv")
print(r.round(3).to_string())
