"""Re-render Figure 5 from outputs/tables/figure5_rsa.csv in the original's layout:
one panel per model set, violins across networks, one dot per network, lines linking
all / top-10 / top-5 versions of the same network. Values are subject-averaged RSA."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tables', type=Path, default=REPO / 'outputs/tables')
parser.add_argument('--output', type=Path, default=REPO / 'outputs/figures')
args = parser.parse_args()
T = args.tables
out = args.output
out.mkdir(parents=True, exist_ok=True)

SETS = [
    ("resnet18_classification_imagenet1k_varied_seeds", "Different initializations"),
    ("classification_imagenet1k_varied_architectures", "Different architectures"),
    ("resnet50_imagenet1k_varied_tasks", "Different tasks"),
    ("untrained_resnet18_varied_seeds", "Untrained"),
]
CONDITIONS = [("all", "All"), ("within_basis_ui.n_top=10", "Top 10"), ("within_basis_ui.n_top=5", "Top 5")]

rsa = pd.read_csv(T / "figure5_rsa.csv")
net = rsa.groupby(["model_set", "model", "condition"]).rsa.mean().unstack("condition")

fig, axes = plt.subplots(1, 4, figsize=(12, 3.6), sharey=True)
for axis, (model_set, title) in zip(axes, SETS):
    values = net.loc[model_set, [c for c, _ in CONDITIONS]]
    parts = axis.violinplot([values[c] for c, _ in CONDITIONS], positions=range(3),
                            showextrema=False, widths=0.8)
    for body in parts["bodies"]:
        body.set_facecolor("#9ecae1")
        body.set_edgecolor("none")
        body.set_alpha(0.6)
    for _, row in values.iterrows():
        axis.plot(range(3), row.to_numpy(), color="#636363", linewidth=0.6, alpha=0.7)
        axis.scatter(range(3), row.to_numpy(), color="#252525", s=8, zorder=3)
    axis.set_xticks(range(3), [label for _, label in CONDITIONS])
    axis.set_title(title, fontsize=11)
    axis.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Representational similarity\n(Spearman ρ, held-out)")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(out / f"figure5_rsa_original_style.{ext}", dpi=300, bbox_inches="tight")
