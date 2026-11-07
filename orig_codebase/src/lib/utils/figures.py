from pathlib import Path
import re
import pickle
import numpy as np
import xarray as xr
import pandas as pd
import torch
from tqdm.auto import tqdm
import torchlens as tl

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches

sns.set_palette("Set2")

# Set the font to Myriad if available, otherwise fallback to a similar sans-serif font
try:
    # Check if Myriad is available
    myriad_available = any('Myriad' in f.name for f in fm.fontManager.ttflist)
    font_family = 'Myriad Pro' if myriad_available else 'sans-serif'
except:
    font_family = 'sans-serif'

# Configure Seaborn with your requirements
sns.set_theme(
    context="paper",
    style="white",
    font=font_family,
    rc={
        # Resolution settings
        "figure.dpi": 600, 
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        
        # Font settings
        "font.family": font_family,
        "font.sans-serif": ["Myriad Pro", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,        # Default label font size (8pt)
        
        # Axes settings
        "axes.titlesize": 8,   # Title size (8pt)
        "axes.labelsize": 8,   # Axis label size (8pt)
        
        # Tick settings
        "xtick.labelsize": 6,  # X-axis tick labels (6pt)
        "ytick.labelsize": 6,  # Y-axis tick labels (6pt)
        
        # Legend settings
        "legend.fontsize": 8,  # Legend text (8pt)
        
        # Panel part labels settings (for subplots)
        "figure.titlesize": 9,
        "figure.titleweight": "bold",
    }
)

import os
from dotenv import load_dotenv
load_dotenv()

from joblib import Parallel, delayed

from lib.utils.binning import *
from lib.analyses.yield_models import yield_models, load_yielder
from itertools import combinations, permutations
from bonner.computation.metrics import pearson_r
from bonner.computation.decomposition import PCA
from sklearn.linear_model import RidgeCV
from bonner.caching import BONNER_CACHING_HOME, cache
from bonner.datasets.allen2021_natural_scenes._stimuli import load_stimulus_metadata
from scipy.stats import pearsonr, spearmanr, mannwhitneyu, rankdata
from pingouin import partial_corr
import statsmodels.api as sm

from sentence_transformers import SentenceTransformer

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
warnings.filterwarnings('ignore', category=UserWarning, message='Log scale')
warnings.filterwarnings('ignore', category=UserWarning, message='The palette')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='The data contains')

root = BONNER_CACHING_HOME / "summary_results"
# root = Path("/Users/r/Downloads/summary_results")

def _yielder_id(idx=[0, 1, 2, 3], label=False):
    if not label:
        l = [
            "resnet18_classification_imagenet1k_varied_seeds",
            "classification_imagenet1k_varied_architectures",
            "resnet50_imagenet1k_varied_tasks",
            "untrained_resnet18_varied_seeds",
            "varied_visual_diets",
        ]
    else:
        l = [
            "Different initializations",
            "Different architectures",
            "Different tasks",
            "Untrained",
            "Different visual diets",
        ]
    if isinstance(idx, int):
        return l[idx]
    else:
        return [l[i] for i in idx]

def _relative_node_depths():
    path = Path(f"{os.getcwd()}/node_depth_dicts/all.pkl")
    if path.exists():
        return
    yield_id = _yielder_id(idx=1)
    dct = {}
    yielder = load_yielder(yield_id)
    x = torch.rand(2, 3, 224, 224)
    for model in tqdm(yield_models(yielder), desc="model", leave=False):
        architecture = model.identifier.split(".")[0]
        if architecture in dct.keys():
            continue
        dct[architecture] = {}
        model_history = tl.log_forward_pass(model.model, x, layers_to_save=None)
        for node in model.nodes:
            tnode = node
            key_index = 0
            while len(tnode.split(".")) > 0:
                if len(model_history.module_layers[tnode]) > 0:
                    break
                else:
                    try:
                        match = re.search(r'[\W_]\d+$', tnode)
                        key_index = int(re.split(r'[\W_]', tnode)[-1])
                    except:
                        match = re.search(r'[\W]\w+$', node)
                    tnode = tnode[:match.start()]
            dct[architecture][node] = float(model_history.module_layers[tnode][key_index].split("_")[-1])/model_history.num_operations
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(dct, f,)

def _add_node_depths(df):
    _relative_node_depths()
    path = f"{os.getcwd()}/node_depth_dicts/all.pkl"
    with open(path, "rb") as f:
        dct = pickle.load(f,)
    df["node_depth"] = [dct[df.model.iloc[i].split(".")[0]][df.node.iloc[i]] for i in range(len(df))]
    return df

def _load_main_results(yield_idx, basis, mode, roi="general", reg="ridgecv", individual_subj=False, postfix="", sr=False):
    mode_dict = {"NSD": "shared", "NSD.shared": "same", "RP": "RP",}
    if isinstance(yield_idx, int):
        yield_idx = [yield_idx]
    yielders, sets = _yielder_id(yield_idx), _yielder_id(yield_idx, label=True)
    X = []
    null_dfs = []  # List to store null dataframes
    
    for i in range(len(yielders)):
        # Load brain similarity and universality index datasets
        brain_sim = xr.open_dataarray(root / f"brain_similarity/brain_similarity{postfix}.regression={reg}.basis={basis}.roi={roi}.mode={mode_dict[mode]}.yielder={yielders[i]}.nc").to_dataset(name="brain_similarity")
        univ_idx = xr.open_dataset(root / f"universality_index/universality_index{postfix}.basis={basis}.type=within_basis.regression={reg}.mode={mode}.yielder={yielders[i]}.nc").rename({"null.correlation.n_permutations=5000": "null"})
        
        # Process brain_sim to filter out specific model
        brain_sim = brain_sim.sel(neuroid=brain_sim.model!="alexnetgn.weights=ref01")
        
        # Handle individual_subj case
        if individual_subj:
            brain_sim = xr.open_dataarray(root / f"brain_similarity/brain_similarity.regression={reg}.basis={basis}.roi={roi}.mode={mode_dict[mode]}.yielder={yielders[i]}.individual_subj.nc").to_dataset(name="brain_similarity")
            brain_sim = brain_sim.sortby("subject")
        
        # Handle default_list roi case
        if roi in ("default_list"):
            brain_sim = brain_sim.sortby("roi")
        
        # Create dataset list starting with brain_sim and correlation from univ_idx
        x = [
            brain_sim,
            univ_idx.correlation.to_dataset(name="within_basis_ui"),
        ]
        
        # Create null dataframe with proper expansion
        if 'null' in univ_idx:
            # Calculate expansion factor - same as used for within_basis_ui
            expansion_factor = int(len(brain_sim.neuroid) / len(univ_idx.neuroid))
            
            # Get the null values array
            null_values = univ_idx.null.values  # Shape: (permutations, neuroids)
            
            # Transpose to get shape (neuroids, permutations)
            null_transposed = null_values.T
            
            # Expand the null values by repeating rows to match the brain_sim neuroids
            expanded_null = np.repeat(null_transposed, expansion_factor, axis=0)
            
            # Create a dataframe with each permutation as a column
            null_df = pd.DataFrame(
                expanded_null,
                columns=[f'null_perm_{j}' for j in range(null_values.shape[0])]
            )
            
            # Add neuroid, node, model, roi coordinates from brain_sim
            null_df['neuroid'] = brain_sim.neuroid.values
            null_df['node'] = [str(s) for s in brain_sim.node.values]
            null_df['model'] = [str(s) for s in brain_sim.model.values]
            null_df['roi'] = [str(s) for s in brain_sim.roi.values]
            null_df['model_set'] = [sets[i] for _ in range(len(brain_sim.neuroid))]
            
            # Add subject information if individual_subj is True
            if individual_subj and 'subject' in brain_sim.coords:
                null_df['subject'] = [str(s) for s in brain_sim.subject.values]
            
            # Add to list of null dataframes
            null_dfs.append(null_df)
        
        if sr:
            x.append(xr.open_dataarray(root / f"universality_index_without_permute/universality_index{postfix}.basis={basis}.type=cross_basis_v2.regression={reg}.roi={roi}.mode={mode}.yielder={yielders[i]}.nc").to_dataset(name="cross_basis_ui"))
        
        # Apply the expansion to within_basis_ui (keeping this code as in original)
        x[1] = xr.DataArray(
            list(x[1].within_basis_ui.values) * int(len(x[0].neuroid) / len(x[1].neuroid)),
            dims="neuroid",
            coords={
                "neuroid": x[0].neuroid.values, 
                "model": ("neuroid", x[0].model.values),
                "node": ("neuroid", x[0].node.values),
                "roi": ("neuroid", x[0].roi.values),
            }
        ).to_dataset(name="within_basis_ui")
        
        if sr:
            x[2] = xr.DataArray(
                list(x[2].cross_basis_ui.values) * int(len(x[0].neuroid) / len(x[2].neuroid)),
                dims="neuroid",
                coords={
                    "neuroid": x[0].neuroid.values, 
                    "model": ("neuroid", x[0].model.values),
                    "node": ("neuroid", x[0].node.values),
                    "roi": ("neuroid", x[0].roi.values),
                }
            ).to_dataset(name="cross_basis_ui")
        
        x = xr.merge(x)
        df = x.to_dataframe()    
        if basis == "activation_pc":
            df["pc"] = df.index.values
        df["model_set"] = [sets[i] for _ in df.index]
        df = df.reset_index()
        df["model"] = [str(s) for s in df["model"]]
        df["node"] = [str(s) for s in df["node"]]
        df["brain_similarity"] = [float(s) for s in df["brain_similarity"]]
        df["within_basis_ui"] = [float(s) for s in df["within_basis_ui"]]
        if sr:
            df["cross_basis_ui"] = [float(s) for s in df["cross_basis_ui"]]
        
        ev = xr.open_dataset(root / f"ev/basis={basis}/mode={mode}/yielder={yielders[i]}.nc")
        eigenvalue = ev.eigenvalue.values
        eigenvalue = list(eigenvalue) * int(len(df.neuroid) / len(eigenvalue))
        percent_variance = ev.percent_variance.values
        percent_variance = list(percent_variance) * int(len(df.neuroid) / len(percent_variance))
        df["eigenvalue"] = eigenvalue
        df["percent_variance"] = np.array(percent_variance) * 100
        df["log_pc"] = np.log10(df["pc"])
        df["log_eigenvalue"] = np.log10(df["eigenvalue"])
        df["log_percent_variance"] = np.log10(df["percent_variance"])
        X.append(df)
    
    # Concatenate the results
    main_df = pd.concat(X).reset_index()
    null_df = pd.concat(null_dfs).reset_index(drop=True) if null_dfs else None
    
    return main_df, null_df

def _plot_results_in_grid(
    df, x, y=None,
    main_plots=("line",), n_bin=100, 
    group=None, hue=None, group_order=None, hue_order=None, errorbar=None,
    palette=None, markercolor=None, xlabel=None, ylabel=None, hue_label=None, legend=None,
    rowcol=(1, 4), xylim=(-.2, 1), figname = None,
    color="orange", cmap="mako",
    kde_line=False, bw_adjust=1, thresh=1e-2,
    norm=LogNorm(vmin=1e-2, vmax=1e1),
    log_kde=False, kdelim=(1e-2, 1e1),
    log_x=False, log_y=False,
):
    assert "line" in main_plots
    colorbar = ("density" in main_plots and hue is None) or (hue is not None and hue != "model" and hue != "roi")
    tdf = df.copy().dropna().reset_index()
    if group is None:
        group = "group"
        tdf["group"] = ["" for _ in range(len(tdf))]
    if group_order is None:
        if rowcol == (1, 1):
            group_order = ["all"]
        else:
            group_order = tdf[group].unique().tolist()
    if isinstance(group_order, str):
        group_order = [group_order]
            
    if hue is not None:
        tdf["gh"] = [f"{tdf[group][i]}_{tdf[hue][i]}" for i in range(len(tdf))]
        
    plt.close()
    fig, ax = plt.subplots(
        figsize=(rowcol[1]*1.9+0.1, rowcol[0]*1.9-.3,) if colorbar else (rowcol[1]*1.9-.3, rowcol[0]*1.9-.3,),
        nrows=rowcol[0]*2, ncols=rowcol[1]*2+1 if colorbar else rowcol[1]*2,
        sharex="col", sharey="row",
        gridspec_kw={"height_ratios": [1, 3.5]*rowcol[0], "width_ratios": ([3.5, 1]*rowcol[1] + [.5])} if colorbar else {"height_ratios": [1, 3.5]*rowcol[0], "width_ratios": [3.5, 1]*rowcol[1]},
    )

    tdf = df_quantile_bin(
        df=tdf, col=x,
        label_col=f"{x}_bin",
        n_bin=n_bin,
        level="gh" if hue is not None else group,
    )
    if colorbar:
        for i in range(len(ax)):
            fig.delaxes(ax[i][rowcol[1]*2])     
    for i in range(len(group_order)):
        if rowcol == (1, 1):
            ttdf = tdf
        else:
            ttdf = tdf[tdf[group] == group_order[i]]
        if xylim is not None:
            ax[(i//rowcol[1])*2+1][(i%rowcol[1])*2].set(xlim=xylim, ylim=xylim)
        if xlabel and i//rowcol[1] == rowcol[0]-1:
            ax[(i//rowcol[1])*2+1][(i%rowcol[1])*2].set(xlabel=xlabel)
        if ylabel and i%rowcol[1] == 0:
            ax[(i//rowcol[1])*2+1][(i%rowcol[1])*2].set(ylabel=ylabel) 

        if "scatter" in main_plots:
            sns.scatterplot(
                data=ttdf,
                x=x, y=y, color="slategray",
                s=3, alpha=.33, legend=False,
                ax=ax[(i//rowcol[1])*2+1][(i%rowcol[1])*2],
            )
            
        if "density" in main_plots:
            if hue is None:
                sns.kdeplot(
                    data=ttdf,
                    x=x, y=y, color="#cacad8",
                    levels=100,
                    cmap=cmap, 
                    fill=True, 
                    legend=False, alpha=.9,
                    norm=norm,
                    bw_adjust=bw_adjust,
                    ax=ax[(i//rowcol[1])*2+1][(i%rowcol[1])*2],
                    thresh=thresh,
                )
            else:
                for j, h in enumerate(hue_order):
                    sns.kdeplot(
                        data=ttdf[ttdf[hue] == h],
                        x=x, y=y, color=sns.color_palette(palette)[j],
                        levels=3,
                        fill=None, 
                        legend=False, alpha=.9,
                        norm=norm,
                        bw_adjust=bw_adjust,
                        ax=ax[(i//rowcol[1])*2+1][(i%rowcol[1])*2],
                        thresh=thresh,
                    )
            if kde_line:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde([ttdf[x], ttdf[y]])
                kde.set_bandwidth(kde.factor * bw_adjust)
                lin_x = np.linspace(ttdf[x].min(), ttdf[x].max(), n_bin)
                lin_y = np.linspace(ttdf[y].min(), ttdf[y].max(), n_bin)
                grid_x, grid_y = np.meshgrid(lin_x, lin_y)
                density = kde.evaluate(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)
                sns.lineplot(
                    x=lin_x, y=lin_y @ density / density.sum(axis=0),
                    hue=hue, palette=palette if hue is not None else None, hue_order=hue_order,
                    style=group if rowcol == (1, 1) else None, dashes=False,
                    ls="None" if hue is None else None, errorbar=errorbar,
                    marker="o", markersize=2, linewidth=1, markeredgewidth=0,
                    markerfacecolor="limegreen" if markercolor is None and hue is None else markercolor,
                    markeredgecolor="limegreen" if markercolor is None and hue is None else markercolor,
                    ax=ax[(i//rowcol[1])*2+1][(i%rowcol[1])*2],
                )
        
        if log_x:
            ax[(i//rowcol[1])*2+1][(i%rowcol[1])*2].set(xscale="log",)
        if log_y:
            ax[(i//rowcol[1])*2+1][(i%rowcol[1])*2].set(yscale="log",)
                       
        if "diagonal" in main_plots:
            sns.lineplot(x=xylim, y=xylim, color="black", ax=ax[(i//rowcol[1])*2+1][(i%rowcol[1])*2], linewidth=1, linestyle="--")
        
        if markercolor is None:
            if hue is None:
                tcolor = color
            elif hue == "model":
                tcolor = "black"
            else:
                tcolor = None
        else:
            tcolor = markercolor
        
        sns.lineplot(
            data=ttdf,
            x=f"{x}_bin", y=y,
            hue=hue, palette=sns.color_palette([tcolor]) if hue is not None and hue == "model" else palette, hue_order=hue_order,
            style=group if rowcol == (1, 1) else None, dashes=False,
            ls="None" if hue is None else None, 
            errorbar=errorbar,
            marker="o", markersize=2, linewidth=.67, markeredgewidth=0,
            markerfacecolor=tcolor,
            markeredgecolor=tcolor,
            ax=ax[(i//rowcol[1])*2+1][(i%rowcol[1])*2],
        )
        sns.kdeplot(
            data=ttdf,
            x=x, 
            hue=None if hue is None or hue == "model" else hue,
            palette=palette, hue_order=hue_order,
            legend=False, 
            common_norm=False, 
            color="black" if hue is None or hue == "model" else None,
            ax=ax[(i//rowcol[1])*2][(i%rowcol[1])*2],
        )
        sns.kdeplot(
            data=ttdf,
            y=y,
            hue=None if hue is None or hue == "model" else hue,  
            palette=palette, hue_order=hue_order,
            legend=False, 
            common_norm=False, 
            color="black" if hue is None or hue == "model" else None,
            ax=ax[(i//rowcol[1])*2+1][(i%rowcol[1])*2+1],
        )
        if log_kde:
            ax[(i//rowcol[1])*2][(i%rowcol[1])*2].set(yscale="log", ylim=kdelim, yticks=[1e-1, 1e1])
            ax[(i//rowcol[1])*2+1][(i%rowcol[1])*2+1].set(xscale="log", xlim=kdelim, xticks=[1e-1, 1e1])
        if hue is not None:
            ax[(i//rowcol[1])*2+1][(i%rowcol[1])*2].get_legend().remove()
        fig.delaxes(ax[(i//rowcol[1])*2][(i%rowcol[1])*2+1])
        if rowcol != (1, 1):
            ax[(i//rowcol[1])*2][(i%rowcol[1])*2].set_title(group_order[i])
       
    if hue is not None:
        if isinstance(tdf[hue][0], float):
            sm=plt.cm.ScalarMappable(cmap=palette,)
            sm.set_array([])
            fig.colorbar(
                sm, 
                ax=ax[rowcol[0]//2*2+1][rowcol[1]*2],
                orientation='vertical',
                shrink=.8,
                fraction=1,
                label=hue_label if hue_label is not None else hue,
            )
        elif legend == "roi":
            handles, labels = ax[1, 0].get_legend_handles_labels()
            updated_labels = []
            for label in labels:
                if label.split("-")[0] == "streams":
                    updated_labels.append(label.split("-")[1].capitalize())
                else:
                    updated_labels.append(label.capitalize())
            fig.legend(handles, updated_labels, loc='lower center', ncols=len(tdf[hue].unique()), bbox_to_anchor=(.5, -.1), fontsize="medium")
        elif legend == "model_set":
            handles, labels = ax[1, 0].get_legend_handles_labels()
            ax[1, 0].legend(handles=handles[1:1+len(hue_order)], labels=labels[1:1+len(hue_order)], loc="upper left", fontsize="small",)
            ax[1, 0].get_legend().get_frame().set_linewidth(0)
    
    if "density" in main_plots and hue is None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cb = fig.colorbar(
            sm, 
            ax=ax[rowcol[0]//2*2+1][rowcol[1]*2],
            orientation='vertical',
            shrink=.8,
            fraction=1,
        )
        cb.minorticks_off()
        cb.set_ticks([1e-2, 1e1])
        cb.set_label(label="Density")
    
    sns.despine()
    if figname is not None:
        fig.savefig(f"../../figures/{figname}.png", bbox_inches="tight")

def _plot_results_in_simple_grid(
    df, x, y=None,
    main_plots=("line",), n_bin=100, 
    group=None, hue=None, group_order=None, hue_order=None, errorbar=None,
    palette=None, markercolor=None, xlabel=None, ylabel=None, hue_label=None, legend=None,
    rowcol=(1, 4), xylim=(-.2, 1), figname = None,
    color="orange", cmap="mako",
    kde_line=False, bw_adjust=1, thresh=1e-2,
    norm=LogNorm(vmin=1e-2, vmax=1e1),
    log_kde=False, kdelim=(1e-2, 1e1),
    log_x=False, log_y=False,
    size_factors=(1.9, .2)
):
    assert "line" in main_plots
    colorbar = ("density" in main_plots and hue is None) or (hue is not None and hue != "model" and hue != "roi")
    
    tdf = df.copy().dropna().reset_index()
    if group is None:
        group = "group"
        tdf["group"] = ["" for _ in range(len(tdf))]
    if group_order is None:
        if rowcol == (1, 1):
            group_order = ["all"]
        else:
            group_order = tdf[group].unique().tolist()
    if isinstance(group_order, str):
        group_order = [group_order]
            
    if hue is not None:
        tdf["gh"] = [f"{tdf[group][i]}_{tdf[hue][i]}" for i in range(len(tdf))]
        
    plt.close()
    fig, ax = plt.subplots(
        figsize=(rowcol[1]*size_factors[0]+size_factors[1], rowcol[0]*size_factors[0]-size_factors[1],) if colorbar else (rowcol[1]*size_factors[0]-size_factors[1], rowcol[0]*size_factors[0]-size_factors[1],),
        # figsize=(rowcol[1]*4+.5, rowcol[0]*4-.5,) if colorbar else (rowcol[1]*4-.5, rowcol[0]*4-.5,),
        nrows=rowcol[0], ncols=rowcol[1]+1 if colorbar else rowcol[1],
        sharex="col", sharey="row",
        gridspec_kw={"height_ratios": [3.5]*rowcol[0], "width_ratios": ([3.5]*rowcol[1]+[.5])} if colorbar else {"height_ratios": [3.5]*rowcol[0], "width_ratios": [3.5]*rowcol[1]},
    )
    if rowcol[0] == 1:
        axs = [ax]
    else:
        axs = ax

    tdf = df_quantile_bin(
        df=tdf, col=x,
        label_col=f"{x}_bin",
        n_bin=n_bin,
        level="gh" if hue is not None else group,
    )
    if colorbar:
        for i in range(len(axs)):
            fig.delaxes(axs[i][rowcol[1]]) 
    for i in range(len(group_order)):
        if rowcol == (1, 1):
            ttdf = tdf
        else:
            ttdf = tdf[tdf[group] == group_order[i]]
        if xylim is not None:
            axs[i//rowcol[1]][i%rowcol[1]].set(xlim=xylim, ylim=xylim)
        if xlabel and i//rowcol[1] == rowcol[0]-1:
            axs[i//rowcol[1]][i%rowcol[1]].set(xlabel=xlabel)
        if ylabel and i%rowcol[1] == 0:
            axs[i//rowcol[1]][i%rowcol[1]].set(ylabel=ylabel) 

        if "scatter" in main_plots:
            sns.scatterplot(
                data=ttdf,
                x=x, y=y, color="slategray",
                s=3, alpha=.33, legend=False,
                ax=axs[i//rowcol[1]][i%rowcol[1]],
            )
            
        if "density" in main_plots:
            if hue is None:
                sns.kdeplot(
                    data=ttdf,
                    x=x, y=y, color="#cacad8",
                    levels=200 if thresh==1e-3 else 100,
                    cmap=cmap, 
                    # cmap=cmap if (i<4 or len(group_order)!=5) else sns.cubehelix_palette(as_cmap=True, reverse=True), 
                    fill=True, 
                    legend=False, alpha=.9,
                    norm=norm,
                    bw_adjust=bw_adjust,
                    ax=axs[i//rowcol[1]][i%rowcol[1]],
                    thresh=thresh,
                )
            else:
                for j, h in enumerate(hue_order):
                    sns.kdeplot(
                        data=ttdf[ttdf[hue] == h],
                        x=x, y=y, color=sns.color_palette(palette)[j],
                        levels=3,
                        fill=None, 
                        legend=False, alpha=.9,
                        norm=norm,
                        bw_adjust=bw_adjust,
                        ax=axs[i//rowcol[1]][i%rowcol[1]],
                        thresh=thresh,
                    )
            if kde_line:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde([ttdf[x], ttdf[y]])
                kde.set_bandwidth(kde.factor * bw_adjust)
                lin_x = np.linspace(ttdf[x].min(), ttdf[x].max(), n_bin)
                lin_y = np.linspace(ttdf[y].min(), ttdf[y].max(), n_bin)
                grid_x, grid_y = np.meshgrid(lin_x, lin_y)
                density = kde.evaluate(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)
                sns.lineplot(
                    x=lin_x, y=lin_y @ density / density.sum(axis=0),
                    hue=hue, palette=palette if hue is not None else None, hue_order=hue_order,
                    style=group if rowcol == (1, 1) else None, dashes=False,
                    ls="None" if hue is None else None, errorbar=errorbar,
                    marker="o", markersize=2, linewidth=1, markeredgewidth=0,
                    markerfacecolor="limegreen" if markercolor is None and hue is None else markercolor,
                    markeredgecolor="limegreen" if markercolor is None and hue is None else markercolor,
                    ax=axs[i//rowcol[1]][i%rowcol[1]],
                )
        
        if log_x:
            axs[i//rowcol[1]][i%rowcol[1]].set(xscale="log",)
        if log_y:
            axs[i//rowcol[1]][i%rowcol[1]].set(yscale="log",)
                       
        if "diagonal" in main_plots:
            sns.lineplot(x=xylim, y=xylim, color="black", ax=axs[i//rowcol[1]][i%rowcol[1]], linewidth=1, linestyle="--")
        
        if markercolor is None:
            if hue is None:
                tcolor = color
            elif hue == "model":
                tcolor = "black"
            else:
                tcolor = None
        else:
            tcolor = markercolor
        
        sns.lineplot(
            data=ttdf,
            x=f"{x}_bin", y=y,
            hue=hue, palette=sns.color_palette([tcolor]) if hue is not None and hue == "model" else palette, hue_order=hue_order,
            style=group if rowcol == (1, 1) else None, dashes=False,
            ls="None" if hue is None else None, 
            errorbar=errorbar,
            marker="o", markersize=2 if size_factors[0]<2 else 4, linewidth=.67, markeredgewidth=0,
            markerfacecolor=tcolor,
            markeredgecolor=tcolor,
            ax=axs[i//rowcol[1]][i%rowcol[1]],
        )
        if hue is not None or rowcol == (1, 1):
            axs[i//rowcol[1]][i%rowcol[1]].get_legend().remove()
        if (rowcol != (1, 1) or hue is None):
            if i//rowcol[1] == 0:
                axs[i//rowcol[1]][i%rowcol[1]].set_title(group_order[i])
            else:
                axs[i//rowcol[1]][i%rowcol[1]].set_title(group_order[i].split("\n")[0])
       
    if hue is not None:
        if isinstance(tdf[hue][0], float):
            sm=plt.cm.ScalarMappable(cmap=palette,)
            sm.set_array([])
            fig.colorbar(
                sm, ax=axs[rowcol[0]//2][rowcol[1]],
                orientation='vertical',
                shrink=.8,
                fraction=1,
                label=hue_label if hue_label is not None else hue,
            )
        elif legend == "roi":
            handles, labels = axs[0][0].get_legend_handles_labels()
            updated_labels = []
            for label in labels:
                if label.split("-")[0] == "streams":
                    updated_labels.append(label.split("-")[1].capitalize())
                else:
                    updated_labels.append(label.capitalize())
            fig.legend(handles, updated_labels, loc='lower center', ncols=len(tdf[hue].unique()), bbox_to_anchor=(.5, -.4))
        elif legend == "model_set":
            handles, labels = ax[1, 0].get_legend_handles_labels()
            ax[1, 0].legend(handles=handles[1:1+len(hue_order)], labels=labels[1:1+len(hue_order)], loc="upper left", fontsize="small",)
            ax[1, 0].get_legend().get_frame().set_linewidth(0)
    
    if "density" in main_plots and hue is None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cb = fig.colorbar(
            sm, 
            ax=axs[rowcol[0]//2][rowcol[1]],
            orientation='vertical',
            shrink=.8,
            fraction=1,
        )
        cb.minorticks_off()
        cb.set_ticks([kdelim[0], kdelim[1]])
        cb.set_label(label="Density")
    
    if rowcol[0] > 1:
        plt.tight_layout()
    
    sns.despine()
    if figname is not None:
        fig.savefig(f"../../figures/manuscript/{figname}.png", bbox_inches="tight")
  
def _plot_rsas(sort_by_ids, model_sets_idx, individual=None, individual_order=None):
    dfs = []
    for i, yid in enumerate(_yielder_id(model_sets_idx)):
        if individual is None:
            s = [
                xr.open_dataarray(root / f"rsa_score/rsa_score.roi=general.mode=shared.yielder={yid}.sort_by={i}.nc").load().mean(["roi", "subject"]).assign_coords(sort_by=i)
                for i in sort_by_ids.keys()
            ]
            height = 1.8
            aspect = .95
        elif individual == "subject":
            s = [
                xr.open_dataarray(root / f"rsa_score/rsa_score.roi=general.mode=shared.yielder={yid}.sort_by={i}.nc").load().mean(["roi"]).assign_coords(sort_by=i)
                for i in sort_by_ids.keys()
            ]
            height = 1.1
            aspect = .95
        elif individual == "roi":
            s = [
                xr.open_dataarray(root / f"rsa_score/rsa_score.roi=default_list.mode=shared.yielder={yid}.sort_by={i}.nc").load().mean(["subject"]).assign_coords(sort_by=i)
                for i in sort_by_ids.keys()
            ]
            height = 1.3
            aspect = .95
        s = xr.concat(s, "sort_by")
        if individual == "roi":
            s["roi"] = [i.split('-')[1] for i in s.roi.values]
        df = s.to_dataframe("r").reset_index()
        df["model_set"] = _yielder_id(idx=model_sets_idx[i], label=True)
        if individual_order is not None:
            assert individual is not None
            df["cat"] = pd.Categorical(df[individual], categories=individual_order, ordered=True)
            df = df.sort_values(["cat", "sort_by"])
        dfs.append(df)
    df = pd.concat(dfs)
    df["n_top"] = df.sort_by.map(sort_by_ids)
        
    
    plt.close()
    if individual is None:
        g = sns.FacetGrid(df, col="model_set", height=height, aspect=aspect,)
    elif individual == "subject":
        g = sns.FacetGrid(df, col="model_set", row="subject", height=height, aspect=aspect,)
    elif individual == "roi":
        g = sns.FacetGrid(df, col="model_set", row="roi", height=height, aspect=aspect,)
    g.map_dataframe(sns.violinplot, x="n_top", y="r", hue="n_top", palette="PuBu_r", inner=None, legend=False)
    g.map_dataframe(sns.lineplot, x="n_top", y="r", hue="model", alpha=0.3, marker="o", markersize=2, markeredgewidth=0, linewidth=1, dashes=False, legend=False, palette=["black"]*df['model'].nunique())
    g.add_legend()
    g.set_axis_labels("", "Representational"
                      "\n"
                      "similarity"
                      "\n"
                      r" (Spearman's $\rho$)")
    if individual is None:
        g.fig.text(0.5, 0.1, "Number of retained dimensions", ha='center', va='center')
    elif individual == "subject":
        g.fig.text(0.5, 0.03, "Number of retained dimensions", ha='center', va='center')
    elif individual == "roi":
        g.fig.text(0.5, 0.03, "Number of retained dimensions", ha='center', va='center')
    
    # Custom title formatting
    if individual is not None:  # Only needed when we have rows
        # Get row and column values
        row_names = g.row_names
        col_names = g.col_names
        
        # Clear all existing titles
        for ax in g.axes.flat:
            ax.set_title("")
        
        # Set custom titles for each subplot
        for row_i, row_val in enumerate(row_names):
            for col_i, col_val in enumerate(col_names):
                col_val = col_val.replace(" ", "\n")
                if row_i == 0:  # First row
                    # Format: "xxx\nyyy" (column name below row name)
                    title = f"{individual}={row_val}\n{col_val}"
                else:  # Other rows
                    # Format: just "xxx" (row name only)
                    title = f"{individual}={row_val}"
                g.axes[row_i, col_i].set_title(title)
    else:
        # If no rows, use default column titles
        g.set_titles(col_template="{col_name}")
    
    g.despine()

def _plot_mapping_comparison(df, rows, row_labels, cmap, n_bin=.005, size_factors=(2, .2)):
    plt.close()
    rowcol = (len(rows), 2)
    
    # Create figure with more overall width
    fig, axs = plt.subplots(
        figsize=(rowcol[1]*size_factors[0]+size_factors[1]+1, rowcol[0]*size_factors[0]-size_factors[1],),
        nrows=rowcol[0], ncols=rowcol[1]+1,
        sharey=False,
        gridspec_kw={
            "height_ratios": [5]*rowcol[0], 
            "width_ratios": ([10]*rowcol[1]+[1]),
        },
    )
    
    norm = LogNorm(vmin=1e-2, vmax=1e1)
    zdf = df.copy()
    for i in range(len(axs)):
        fig.delaxes(axs[i][rowcol[1]]) 
    
    # Set up the figure first
    plt.subplots_adjust(wspace=1, left=0.15)
    
    for i in range(len(rows)):
        xys = [
            ("within_basis_ui", f"within_basis_ui_{rows[i]}"),
            ("brain_similarity", f"brain_similarity_{rows[i]}"),
        ]
        xy_labels = [
            ("Universality (ridge)", f"Universality\n({row_labels[i]})"),
            ("Brain similarity (ridge)", f"Brain similarity\n({row_labels[i]})"),
        ]
        for j in range(2):
            ax = axs[i][j]
            x, y = xys[j]
            x_label, y_label = xy_labels[j]
            try:
                sns.kdeplot(
                    data=df,
                    x=x, y=y, color="#cacad8",
                    levels=100,
                    cmap=cmap[j], 
                    fill=True, 
                    legend=False, alpha=.9,
                    norm=norm,
                    bw_adjust=1,
                    ax=ax,
                    thresh=1e-2,
                )
            except:
                sns.kdeplot(
                    data=df,
                    x=x, y=y, color="#cacad8",
                    levels=60,
                    cmap=cmap[j], 
                    fill=True, 
                    legend=False, alpha=.9,
                    norm=norm,
                    bw_adjust=1,
                    ax=ax,
                    thresh=1e-2,
                )
            tdf = df_quantile_bin(
                df=df, col=x,
                label_col=f"{x}_bin",
                n_bin=n_bin,
            )
            sns.lineplot(
                data=tdf,
                x=f"{x}_bin", y=y,
                dashes=False,
                ls="None", 
                errorbar=None,
                marker="o", markersize=2, linewidth=.67, markeredgewidth=0,
                markerfacecolor="blue",
                markeredgecolor="blue",
                ax=ax,
            )
            ax.set(xlabel=x_label, ylabel=y_label)
            
            # Force square aspect ratio for ALL subplots
            ax.set_box_aspect(1)
    
    # Final adjustments to maintain figure layout
    fig.canvas.draw()
    
    sm = plt.cm.ScalarMappable(cmap=cmap[0], norm=norm)
    cb = fig.colorbar(
        sm, 
        ax=axs[rowcol[0]//2][rowcol[1]],
        orientation='vertical',
        shrink=.8,
        fraction=1,
    )
    cb.minorticks_off()
    cb.set_ticks([1e-2, 1e1])
    cb.set_label(label="Density")
    
    # No tight_layout() - it would override our manual settings
    sns.despine()

def _categorize_pc_log_group(pc):
    if pc <= 10:
        return "[1, 10]"
    elif pc <= 100:
        return "[11, 100]"
    else:
        return "[101, 1000]"

def _plot_distribution(df, group_fn=_categorize_pc_log_group):
    df['PC rank'] = df['pc'].apply(group_fn)

    nrows = 2
    ncols = 4
    row_var = ["within_basis_ui", "brain_similarity"]
    raw_labels = ["Universality", "Brain similarity"]
    plt.close()
    fig, axs = plt.subplots(
        figsize=(1.7*ncols, 1.2*nrows),
        nrows=nrows, ncols=ncols,
    )

    hue_order = df["PC rank"].unique() 
    palette = sns.color_palette("viridis", len(hue_order))
    hue_color_map = dict(zip(hue_order, palette))

    for j, ms in enumerate(_yielder_id([0,1,2,3], label=True)):
        sdf = df[df.model_set == ms].reset_index()
        for i in range(nrows):
            sns.kdeplot(sdf, x=row_var[i], ax=axs[i][j], alpha=.9, fill=False, common_norm=False, palette="viridis", legend=False, hue="PC rank", hue_order=hue_order)
            axs[i][j].set(title=ms)
            axs[i][j].set_yscale('log')
            axs[i][j].set_xlabel(raw_labels[i])
            axs[i][j].set_ylim(bottom=1e-1)
            
    handles = [mpatches.Patch(color=hue_color_map[h], label=str(h)) for h in hue_order]
    fig.legend(
        handles, [str(h) for h in hue_order], loc='lower center',
        title="PC rank",
        bbox_to_anchor=(0.5, -.3), borderaxespad=0, frameon=False
    )

    plt.tight_layout()
    sns.despine()

def _rand_orthonormal(shape0, shape1):
    return torch.linalg.qr(torch.randn(shape0, shape1))[0]

def _get_column_wise_rs(X, Y, n_x, n_y):
    rxx, rxy = [], []
    for i, _ in enumerate(X):
        rxx.append(torch.mean(torch.stack([pearson_r(X[i], X[j]) for j in list(set(np.arange(n_x))-{i})]), dim=0))
        rxy.append(torch.mean(torch.stack([pearson_r(X[i], Y[j]) for j in range(n_y)]), dim=0))
    rxx, rxy = torch.cat(rxx), torch.cat(rxy)
    return rxx, rxy

def _simulation_data(n_stim, n_feature, n_rep):
    torch.manual_seed(11)
    S = _rand_orthonormal(n_stim, n_feature*2)
    S /= S.std(dim=0)
    yS = S[:, :n_feature]
    xS = S[:, n_feature:]
    y = [
        yS + torch.normal(0, .5, (n_stim, n_feature))
        for i in range(n_rep)
    ]
    x = [yS + torch.normal(0, .75, (n_stim, n_feature)) * torch.rand(n_feature)]
    for _ in range(n_rep-1):
        x.append(xS + torch.normal(0, .75, (n_stim, n_feature)) * torch.rand(n_feature))
    return x, y

def _flattend_tril(x: torch.Tensor) -> torch.Tensor:
    lower_tri_indices = torch.tril_indices(x.size(0), x.size(1), offset=-1)
    return x[lower_tri_indices[0], lower_tri_indices[1]]

@cache("r_caption_embeddings/stim={stim}.npy")
def _r_ce(stim, u):
    sm = load_stimulus_metadata().reset_index()
    captions = sm[sm.stimulus_id.isin(u.stimulus_id.values)].captions.tolist()
    l = len(captions)
    captions = np.concatenate([c[:5] for c in captions])
    model = SentenceTransformer("all-mpnet-base-v2")
    ce = model.encode(captions, device="cuda", show_progress_bar=True).reshape(l, 5, -1).mean(axis=1)
    r_ce = _flattend_tril(1-pearson_r(torch.tensor(ce).T, return_diagonal=False))
    return r_ce.cpu().numpy()

def compute_p(true, null, H0):
    assert null.shape[1:] == true.shape
    n_perm = null.shape[0]
    denominator = n_perm + 1
    match H0:
        case 'two_tailed':
            p = (np.sum(np.abs(null) >= np.abs(true), axis=0) + 1) / denominator
        case 'greater':
            p = 1 - ((np.sum(true > null, axis=0) + 1) / denominator)
        case 'less':
            p = 1 - ((np.sum(true < null, axis=0) + 1) / denominator)
    return p

def _calculate_null_corr(null_col, null_df, main_df, y):
    null_map = dict(zip(null_df['neuroid'], null_df[null_col]))
    null_values = [null_map.get(nid) for nid in main_df['neuroid']]
    null_corr, _ = spearmanr(null_values, main_df[y])
    return null_corr

def _calculate_null_partial_corr(null_col, null_df, main_df, y, control):
    null_map = dict(zip(null_df['neuroid'], null_df[null_col]))
    null_values = [null_map.get(nid) for nid in main_df['neuroid']]
    temp_df = main_df.copy()
    temp_df['null_value'] = null_values
    result = partial_corr(
        data=temp_df,
        x='null_value', y=y,
        covar=control,
        method='spearman'
    )
    return result['r'].iloc[0]

def _corr_and_partial_corr_aux(main_df, null_df, control, x, y, print_r, n_jobs=-1):
    main_df = main_df.copy()
    
    # Calculate standard correlation
    corr, _ = spearmanr(main_df[x], main_df[y])
    
    # Get all null permutation columns
    null_cols = [col for col in null_df.columns if col.startswith('null_perm_')]
    
    # Parallelized computation of null correlations using joblib
    batch_size = min(100, len(null_cols))
    null_corrs = Parallel(n_jobs=n_jobs, batch_size=batch_size, timeout=300)(
        delayed(_calculate_null_corr)(null_col, null_df, main_df, y) 
        for null_col in null_cols
    )
    
    # Calculate p-value for standard correlation
    p_value = compute_p(corr, np.array(null_corrs), H0='greater')
    
    if print_r:
        print(f"X-Y spearman r: {corr:.4f}, p-val: {p_value:.4f}")
    
    # Calculate partial correlation using the library function
    partial_result = partial_corr(
        data=main_df,
        x=x, y=y,
        covar=control,
        method='spearman'
    )
    
    # Extract the partial correlation coefficient
    partial_corr_value = partial_result['r'].iloc[0]
    
    # Parallelized computation of null partial correlations using joblib
    null_partial_corrs = Parallel(n_jobs=n_jobs, batch_size=batch_size, timeout=300)(
        delayed(_calculate_null_partial_corr)(null_col, null_df, main_df, y, control)
        for null_col in null_cols
    )
    
    # Calculate p-value for partial correlation
    partial_p_value = compute_p(partial_corr_value, np.array(null_partial_corrs), H0='greater')
    
    if print_r:
        print(f"X-Y spearman partial r: {partial_corr_value:.4f}, p-val: {partial_p_value:.4f}")
    
    # For Spearman correlation, we need to work with ranked data
    # Rank the variables
    main_df[f'{x}_rank'] = rankdata(main_df[x])
    main_df[f'{y}_rank'] = rankdata(main_df[y])
    
    # If control is a list, rank each control variable
    if isinstance(control, list):
        for ctrl in control:
            main_df[f'{ctrl}_rank'] = rankdata(main_df[ctrl])
        control_ranks = [f'{ctrl}_rank' for ctrl in control]
        
        # Create design matrix with ranked control variables
        df_cc = sm.add_constant(main_df[control_ranks])
        
        # Fit models to get residuals on ranked data
        y_cc = sm.OLS(main_df[f'{y}_rank'], df_cc).fit()
        main_df[f'{y}_controlled'] = y_cc.resid
        
        x_cc = sm.OLS(main_df[f'{x}_rank'], df_cc).fit()
        main_df[f'{x}_controlled'] = x_cc.resid
    else:
        # Single control variable case
        main_df[f'{control}_rank'] = rankdata(main_df[control])
        
        # Create design matrix with the ranked control variable
        df_cc = sm.add_constant(main_df[[f'{control}_rank']])
        
        # Fit models to get residuals on ranked data
        y_cc = sm.OLS(main_df[f'{y}_rank'], df_cc).fit()
        main_df[f'{y}_controlled'] = y_cc.resid
        
        x_cc = sm.OLS(main_df[f'{x}_rank'], df_cc).fit()
        main_df[f'{x}_controlled'] = x_cc.resid
        
        # print(spearmanr(main_df[f'{x}_controlled'], main_df[f'{y}_controlled']))
    
    # Return all five values
    return main_df, corr, p_value, partial_corr_value, partial_p_value

@cache("corr_and_partial_corr/{cache_path}.pkl")
def _corr_and_partial_corr(main_df, null_df, control, x="within_basis_ui", y="brain_similarity", group="model_set", print_r=True, n_jobs=-1, cache_path=None):
    if group is None:
        return _corr_and_partial_corr_aux(main_df, null_df, control, x, y, print_r=print_r, n_jobs=n_jobs)
    else:
        rdf, r, p, partial_r, partial_p = [], [], [], [], []
        
        for g, main_group in main_df.groupby(group):
            if print_r:
                print(f"Group: {g}")
            
            # Create filter condition for null_df based on the group
            if isinstance(group, list):
                # For list groups, we need to match on all columns
                filter_condition = True
                for i, col in enumerate(group):
                    filter_condition = filter_condition & (null_df[col] == main_group[col].iloc[0])
            else:
                # For single column group
                filter_condition = null_df[group] == g
                
            null_group = null_df[filter_condition]
            
            if len(main_group) == 0 or len(null_group) == 0:
                print(f"Skipping group {g}: insufficient data")
                continue
                
            # Get all five return values directly from aux function
            t0, t1, t2, t3, t4 = _corr_and_partial_corr_aux(
                main_group, null_group, control, x, y, print_r=print_r, n_jobs=n_jobs
            )
            
            rdf.append(t0)
            r.append(t1)
            p.append(t2)
            partial_r.append(t3)
            partial_p.append(t4)
            
        return pd.concat(rdf).reset_index(drop=True), r, p, partial_r, partial_p