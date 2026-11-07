import sys
from loguru import logger
import logging

#  to adjust raj's engineering preference
logger.remove()
logger.add(sys.stderr, level="INFO")

logging.basicConfig(level=logging.INFO)

from copy import deepcopy
from tqdm.auto import tqdm

import torch
import numpy as np
import xarray as xr

from bonner.caching import cache
from bonner.datasets.allen2021_natural_scenes import N_SUBJECTS
from bonner.models.hooks import Hook, GlobalMaxpool, GlobalAveragePool, RandomProjection
from lib.analyses.yield_models import yield_models

from lib.analyses.target_pca_model import target_pca_model
from lib.datasets.allen2021_natural_scenes import get_shared_stimulus_ids, load_activations_from_all
from lib.models import Model
from lib.scorers import RegressionScorer
from lib.datasets import load_dataset, load_stimulus_set


ROIS = [
    f"streams-{stream}" for stream in [
        "lateral",
        "parietal",
        "ventral",
]]
ROIS_v2 = (
    [f"V{i}" for i in range(1,5)]
    +
    [f"streams-{stream}" for stream in [
        "early",
        "lateral",
        "parietal",
        "ventral",
        "midlateral",
        "midparietal",
        "midventral",
    ]]
    + 
    ["V1-4", "faces", "words", "places", "bodies", "general",]
)


def _ev(
    yielder: callable,
    mode: str,
) -> xr.DataArray:
    ev = []
    for model in yield_models(yielder):
        logging.info(model.identifier)
        for node in model.nodes:
            target_model = deepcopy(model)
            target_model.update_nodes([node])
            
            pmode = mode if mode not in ["RP",  "NSD.unshared"] else "NSD"
            pca_model = target_pca_model(target_model, identifier=pmode)
            le = len(pca_model.eigenvalues)
            def _xd(x):
                return xr.DataArray(
                x,
                dims=["neuroid"],
                coords={
                    "pc": ("neuroid", np.arange(le)+1),
                    "node": ("neuroid", [node] * le),
                    "model": ("neuroid", [model.identifier] * le),
                },
            )
            tev = {}
            
            tev["eigenvalue"] = _xd(pca_model.eigenvalues.cpu().numpy())
            tev["percent_variance"] = _xd((pca_model.eigenvalues/pca_model.eigenvalues.sum()).cpu().numpy())
            ev.append(xr.Dataset(tev))
    return xr.concat(ev, dim="neuroid")


def ev(
    yielder: callable,
    yielder_name: str,
    mode: str = "shared",
    basis: str = "activation_pc",
) -> xr.DataArray: 
    assert mode in {"NSD.shared", "NSD", "RP", "NSD.z", "NSD.nonfiltered", "NSD.unshared"}
    
    cacher = cache(f"ev/basis={basis}/mode={mode}/yielder={yielder_name}.nc")
    
    return cacher(_ev)(
        yielder=yielder,
        mode=mode,
    )

