from __future__ import annotations

import importlib
import re
import sys

import pandas as pd
import torch
import torchlens as tl

from .config import REPO_ROOT, RunConfig
from .data import Inventory


def _original_model_tools():
    paths = [
        REPO_ROOT / "orig_codebase" / "src",
    ]
    for path in reversed(paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    module = importlib.import_module("lib.analyses.yield_models")
    return module.load_yielder, module.yield_models


def _history_depth(history, node: str) -> float:
    candidate = node
    index = 0
    while candidate:
        if candidate in history.module_layers and history.module_layers[candidate]:
            layer = history.module_layers[candidate][index]
            return float(layer.rsplit("_", 1)[-1]) / history.num_operations
        match = re.search(r"[\W_]\d+$", candidate)
        if match:
            index = int(re.split(r"[\W_]", candidate)[-1])
        else:
            match = re.search(r"[\W]\w+$", candidate)
            index = 0
        if match is None:
            break
        candidate = candidate[: match.start()]
    raise KeyError(f"Could not map node {node!r} into the TorchLens history")


def run_depths(config: RunConfig, inventory: Inventory, *, overwrite: bool) -> None:
    path = config.tables_dir / "node_depths.csv"
    if path.is_file() and not overwrite:
        return
    load_yielder, yield_models = _original_model_tools()
    requested = set(inventory.models("classification_imagenet1k_varied_architectures"))
    rows = []
    sample = torch.rand(2, 3, 224, 224)
    for model in yield_models(
        load_yielder("classification_imagenet1k_varied_architectures")
    ):
        if model.identifier not in requested:
            continue
        architecture = model.identifier.split(".weights=", 1)[0]
        history = tl.log_forward_pass(model.model, sample, layers_to_save=None)
        for node in inventory.nodes(model.identifier):
            rows.append(
                {
                    "architecture": architecture,
                    "node": node,
                    "node_depth": _history_depth(history, node),
                }
            )
        print(f"depths {architecture}", flush=True)
    frame = pd.DataFrame(rows).drop_duplicates(["architecture", "node"])
    expected = {(item.architecture, item.node) for item in inventory.layers}
    found = set(zip(frame["architecture"], frame["node"], strict=True))
    missing = expected - found
    if missing:
        raise ValueError(
            f"Missing node depths: {len(missing)}; first={sorted(missing)[0]}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["architecture", "node"]).to_csv(path, index=False)
