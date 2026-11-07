#!/usr/bin/env python
"""Fit Chen & Bonner PCA bases on LAION unique images and export a handoff.

Each physical model/layer cache is processed once.  PCA is fitted on the
23,560 unique-image activations and applied unchanged to shared and OOD
activations.  Outputs are atomic and resumable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import xarray as xr


MODES = ("unique", "shared", "ood")
EXPECTED_PRESENTATIONS = {"unique": 23_560, "shared": 1_121, "ood": 371}
YIELDER_R18_TRAINED = "resnet18_classification_imagenet1k_varied_seeds"
YIELDER_ARCH = "classification_imagenet1k_varied_architectures"
YIELDER_TASK = "resnet50_imagenet1k_varied_tasks"
YIELDER_R18_UNTRAINED = "untrained_resnet18_varied_seeds"


@dataclass(frozen=True)
class Task:
    model: str
    node: str
    hook: str
    unique_path: str
    shared_path: str
    ood_path: str
    output_path: str
    model_sets: str
    unique_input_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-root",
        type=Path,
        default=Path("data/cache/bonner-models/features"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=72)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--inventory-only", action="store_true")
    return parser.parse_args()


def model_sets(model: str) -> str:
    sets: list[str] = []
    if model.startswith("resnet18.weights=tiny_imagenet.seed="):
        sets.append(YIELDER_R18_TRAINED)
    if model.startswith("resnet18.weights=untrained.seed="):
        sets.append(YIELDER_R18_UNTRAINED)
    if model == "resnet50.weights=imagenet1k_v1" or (
        model.startswith("resnet50.weights=")
        and model != "resnet50.weights=untrained"
        and model != "resnet50.weights=imagenet1k_v1"
    ):
        sets.append(YIELDER_TASK)
    if ".weights=imagenet1k_v1" in model:
        sets.append(YIELDER_ARCH)
    if not sets:
        raise ValueError(f"Could not assign model set: {model}")
    return ";".join(sets)


def node_files(mode_dir: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for node_dir in sorted(mode_dir.glob("node=*")):
        files = sorted(node_dir.glob("hook=*.nc"))
        if len(files) != 1:
            raise RuntimeError(f"Expected one hook file in {node_dir}, found {len(files)}")
        result[node_dir.name.removeprefix("node=")] = files[0]
    return result


def build_tasks(features_root: Path, output_root: Path) -> list[Task]:
    tasks: list[Task] = []
    for model_dir in sorted(path for path in features_root.iterdir() if path.is_dir()):
        by_mode = {
            mode: node_files(model_dir / f"datapipe=laion_fmri.{mode}") for mode in MODES
        }
        node_names = set(by_mode["unique"])
        for mode in ("shared", "ood"):
            if set(by_mode[mode]) != node_names:
                raise RuntimeError(f"Node mismatch for {model_dir.name}: unique versus {mode}")
        for node in sorted(node_names):
            paths = {mode: by_mode[mode][node] for mode in MODES}
            hooks = {path.stem.removeprefix("hook=") for path in paths.values()}
            if len(hooks) != 1:
                raise RuntimeError(f"Hook mismatch for {model_dir.name}, {node}: {hooks}")
            output = output_root / "layers" / model_dir.name / f"node={node}" / "pca.npz"
            tasks.append(
                Task(
                    model=model_dir.name,
                    node=node,
                    hook=hooks.pop(),
                    unique_path=str(paths["unique"]),
                    shared_path=str(paths["shared"]),
                    ood_path=str(paths["ood"]),
                    output_path=str(output),
                    model_sets=model_sets(model_dir.name),
                    unique_input_bytes=paths["unique"].stat().st_size,
                )
            )
    return sorted(tasks, key=lambda task: task.unique_input_bytes, reverse=True)


def load_array(path: str, node: str) -> tuple[np.ndarray, np.ndarray]:
    with xr.open_dataset(path) as dataset:
        values = np.asarray(dataset[node].values)
        stimulus_ids = dataset["stimulus_id"].values.astype(str)
    if values.dtype != np.float32:
        values = values.astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite features in {path}")
    return np.ascontiguousarray(values), stimulus_ids


def validate_existing(path: Path, task: Task) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as artifact:
            rank = int(artifact["rank"])
            input_features = int(artifact["n_input_features"])
            if str(artifact["model"]) != task.model or str(artifact["node"]) != task.node:
                return None
            if artifact["eigenvectors"].shape != (input_features, rank):
                return None
            if artifact["shared_scores"].shape != (EXPECTED_PRESENTATIONS["shared"], rank):
                return None
            if artifact["ood_scores"].shape != (EXPECTED_PRESENTATIONS["ood"], rank):
                return None
        return {
            "status": "existing",
            "rank": rank,
            "input_features": input_features,
            "output_bytes": path.stat().st_size,
            "seconds": 0.0,
        }
    except Exception:
        return None


def init_worker(threads: int) -> None:
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)


def ids_digest(ids: np.ndarray) -> str:
    return hashlib.sha256(("\n".join(ids.tolist()) + "\n").encode("utf-8")).hexdigest()


def process_task(
    task: Task,
    overwrite: bool,
    expected_id_hashes: dict[str, str],
) -> dict[str, object]:
    started = time.monotonic()
    output_path = Path(task.output_path)
    if not overwrite:
        existing = validate_existing(output_path, task)
        if existing is not None:
            return {**asdict(task), **existing}

    unique, unique_ids = load_array(task.unique_path, task.node)
    shared, shared_ids = load_array(task.shared_path, task.node)
    ood, ood_ids = load_array(task.ood_path, task.node)
    arrays = {"unique": unique, "shared": shared, "ood": ood}
    ids_by_mode = {"unique": unique_ids, "shared": shared_ids, "ood": ood_ids}
    if len({array.shape[1] for array in arrays.values()}) != 1:
        raise ValueError(f"Feature-count mismatch: {task.model}, {task.node}")
    for mode in MODES:
        if arrays[mode].shape[0] != EXPECTED_PRESENTATIONS[mode]:
            raise ValueError(
                f"Presentation mismatch for {task.model}, {task.node}, {mode}: "
                f"{arrays[mode].shape[0]} != {EXPECTED_PRESENTATIONS[mode]}"
            )
        if ids_digest(ids_by_mode[mode]) != expected_id_hashes[mode]:
            raise ValueError(f"Stimulus ID/order mismatch: {task.model}, {task.node}, {mode}")

    x = torch.from_numpy(unique)
    with torch.no_grad():
        # This is the RankFilteredPCA rule used in the original analysis.
        rank = int(torch.linalg.matrix_rank(x).item())
        mean = x.mean(dim=0, keepdim=True)
        centered = x - mean
        u, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
        max_abs_cols = torch.argmax(torch.abs(u), dim=0)
        signs = torch.sign(u[max_abs_cols, torch.arange(u.shape[1])])
        signs[signs == 0] = 1
        vh *= signs.unsqueeze(-1)
        eigenvectors = vh[:rank].T.contiguous()
        eigenvalues = (singular_values[:rank] ** 2) / (x.shape[0] - 1)
        shared_scores = (torch.from_numpy(shared) - mean) @ eigenvectors
        ood_scores = (torch.from_numpy(ood) - mean) @ eigenvectors

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    with temp_path.open("wb") as stream:
        np.savez(
            stream,
            format_version=np.asarray("1.0"),
            model=np.asarray(task.model),
            node=np.asarray(task.node),
            hook=np.asarray(task.hook),
            model_sets=np.asarray(task.model_sets),
            fit_split=np.asarray("unique"),
            projected_splits=np.asarray(["shared", "ood"]),
            rank=np.asarray(rank, dtype=np.int64),
            n_fit_presentations=np.asarray(unique.shape[0], dtype=np.int64),
            n_input_features=np.asarray(unique.shape[1], dtype=np.int64),
            mean=mean.numpy(),
            std=np.ones(1, dtype=np.float32),
            source_mean=np.zeros(1, dtype=np.float32),
            source_std=np.ones(1, dtype=np.float32),
            eigenvectors=eigenvectors.numpy(),
            eigenvalues=eigenvalues.numpy(),
            shared_scores=shared_scores.numpy(),
            ood_scores=ood_scores.numpy(),
        )
        stream.flush()
        os.fsync(stream.fileno())
    temp_path.replace(output_path)

    input_features = unique.shape[1]
    del unique, shared, ood, x, centered, u, singular_values, vh
    return {
        **asdict(task),
        "status": "written",
        "rank": rank,
        "input_features": input_features,
        "output_bytes": output_path.stat().st_size,
        "seconds": round(time.monotonic() - started, 3),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    fields = list(rows[0]) if rows else []
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def load_reference_ids(task: Task) -> dict[str, np.ndarray]:
    return {
        mode: load_array(getattr(task, f"{mode}_path"), task.node)[1]
        for mode in MODES
    }


def main() -> int:
    args = parse_args()
    features_root = args.features_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(features_root, output_root)
    if not tasks:
        raise RuntimeError(f"No layer caches found under {features_root}")

    reference_ids = load_reference_ids(tasks[0])
    ids_dir = output_root / "stimulus_ids"
    ids_dir.mkdir(parents=True, exist_ok=True)
    id_manifest_rows = []
    for mode, ids in reference_ids.items():
        path = ids_dir / f"{mode}.txt"
        path.write_text("\n".join(ids.tolist()) + "\n", encoding="utf-8")
        id_manifest_rows.append(
            {
                "split": mode,
                "n_stimuli": len(ids),
                "path": str(path.relative_to(output_root)),
            }
        )
    write_csv(output_root / "stimulus_ids.csv", id_manifest_rows)
    expected_id_hashes = {mode: ids_digest(ids) for mode, ids in reference_ids.items()}

    source_rows = []
    for task in sorted(tasks, key=lambda item: (item.model, item.node)):
        source_rows.append(
            {
                "model": task.model,
                "node": task.node,
                "hook": task.hook,
                "model_sets": task.model_sets,
                "unique_input_bytes": task.unique_input_bytes,
                "unique_path": task.unique_path,
                "shared_path": task.shared_path,
                "ood_path": task.ood_path,
                "output_path": str(Path(task.output_path).relative_to(output_root)),
            }
        )
    write_csv(output_root / "source_inventory.csv", source_rows)

    run = {
        "format_version": "1.0",
        "created_unix": time.time(),
        "features_root": str(features_root),
        "output_root": str(output_root),
        "n_physical_layers": len(tasks),
        "workers": args.workers,
        "threads_per_worker": args.threads_per_worker,
        "logical_cpu_budget": args.workers * args.threads_per_worker,
        "pca_fit_split": "unique",
        "projected_splits": ["shared", "ood"],
        "rank_rule": "torch.linalg.matrix_rank on uncentered unique activations",
        "pca_solver": "torch.linalg.svd(full_matrices=False) on feature-centered unique activations",
        "pca_sign_rule": "largest-absolute left-singular-vector loading positive",
        "dtype": "float32",
    }
    (output_root / "run_config.json").write_text(json.dumps(run, indent=2) + "\n")

    if args.inventory_only:
        print(
            f"Inventory complete: layers={len(tasks)} models={len({task.model for task in tasks})} "
            f"logical_cpu_budget={args.workers * args.threads_per_worker}",
            flush=True,
        )
        return 0

    results: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=init_worker,
        initargs=(args.threads_per_worker,),
    ) as pool:
        futures = {
            pool.submit(process_task, task, args.overwrite, expected_id_hashes): task
            for task in tasks
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    f"[{completed}/{len(tasks)}] {result['status']} model={task.model} "
                    f"node={task.node} features={result['input_features']} rank={result['rank']} "
                    f"seconds={result['seconds']}",
                    flush=True,
                )
            except Exception as error:
                failure = {
                    "model": task.model,
                    "node": task.node,
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                print(
                    f"[{completed}/{len(tasks)}] FAILED model={task.model} node={task.node}: {error!r}",
                    file=sys.stderr,
                    flush=True,
                )
            write_csv(
                output_root / "layer_manifest.partial.csv",
                sorted(results, key=lambda item: (str(item["model"]), str(item["node"]))),
            )
            (output_root / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")

    if failures:
        print(f"Failed layers: {len(failures)}", file=sys.stderr)
        return 1
    final_rows = sorted(results, key=lambda item: (str(item["model"]), str(item["node"])))
    write_csv(output_root / "layer_manifest.csv", final_rows)
    (output_root / "COMPLETE").write_text(
        json.dumps(
            {
                "n_layers": len(final_rows),
                "total_bytes": sum(int(row["output_bytes"]) for row in final_rows),
                "completed_unix": time.time(),
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
