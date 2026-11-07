from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "config"


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class StageSpec:
    model: str
    identifier: str
    stage: str
    node: str
    hook: str

    @property
    def view_id(self) -> str:
        return f"{self.model}_{self.stage}"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def profile_path(profile: str | Path) -> Path:
    path = Path(profile)
    if path.exists():
        return path
    if path.suffix:
        candidate = CONFIG_ROOT / path
    else:
        candidate = CONFIG_ROOT / "profiles" / f"{profile}.json"
    if not candidate.exists():
        raise ConfigError(f"Profile config not found: {profile}")
    return candidate


def profile_from_args(profile: str | None, debug: str | None) -> str:
    if profile is not None and debug is not None:
        raise ConfigError("Use either --profile or --debug, not both")
    if debug is None:
        return profile or "default"
    if debug == "1":
        return "debug"
    if debug == "2":
        return "full_debug"
    if debug == "0":
        return "default"
    raise ConfigError("--debug must be 0, 1, or 2")


def load_profile(profile: str | Path) -> dict[str, Any]:
    path = profile_path(profile)
    cfg = load_json(path)
    cfg["_profile_path"] = str(path)

    paths = cfg.setdefault("paths", {})
    for key in ("replication_root", "output_root", "results_root"):
        if key not in paths:
            raise ConfigError(f"Missing paths.{key} in {path}")
        paths[key] = str(resolve_project_path(paths[key]))
    for key in ("beta_root", "feature_root", "laion_data_root"):
        if key in paths:
            paths[key] = str(resolve_project_path(paths[key]))

    dataset = cfg.setdefault("dataset", {})
    dataset.setdefault("stimulus_mode", "shared")
    dataset.setdefault("n_stimuli", None)
    dataset.setdefault("random_seed", 42)

    preprocess = cfg.setdefault("preprocess", {})
    preprocess.setdefault("materialize", "pca")
    preprocess.setdefault("pca_dim", None)
    preprocess.setdefault("compress", True)

    alignment = cfg.setdefault("alignment", {})
    alignment.setdefault("n_components", 10)
    alignment.setdefault("ridge", 0.1)
    alignment.setdefault("n_folds", 3)
    alignment.setdefault("n_reps", 1)
    alignment.setdefault("random_seed", dataset["random_seed"])
    alignment.setdefault("backend", "pyrcca")

    return cfg


def load_model_stages(
    models_path: Path | None = None,
    selected_models: list[str] | None = None,
) -> list[StageSpec]:
    models_path = models_path or CONFIG_ROOT / "models.json"
    cfg = load_json(models_path)
    selected = set(selected_models or [])
    stages: list[StageSpec] = []

    for model_cfg in cfg.get("models", []):
        if not model_cfg.get("enabled", True):
            continue
        name = model_cfg["name"]
        if selected and name not in selected:
            continue
        identifier = model_cfg["identifier"]
        stage_cfg = model_cfg.get("stages", {})
        for stage in ("early", "middle", "late"):
            if stage not in stage_cfg:
                raise ConfigError(f"Model {name} is missing stage {stage}")
            spec = stage_cfg[stage]
            stages.append(
                StageSpec(
                    model=name,
                    identifier=identifier,
                    stage=stage,
                    node=spec["node"],
                    hook=spec.get("hook", "global_maxpool"),
                )
            )

    if selected:
        found = {stage.model for stage in stages}
        missing = sorted(selected - found)
        if missing:
            raise ConfigError(f"Selected models not found in config: {', '.join(missing)}")
    return stages


def ensure_dirs(cfg: dict[str, Any]) -> tuple[Path, Path]:
    output_root = Path(cfg["paths"]["output_root"])
    results_root = Path(cfg["paths"]["results_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    return output_root, results_root


def safe_name(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("=", "-")
        .replace(".", "-")
        .replace(" ", "_")
    )
