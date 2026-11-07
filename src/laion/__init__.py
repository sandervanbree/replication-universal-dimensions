from src.laion.pca_handoff import PCAHandoff, load_pca_scores

__all__ = [
    "LaionStimulusSet",
    "PCAHandoff",
    "load_features",
    "load_laion_stimulus_set",
    "load_node",
    "load_pca_scores",
]


def __getattr__(name: str):
    if name in {"load_features", "load_node"}:
        from src.laion import features
        return getattr(features, name)
    if name in {"LaionStimulusSet", "load_laion_stimulus_set"}:
        from src.laion import stimulus_set
        return getattr(stimulus_set, name)
    raise AttributeError(name)
