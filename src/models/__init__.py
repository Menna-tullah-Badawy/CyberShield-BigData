"""
Models Package (torch + tabular track).
Lazy Loading حتى تعمل الموديلات خفيفة بدون تحميل زائد.
"""

from typing import Any

_LAZY_EXPORTS = {
    "BiLSTMModel": "src.models.deep_learning_models",
    "BiGRUModel": "src.models.deep_learning_models",
    "MambaNIDS": "src.models.deep_learning_models",
    "CNNBiLSTMModel": "src.models.deep_learning_models",
    "SequenceClassifier": "src.models.deep_learning_models",
    "CandidateModelFactory": "src.models.candidate_models",
    "DLTrainer": "src.models.dl_trainer",
    "ModelSelector": "src.models.model_selector",
    "HYBRID_WEIGHTS": "src.models.model_selector",
    "NIDSTemporalDataset": "src.models.sequence_dataset",
    "get_temporal_dataloaders": "src.models.sequence_dataset",
}

__all__ = list(_LAZY_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        import importlib

        module = importlib.import_module(_LAZY_EXPORTS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
