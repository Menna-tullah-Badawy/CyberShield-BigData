"""
Models and Algorithm Selection Package.
تصدير كلاسات النماذج الأساسية، مصنع الخوارزميات، ومحركات الضبط والاختيار.

Lazy Loading (PEP 562): موديولات Spark لا تُستورد إلا عند طلبها،
حتى تعمل موديلات PyTorch (deep_learning_models) بدون pyspark — كما في Kaggle.
"""

from typing import Any

_LAZY_EXPORTS = {
    # Pure PyTorch (no Spark needed)
    "BiLSTMModel": "src.models.deep_learning_models",
    "BiGRUModel": "src.models.deep_learning_models",
    "MambaNIDS": "src.models.deep_learning_models",
    "CNNBiLSTMModel": "src.models.deep_learning_models",
    "SequenceClassifier": "src.models.deep_learning_models",
    # Spark-based (imported on demand only)
    "BaseCyberModel": "src.models.base_model",
    "CandidateModelFactory": "src.models.candidate_models",
    "SparkHyperparameterTuner": "src.models.hyperparameter_tuner",
    "SparkModelSelector": "src.models.model_selector",
    "DLTrainer": "src.models.dl_trainer",
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
