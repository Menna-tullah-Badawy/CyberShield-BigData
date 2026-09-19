"""
Distributed Hyperparameter Tuning Module.
إعداد شبكة المعلمات الفائقة (ParamGrid) لـ Spark ML Estimators.
"""

from typing import List, Any, Dict, Optional
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, LogisticRegression
from src.common.logger import get_logger

logger = get_logger("Hyperparameter-Tuner")


class SparkHyperparameterTuner:
    def __init__(self):
        pass

    @staticmethod
    def build_param_grid(model_name: str, estimator_obj: Any) -> List[Any]:
        """بناء شبكة المعلمات الفائقة بناءً على نوع النموذج."""
        logger.info(f"⚙️ توليد ParamGrid للنموذج: {model_name}...")
        grid_builder = ParamGridBuilder()

        if "RandomForest" in model_name or isinstance(estimator_obj, RandomForestClassifier):
            return (
                grid_builder.addGrid(estimator_obj.numTrees, [20, 50])
                .addGrid(estimator_obj.maxDepth, [5, 10])
                .build()
            )
        elif "GBT" in model_name or "GradientBoosted" in model_name or isinstance(estimator_obj, GBTClassifier):
            return (
                grid_builder.addGrid(estimator_obj.maxIter, [20, 40])
                .addGrid(estimator_obj.maxDepth, [3, 5])
                .build()
            )
        elif "LogisticRegression" in model_name or isinstance(estimator_obj, LogisticRegression):
            return (
                grid_builder.addGrid(estimator_obj.regParam, [0.01, 0.1])
                .addGrid(estimator_obj.elasticNetParam, [0.0, 0.5])
                .build()
            )
        else:
            return grid_builder.build()