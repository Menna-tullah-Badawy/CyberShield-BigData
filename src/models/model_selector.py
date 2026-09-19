"""
Enterprise Distributed Model Selection and Champion Evaluation Engine.
محرك تدريب Random Forest الموزع مع رفع العمق والأشجار ودمج ضبط العتبة التلقائي.
"""

from typing import Any, Dict, Tuple, Optional
import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf, col
from pyspark.ml.linalg import DenseVector, VectorUDT
from pyspark.ml.feature import StringIndexer, StringIndexerModel
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel

from src.common.logger import get_logger
from src.common.decorators import measure_performance
from src.evaluation.metrics import CyberEvaluationMetrics
from src.evaluation.threshold_optimizer import ThresholdOptimizer, extract_attack_probability

logger = get_logger("Model-Selector")


@udf(returnType=VectorUDT())
def clean_vector_udf(vec):
    """استبدال أي قيم NaN أو Infinity داخل متجه الخصائص بقيمة 0.0 بشكل آمن."""
    if vec is None:
        return DenseVector([])
    arr = vec.toArray() if hasattr(vec, "toArray") else np.array(vec, dtype=float)
    clean_arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return DenseVector(clean_arr)


class SparkModelSelector:
    def __init__(self, num_trees: int = 100, max_depth: int = 16, seed: int = 42):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.seed = seed
        self.indexer_model: Optional[StringIndexerModel] = None
        self.best_model: Optional[RandomForestClassificationModel] = None
        self.threshold_optimizer = ThresholdOptimizer()
        self.optimal_threshold: float = 0.50

    @measure_performance
    def train_and_select_best(
        self,
        df: DataFrame,
        features_col: str = "final_features",
        label_col: str = "label",
        val_df: Optional[DataFrame] = None,
        test_df: Optional[DataFrame] = None,
        **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """تدريب Random Forest Champion وضبط العتبة على مجموعة التحقق."""
        logger.info("=" * 75)
        logger.info(f"🤖 تدريب Spark Random Forest Champion (Trees={self.num_trees}, Depth={self.max_depth})...")
        logger.info("=" * 75)

        working_label_col = label_col
        train_data = df.withColumn(features_col, clean_vector_udf(col(features_col)))
        val_data = val_df.withColumn(features_col, clean_vector_udf(col(features_col))) if val_df is not None else None
        test_data = test_df.withColumn(features_col, clean_vector_udf(col(features_col))) if test_df is not None else None

        # تحويل التصنيف النصي إن وجد
        if isinstance(df.schema[label_col].dataType, StringType):
            working_label_col = f"{label_col}_idx"
            indexer = StringIndexer(inputCol=label_col, outputCol=working_label_col, handleInvalid="keep")
            self.indexer_model = indexer.fit(train_data)
            train_data = self.indexer_model.transform(train_data)
            if val_data: val_data = self.indexer_model.transform(val_data)
            if test_data: test_data = self.indexer_model.transform(test_data)

        # بناء وتدريب النموذج
        rf = RandomForestClassifier(
            featuresCol=features_col,
            labelCol=working_label_col,
            numTrees=self.num_trees,
            maxDepth=self.max_depth,
            subsamplingRate=0.8,
            featureSubsetStrategy="sqrt",
            seed=self.seed
        )
        self.best_model = rf.fit(train_data)
        logger.info("✅ اكتمال تدريب نموذج Random Forest بنجاح.")

        # ضبط العتبة المثلى على Validation Set
        eval_df = val_data if val_data is not None else (test_data if test_data is not None else train_data)
        val_preds = self.best_model.transform(eval_df)
        opt_res = self.threshold_optimizer.find_optimal_threshold(val_preds, metric_target="f2")
        self.optimal_threshold = opt_res.get("optimal_threshold", 0.50)

        # التقييم النهائي على Test Set بتطبيق العتبة المثلى
        test_target_df = test_data if test_data is not None else eval_df
        test_raw_preds = self.best_model.transform(test_target_df)
        final_test_preds = self.threshold_optimizer.apply_custom_threshold(test_raw_preds, self.optimal_threshold)

        # احتساب المقاييس المتكاملة
        evaluator = CyberEvaluationMetrics(label_col=working_label_col, prediction_col="prediction")
        metrics = evaluator.compute_all_metrics(final_test_preds)
        metrics["optimal_threshold"] = self.optimal_threshold

        return self.best_model, metrics