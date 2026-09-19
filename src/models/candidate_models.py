"""
Distributed Model Candidates (CPU-Optimized).
تعريف نماذج كشف التسلل الموزعة مع ضبط المعاملات للسرعة القصوى على الأجهزة المحلية.
"""

from pyspark.ml.classification import (
    RandomForestClassifier,
    GBTClassifier,
    DecisionTreeClassifier,
    LogisticRegression
)


class CandidateModelFactory:
    """مصنع بناء نماذج التصنيف مع ضبط عمق الأشجار لتقليل استهلاك الذاكرة."""

    @staticmethod
    def get_fast_random_forest(features_col: str = "final_features", label_col: str = "label") -> RandomForestClassifier:
        return RandomForestClassifier(
            featuresCol=features_col,
            labelCol=label_col,
            numTrees=20,
            maxDepth=6,
            maxBins=32,
            seed=42
        )

    @staticmethod
    def get_fast_decision_tree(features_col: str = "final_features", label_col: str = "label") -> DecisionTreeClassifier:
        return DecisionTreeClassifier(
            featuresCol=features_col,
            labelCol=label_col,
            maxDepth=6,
            maxBins=32,
            seed=42
        )

    @staticmethod
    def get_fast_logistic_regression(features_col: str = "final_features", label_col: str = "label") -> LogisticRegression:
        return LogisticRegression(
            featuresCol=features_col,
            labelCol=label_col,
            maxIter=20,
            regParam=0.01
        )