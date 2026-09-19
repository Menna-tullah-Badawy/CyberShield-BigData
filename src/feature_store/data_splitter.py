"""
Stratified Distributed Dataset Splitter.
تقسيم البيانات طبقياً (Train / Validation / Test) بكفاءة مع الحفاظ على نسب التهديدات.
"""

from typing import Tuple
from pyspark.sql import DataFrame
from pyspark.sql.functions import col
from src.common.logger import get_logger

logger = get_logger("Data-Splitter")


class SparkDataSplitter:
    def __init__(self, seed: int = 42):
        self.seed = seed

    def stratified_split_by_label(
        self,
        df: DataFrame,
        label_col: str = "label",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """تقسيم متوازن طبقياً للحفاظ على نسبة التوزيع بين مختلف أنواع الهجمات."""
        logger.info(f"⚡ بدء تقسيم البيانات بنسب ({train_ratio*100:.0f}% Train, {val_ratio*100:.0f}% Val, {test_ratio*100:.0f}% Test)...")

        # استخراج الفئات الفريدة
        unique_labels = [row[label_col] for row in df.select(label_col).distinct().collect()]

        train_dfs = []
        val_dfs = []
        test_dfs = []

        for label_val in unique_labels:
            subset = df.filter(col(label_col) == label_val)
            train_sub, val_sub, test_sub = subset.randomSplit(
                [train_ratio, val_ratio, test_ratio],
                seed=self.seed
            )
            train_dfs.append(train_sub)
            val_dfs.append(val_sub)
            test_dfs.append(test_sub)

        # دمج الأجزاء الموزعة
        train_df = train_dfs[0]
        for sub in train_dfs[1:]:
            train_df = train_df.unionByName(sub, allowMissingColumns=True)

        val_df = val_dfs[0]
        for sub in val_dfs[1:]:
            val_df = val_df.unionByName(sub, allowMissingColumns=True)

        test_df = test_dfs[0]
        for sub in test_dfs[1:]:
            test_df = test_df.unionByName(sub, allowMissingColumns=True)

        logger.info("✅ اكتمل التقسيم الطبقي للبيانات بنجاح.")
        return train_df, val_df, test_df