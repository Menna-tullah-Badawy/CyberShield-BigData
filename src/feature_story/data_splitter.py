"""
Stratified Distributed Dataset Splitter.
تقسيم البيانات طبقياً (Train / Validation / Test) مع الحفاظ على نسب التهديدات.
"""

from typing import Tuple
from pyspark.sql import DataFrame
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
        """تقسيم متوازن طبقياً للحفاظ على نسبة التوزيع بين الفئات."""
        logger.info(f"⚡ بدء تقسيم البيانات بنسب ({train_ratio*100}% Train, {val_ratio*100}% Val, {test_ratio*100}% Test)...")

        fractions = {
            row[label_col]: (train_ratio, val_ratio, test_ratio)
            for row in df.select(label_col).distinct().collect()
        }

        # عزل البيانات وتقسيم كل فئة بنسب دقيقة
        train_df = None
        val_df = None
        test_df = None

        for label_val in fractions.keys():
            subset = df.filter(df[label_col] == label_val)
            train_sub, val_sub, test_sub = subset.randomSplit(
                [train_ratio, val_ratio, test_ratio],
                seed=self.seed
            )

            if train_df is None:
                train_df = train_sub
                val_df = val_sub
                test_df = test_sub
            else:
                train_df = train_df.union(train_sub)
                val_df = val_df.union(val_sub)
                test_df = test_df.union(test_sub)

        logger.info(f"✅ اكتمل التقسيم: [Train: {train_df.count()}, Val: {val_df.count()}, Test: {test_df.count()}]")
        return train_df, val_df, test_df