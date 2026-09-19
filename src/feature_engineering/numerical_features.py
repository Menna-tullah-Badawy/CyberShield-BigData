"""
Distributed Numerical Feature Engineering.
معالجة وتحويل الخصائص الرقمية وتطبيق التحويلات الرياضية واللوغاريتمية عبر Apache Spark.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, log1p, when, sqrt
from src.common.logger import get_logger

logger = get_logger("Numerical-Feature-Transformer")


class NumericalFeatureTransformer:
    def __init__(self):
        pass

    def apply_log_transform(self, df: DataFrame, input_col: str, output_col: str = None) -> DataFrame:
        """تطبيق التحويل اللوغاريتمي log1p للتعامل مع التوزيعات الملتوية وحجوم الحزم الشاذة."""
        if input_col not in df.columns:
            logger.warn(f"⚠️ Column '{input_col}' not found. Skipping log transform.")
            return df

        out_col = output_col if output_col else f"{input_col}_log"
        logger.info(f"⚙️ [Feature Engineering] Applying log transform: [{input_col}] -> [{out_col}]...")
        return df.withColumn(
            out_col,
            when(col(input_col) > 0, log1p(col(input_col))).otherwise(0.0)
        )

    def apply_sqrt_transform(self, df: DataFrame, input_col: str, output_col: str = None) -> DataFrame:
        """تطبيق تحويل الجذر التربيعي."""
        if input_col not in df.columns:
            return df
        out_col = output_col if output_col else f"{input_col}_sqrt"
        return df.withColumn(
            out_col,
            when(col(input_col) >= 0, sqrt(col(input_col))).otherwise(0.0)
        )


# أسماء مستعارة لضمان التوافق التام مع كافة الموديولات
NumericalFeatureEngineering = NumericalFeatureTransformer
NumericalFeatures = NumericalFeatureTransformer