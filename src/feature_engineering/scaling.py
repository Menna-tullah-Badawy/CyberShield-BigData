"""
Distributed Feature Scaling Engine.
معايرة وتطبيع متجهات الخصائص الموزعة باستخدام PySpark StandardScaler.
"""

from typing import Tuple, Optional
from pyspark.sql import DataFrame
from pyspark.ml.feature import StandardScaler, StandardScalerModel
from src.common.logger import get_logger

logger = get_logger("Feature-Scaler")


class FeatureScaler:
    """
    Standardizes feature vectors by removing the mean and scaling to unit variance.
    """

    def __init__(self, with_mean: bool = False, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.model: Optional[StandardScalerModel] = None

    def fit_and_scale(
        self,
        df: DataFrame,
        input_col: str = "assembled_features",
        output_col: str = "final_features"
    ) -> Tuple[DataFrame, StandardScalerModel]:
        """
        تدريب الـ Scaler وتطبيق المعايرة القياسية على جدول البيانات.
        """
        logger.info(f"⚙️ [Feature Scaling] Scaling features from '{input_col}' to '{output_col}' (withMean={self.with_mean}, withStd={self.with_std})...")
        
        scaler = StandardScaler(
            inputCol=input_col,
            outputCol=output_col,
            withMean=self.with_mean,
            withStd=self.with_std
        )
        self.model = scaler.fit(df)
        scaled_df = self.model.transform(df)
        
        logger.info("✅ Feature scaling completed successfully.")
        return scaled_df, self.model

    def scale_features(
        self,
        df: DataFrame,
        input_col: str = "assembled_features",
        output_col: str = "final_features"
    ) -> Tuple[DataFrame, StandardScalerModel]:
        """دالة مساعدة للتوافق مع استدعاءات خط الأنابيب."""
        return self.fit_and_scale(df, input_col=input_col, output_col=output_col)

    def transform(self, df: DataFrame) -> DataFrame:
        """تطبيق النموذج المدرب مسبقاً على بيانات جديدة (Validation / Test Sets)."""
        if self.model is None:
            raise ValueError("❌ Scaler model is not fitted yet. Call 'fit_and_scale' first.")
        return self.model.transform(df)