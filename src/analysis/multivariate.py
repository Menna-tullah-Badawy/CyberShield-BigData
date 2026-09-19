"""
Enterprise Distributed Multivariate Correlation Analysis (High-Performance Engine).

محرك التحليل متعدد المتغيرات فائق السرعة لحساب:
- مصفوفة ارتباط بيرسون (Pearson correlation matrix) في الذاكرة لتجاوز بطء Spark MLLib VectorAssembler.
- أزواج الخصائص عالية الارتباط (Highly correlated feature pairs).
- تحذيرات التعددية الخطية (Multicollinearity warnings).
"""

from typing import Any, Dict, List
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame

from src.common.logger import get_logger

logger = get_logger(__name__)


class MultivariateAnalyzer:
    """
    High-Performance Pearson correlation analyzer.
    """

    def compute_correlation_matrix(
        self,
        df: DataFrame,
        numeric_columns: List[str],
        threshold: float = 0.85,
    ) -> Dict[str, Any]:
        """
        Calculate Pearson correlation matrix and identify multicollinear features.

        Args:
            df: Spark DataFrame.
            numeric_columns: Numeric feature columns.
            threshold: Absolute correlation threshold.

        Returns:
            Correlation matrix and highly correlated feature pairs.
        """
        existing_columns = [
            column
            for column in numeric_columns
            if column in df.columns
        ]

        if len(existing_columns) < 2:
            logger.warning("⚠️ At least two numeric columns are required for multivariate analysis.")
            return {}

        logger.info(
            "📊 [EDA - Multivariate] Calculating high-speed Pearson correlation for %d columns...",
            len(existing_columns),
        )

        try:
            # ----------------------------------------------------
            # 1. سحب عينة سريعة إلى الذاكرة لحساب الارتباط فورياً
            # ----------------------------------------------------
            # تحويل البيانات إلى Pandas يحسب مصفوفة الـ 78 عموداً في أقل من ثانية
            sample_size = min(30000, df.count())
            pdf = (
                df.select(existing_columns)
                .limit(sample_size)
                .toPandas()
            )

            # تنظيف القيم اللانهائية
            pdf.replace([np.inf, -np.inf], np.nan, inplace=True)

            # ----------------------------------------------------
            # 2. حساب مصفوفة الارتباط عبر Pandas / NumPy
            # ----------------------------------------------------
            corr_df = pdf.corr(method="pearson").fillna(0.0)

            matrix_dict: Dict[str, Dict[str, float]] = {}
            high_correlations: List[Dict[str, Any]] = []

            # ----------------------------------------------------
            # 3. استخراج العلاقات القوية وتوليد بنية JSON المطابقة
            # ----------------------------------------------------
            for i, feature_a in enumerate(existing_columns):
                matrix_dict[feature_a] = {}
                for j, feature_b in enumerate(existing_columns):
                    val = corr_df.loc[feature_a, feature_b] if (
                        feature_a in corr_df.index and feature_b in corr_df.columns
                    ) else 0.0
                    
                    try:
                        val = float(val)
                    except (TypeError, ValueError):
                        val = 0.0

                    matrix_dict[feature_a][feature_b] = round(val, 4)

                    # فحص المثلث العلوي للمصفوفة فقط لمنع التكرار
                    if i < j and abs(val) >= threshold:
                        high_correlations.append(
                            {
                                "feature_a": feature_a,
                                "feature_b": feature_b,
                                "correlation": round(val, 4),
                                "absolute_correlation": round(abs(val), 4),
                            }
                        )

            logger.info(
                "✅ Correlation analysis completed in record time. High-correlation pairs: %d",
                len(high_correlations),
            )

            return {
                "columns": existing_columns,
                "matrix": matrix_dict,
                "high_correlations_threshold": threshold,
                "high_correlations_warning": high_correlations,
            }

        except Exception as error:
            logger.exception(
                "❌ Multivariate correlation analysis failed: %s",
                error,
            )
            return {
                "error": str(error),
            }