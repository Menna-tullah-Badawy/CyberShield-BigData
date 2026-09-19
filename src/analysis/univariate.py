"""
Enterprise Distributed Univariate Statistical Analysis (High-Performance One-Pass Engine).

محرك التحليل الإحصائي الأحادي فائق السرعة - ينفذ الحسابات لجميع الأعمدة دفعة واحدة.
"""

from typing import Any, Dict, List
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col,
    count,
    kurtosis,
    max as s_max,
    mean,
    min as s_min,
    skewness,
    stddev,
)

from src.common.logger import get_logger

logger = get_logger(__name__)


class UnivariateAnalyzer:
    """
    Distributed univariate statistical analyzer.
    Performs:
    1. One-pass numerical descriptive statistics across all columns.
    2. Categorical frequency analysis.
    """

    # ============================================================
    # Numerical Analysis (One-Pass Batch Execution)
    # ============================================================

    def analyze_numerical(
        self,
        df: DataFrame,
        numeric_columns: List[str],
    ) -> Dict[str, Any]:
        """
        Calculate descriptive statistics for numeric columns in a single Spark Job.
        """
        logger.info(
            "📊 [EDA - Univariate] Starting high-speed batch numerical analysis for %d columns...",
            len(numeric_columns),
        )

        results: Dict[str, Any] = {}

        if not numeric_columns:
            logger.warning("⚠️ No numeric columns available for univariate analysis.")
            return results

        available_columns = [
            column for column in numeric_columns if column in df.columns
        ]

        if not available_columns:
            logger.warning("⚠️ None of the requested numeric columns exist.")
            return results

        try:
            # ----------------------------------------------------
            # 1. Batch Approximate Quantiles (1 Spark Job for ALL columns)
            # ----------------------------------------------------
            logger.info("⚡ Calculating quantiles for all %d columns in parallel...", len(available_columns))
            quantiles_list = df.approxQuantile(
                available_columns,
                [0.25, 0.50, 0.75],
                0.05,  # دقة تقريب مثالية للسرعة
            )
            quantiles_dict = dict(zip(available_columns, quantiles_list))

            # ----------------------------------------------------
            # 2. Batch Aggregation (1 Single Select for ALL columns)
            # ----------------------------------------------------
            logger.info("⚡ Aggregating mean, std, min, max, skewness, kurtosis in a single pass...")
            agg_expressions = []
            for column in available_columns:
                agg_expressions.extend([
                    mean(col(column)).alias(f"{column}__mean"),
                    stddev(col(column)).alias(f"{column}__std"),
                    s_min(col(column)).alias(f"{column}__min"),
                    s_max(col(column)).alias(f"{column}__max"),
                    skewness(col(column)).alias(f"{column}__skewness"),
                    kurtosis(col(column)).alias(f"{column}__kurtosis"),
                ])

            stats_row = df.select(agg_expressions).first()

            # ----------------------------------------------------
            # 3. Format Results
            # ----------------------------------------------------
            for column in available_columns:
                q_vals = quantiles_dict.get(column, [None, None, None])
                results[column] = {
                    "mean": self._safe_float(stats_row[f"{column}__mean"]),
                    "std_dev": self._safe_float(stats_row[f"{column}__std"]),
                    "min": self._safe_float(stats_row[f"{column}__min"]),
                    "q1_25_percent": self._safe_float(q_vals[0] if len(q_vals) > 0 else None),
                    "median_50_percent": self._safe_float(q_vals[1] if len(q_vals) > 1 else None),
                    "q3_75_percent": self._safe_float(q_vals[2] if len(q_vals) > 2 else None),
                    "max": self._safe_float(stats_row[f"{column}__max"]),
                    "skewness": self._safe_float(stats_row[f"{column}__skewness"]),
                    "kurtosis": self._safe_float(stats_row[f"{column}__kurtosis"]),
                }

        except Exception as error:
            logger.exception("❌ Batch numerical analysis failed: %s", error)

        logger.info(
            "✅ Numerical univariate analysis completed in record time for %d columns.",
            len(results),
        )
        return results

    # ============================================================
    # Categorical Analysis
    # ============================================================

    def analyze_categorical(
        self,
        df: DataFrame,
        categorical_columns: List[str],
        top_n: int = 5,
    ) -> Dict[str, Any]:
        """
        Calculate top categorical values and percentages.
        """
        logger.info("📊 [EDA - Univariate] Starting categorical analysis...")
        results: Dict[str, Any] = {}

        if not categorical_columns:
            logger.warning("⚠️ No categorical columns available.")
            return results

        available_columns = [
            column for column in categorical_columns if column in df.columns
        ]

        if not available_columns:
            return results

        total_rows = df.count()
        if total_rows == 0:
            logger.warning("⚠️ DataFrame is empty. Skipping categorical analysis.")
            return results

        for column in available_columns:
            try:
                top_categories = (
                    df.groupBy(col(column))
                    .agg(count("*").alias("frequency"))
                    .orderBy(col("frequency").desc())
                    .limit(top_n)
                    .collect()
                )

                results[column] = []
                for row in top_categories:
                    category_value = row[column]
                    frequency = row["frequency"]
                    results[column].append({
                        "category": str(category_value) if category_value is not None else None,
                        "count": int(frequency or 0),
                        "percentage": round(
                            (float(frequency) / float(total_rows) * 100), 2
                        ) if frequency is not None else 0.0,
                    })

            except Exception as error:
                logger.exception("❌ Categorical analysis failed for '%s': %s", column, error)
                results[column] = {"error": str(error)}

        logger.info("✅ Categorical univariate analysis completed.")
        return results

    # ============================================================
    # Utility
    # ============================================================

    @staticmethod
    def _safe_float(value: Any) -> float:
        if value is None:
            return 0.0
        try:
            return round(float(value), 4)
        except (TypeError, ValueError):
            return 0.0