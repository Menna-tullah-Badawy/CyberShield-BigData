"""
Enterprise Distributed Bivariate Analysis (High-Performance One-Pass Engine).

محرك التحليل الثنائي فائق السرعة لدراسة العلاقات ومقارنة حركة المرور:
- ينفذ كافة التجميعات الإحصائية لجميع الأعمدة الرقمية في استعلام GroupBy مجمع واحد (Single-Pass).
- يلغي التكرار الحلقي (Loops) الذي كان يسبب مئات الـ Spark Stages غير الضرورية.
- يحافظ بنسبة 100% على بنية المخرجات لتوافق التقارير ولوحة التحكم.
"""

from typing import Any, Dict, List

from pyspark.sql import DataFrame
from pyspark.sql.functions import avg, col, count, stddev

from src.common.logger import get_logger

logger = get_logger(__name__)


class BivariateAnalyzer:
    """
    Distributed bivariate analyzer.

    Performs:
    - One-pass batch numerical features vs target class analysis.
    - Protocol vs target distribution analysis.
    """

    # ============================================================
    # Numerical Features vs Target (Batch Execution)
    # ============================================================

    def analyze_features_by_target(
        self,
        df: DataFrame,
        numeric_columns: List[str],
        target_col: str = "label",
    ) -> Dict[str, Any]:
        """
        Compare numerical feature statistics across target classes in a single Spark Job.
        """
        if target_col not in df.columns:
            logger.warning(
                "⚠️ Target column '%s' does not exist.",
                target_col,
            )
            return {}

        results: Dict[str, Any] = {}

        available_columns = [
            column
            for column in numeric_columns
            if column in df.columns and column != target_col
        ]

        if not available_columns:
            logger.warning("⚠️ No valid numerical columns found for bivariate analysis.")
            return results

        logger.info(
            "📊 [EDA - Bivariate] Starting single-pass batch analysis for %d features grouped by '%s'...",
            len(available_columns),
            target_col,
        )

        try:
            # ----------------------------------------------------
            # 1. تجميع كل تعبيرات الـ Aggregation لكافة الأعمدة دفعة واحدة
            # ----------------------------------------------------
            agg_expressions = [count("*").alias("count")]
            for column in available_columns:
                agg_expressions.extend([
                    avg(col(column)).alias(f"{column}__mean"),
                    stddev(col(column)).alias(f"{column}__std"),
                ])

            # ----------------------------------------------------
            # 2. تنفيذ استعلام Spark GroupBy واحد فقط للبيانات كاملة
            # ----------------------------------------------------
            grouped_stats = (
                df.filter(col(target_col).isNotNull())
                .groupBy(col(target_col))
                .agg(*agg_expressions)
                .collect()
            )

            # ----------------------------------------------------
            # 3. تنظيم النتائج في نفس هيكل القاموس المعتمد
            # ----------------------------------------------------
            for column in available_columns:
                feature_results = []
                for row in grouped_stats:
                    target_value = row[target_col]
                    feature_results.append(
                        {
                            "target_class": self._serialize_value(target_value),
                            "mean": self._safe_float(row[f"{column}__mean"]),
                            "std": self._safe_float(row[f"{column}__std"]),
                            "count": int(row["count"] or 0),
                        }
                    )
                results[column] = feature_results

            logger.info(
                "✅ [EDA - Bivariate] Completed feature-target analysis for %d columns in a single pass.",
                len(results),
            )

        except Exception as error:
            logger.exception(
                "❌ Batch bivariate analysis failed: %s",
                error,
            )
            for column in available_columns:
                results[column] = {"error": str(error)}

        return results

    # ============================================================
    # Protocol vs Target
    # ============================================================

    def analyze_protocol_distribution_by_target(
        self,
        df: DataFrame,
        protocol_col: str = "protocol",
        target_col: str = "label",
    ) -> List[Dict[str, Any]]:
        """
        Analyze protocol distribution across target classes.
        """
        if protocol_col not in df.columns:
            logger.warning(
                "⚠️ Protocol column '%s' does not exist.",
                protocol_col,
            )
            return []

        if target_col not in df.columns:
            logger.warning(
                "⚠️ Target column '%s' does not exist.",
                target_col,
            )
            return []

        logger.info(
            "📊 [EDA - Bivariate] Analyzing protocol distribution by target..."
        )

        try:
            grouped_data = (
                df.filter(
                    col(protocol_col).isNotNull()
                    & col(target_col).isNotNull()
                )
                .groupBy(
                    col(protocol_col),
                    col(target_col),
                )
                .agg(
                    count("*").alias("packet_count")
                )
                .orderBy(
                    col(protocol_col),
                    col(target_col),
                )
                .collect()
            )

        except Exception as error:
            logger.exception(
                "❌ Protocol distribution analysis failed: %s",
                error,
            )
            return []

        results = []
        for row in grouped_data:
            results.append(
                {
                    "protocol": self._serialize_value(row[protocol_col]),
                    "label": self._serialize_value(row[target_col]),
                    "packet_count": int(row["packet_count"] or 0),
                }
            )

        logger.info(
            "✅ Protocol distribution completed: %d groups.",
            len(results),
        )

        return results

    # ============================================================
    # Utilities
    # ============================================================

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """
        Convert Spark values into JSON-safe values.
        """
        if value is None:
            return None

        try:
            if hasattr(value, "item"):
                return value.item()
        except Exception:
            pass

        return value

    @staticmethod
    def _safe_float(value: Any) -> float:
        """
        Safely convert value to float.
        """
        if value is None:
            return 0.0

        try:
            return round(float(value), 4)
        except (TypeError, ValueError):
            return 0.0