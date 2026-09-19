"""
Enterprise EDA and Analytics Orchestrator.

المنسق العام للتحليلات الاستكشافية والإحصائية
ويقوم بتشغيل:

1. Univariate Analysis
2. Bivariate Analysis
3. Multivariate Correlation
4. SOC Business KPIs
5. JSON Report Export
"""

import json
import math
import os
from typing import Any, Dict, List

from pyspark.sql import DataFrame
from pyspark.sql.types import (
    ByteType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
)

import configs.settings as cfg
from src.analysis.bivariate import BivariateAnalyzer
from src.analysis.business_kpis import BusinessKPIAnalyzer
from src.analysis.multivariate import MultivariateAnalyzer
from src.analysis.univariate import UnivariateAnalyzer
from src.common.decorators import time_execution
from src.common.logger import get_logger


logger = get_logger(__name__)


class SparkEDAOrchestrator:
    """
    Enterprise EDA orchestrator for distributed Spark analytics.
    """

    def __init__(self, df: DataFrame):

        self.df = df

        self.univariate = UnivariateAnalyzer()
        self.bivariate = BivariateAnalyzer()
        self.multivariate = MultivariateAnalyzer()
        self.business_kpis = BusinessKPIAnalyzer()

    # ============================================================
    # Numeric Schema Detection
    # ============================================================

    def _get_numeric_columns(
        self,
        candidates: List[str],
    ) -> List[str]:
        """
        Return only columns that:
        1. Exist in the DataFrame.
        2. Have an actual numeric Spark data type.
        """

        numeric_types = (
            ByteType,
            ShortType,
            IntegerType,
            LongType,
            FloatType,
            DoubleType,
            DecimalType,
        )

        numeric_columns = []

        schema_map = {
            field.name: field.dataType
            for field in self.df.schema.fields
        }

        for column in candidates:

            if column not in schema_map:
                continue

            if isinstance(
                schema_map[column],
                numeric_types,
            ):
                numeric_columns.append(column)

            else:

                logger.warning(
                    "⚠️ Column '%s' exists but is not numeric "
                    "(type=%s). Skipped.",
                    column,
                    schema_map[column],
                )

        return numeric_columns

    # ============================================================
    # Available Columns
    # ============================================================

    def _get_available_columns(
        self,
        candidates: List[str],
    ) -> List[str]:

        available = [
            column
            for column in candidates
            if column in self.df.columns
        ]

        missing = [
            column
            for column in candidates
            if column not in self.df.columns
        ]

        if missing:

            logger.warning(
                "⚠️ Missing EDA columns skipped: %s",
                missing,
            )

        return available

    # ============================================================
    # Main Pipeline
    # ============================================================

    @time_execution
    def run_full_analysis(self) -> Dict[str, Any]:

        logger.info("=" * 70)

        logger.info(
            "🚀 Starting Enterprise EDA & SOC Analytics Engine..."
        )

        logger.info("=" * 70)

        # ========================================================
        # Validate DataFrame
        # ========================================================

        if self.df is None:

            logger.error(
                "❌ Spark DataFrame is None."
            )

            return {}

        if not self.df.columns:

            logger.error(
                "❌ Spark DataFrame contains no columns."
            )

            return {}

        logger.info(
            "📋 DataFrame columns: %d",
            len(self.df.columns),
        )

        logger.info(
            "📋 Available schema: %s",
            self.df.dtypes,
        )

        # ========================================================
        # Candidate Numeric Columns
        # ========================================================

        candidate_numeric_cols = [

            "dst_port",
            "src_port",
            "flow_duration",

            "tot_fwd_pkts",
            "tot_bwd_pkts",

            "totlen_fwd_pkts",
            "totlen_bwd_pkts",

            "fwd_pkt_len_max",
            "fwd_pkt_len_min",
            "fwd_pkt_len_mean",
            "fwd_pkt_len_std",

            "bwd_pkt_len_max",
            "bwd_pkt_len_min",
            "bwd_pkt_len_mean",
            "bwd_pkt_len_std",

            "flow_byts_s",
            "flow_pkts_s",

            "flow_iat_mean",
            "flow_iat_std",
            "flow_iat_max",
            "flow_iat_min",

            "fwd_iat_tot",
            "fwd_iat_mean",
            "fwd_iat_std",
            "fwd_iat_max",
            "fwd_iat_min",

            "bwd_iat_tot",
            "bwd_iat_mean",
            "bwd_iat_std",
            "bwd_iat_max",
            "bwd_iat_min",

            "fwd_psh_flags",
            "bwd_psh_flags",

            "fwd_urg_flags",
            "bwd_urg_flags",

            "fwd_header_len",
            "bwd_header_len",

            "fwd_pkts_s",
            "bwd_pkts_s",

            "pkt_len_min",
            "pkt_len_max",
            "pkt_len_mean",
            "pkt_len_std",
            "pkt_len_var",

            "fin_flag_cnt",
            "syn_flag_cnt",
            "rst_flag_cnt",
            "psh_flag_cnt",
            "ack_flag_cnt",
            "urg_flag_cnt",

            "cwe_flag_count",
            "ece_flag_cnt",

            "down_up_ratio",

            "pkt_size_avg",

            "fwd_seg_size_avg",
            "bwd_seg_size_avg",

            "fwd_byts_b_avg",
            "fwd_pkts_b_avg",
            "fwd_blk_rate_avg",

            "bwd_byts_b_avg",
            "bwd_pkts_b_avg",
            "bwd_blk_rate_avg",

            "subflow_fwd_pkts",
            "subflow_fwd_byts",
            "subflow_bwd_pkts",
            "subflow_bwd_byts",

            "init_fwd_win_byts",
            "init_bwd_win_byts",

            "fwd_act_data_pkts",
            "fwd_seg_size_min",

            "active_mean",
            "active_std",
            "active_max",
            "active_min",

            "idle_mean",
            "idle_std",
            "idle_max",
            "idle_min",
        ]

        # ========================================================
        # Detect REAL numeric columns
        # ========================================================

        numeric_cols = self._get_numeric_columns(
            candidate_numeric_cols
        )

        # ========================================================
        # Categorical Columns
        # ========================================================

        candidate_categorical_cols = [
            "protocol",
            "flags",
            "label",
        ]

        categorical_cols = self._get_available_columns(
            candidate_categorical_cols
        )

        logger.info(
            "📊 Numeric features detected: %d",
            len(numeric_cols),
        )

        logger.info(
            "📊 Categorical features detected: %d",
            len(categorical_cols),
        )

        # ========================================================
        # Target
        # ========================================================

        target_col = "label"

        target_exists = (
            target_col in self.df.columns
        )

        if not target_exists:

            logger.warning(
                "⚠️ Target column '%s' does not exist.",
                target_col,
            )

        # ========================================================
        # 1. UNIVARIATE
        # ========================================================

        logger.info(
            "📊 [EDA] Starting Univariate Analysis..."
        )

        try:

            univariate_numeric = (
                self.univariate.analyze_numerical(
                    self.df,
                    numeric_cols,
                )
            )

        except Exception as error:

            logger.exception(
                "❌ Univariate numerical analysis failed: %s",
                error,
            )

            univariate_numeric = {}

        try:

            univariate_categorical = (
                self.univariate.analyze_categorical(
                    self.df,
                    categorical_cols,
                )
            )

        except Exception as error:

            logger.exception(
                "❌ Univariate categorical analysis failed: %s",
                error,
            )

            univariate_categorical = {}

        # ========================================================
        # 2. BIVARIATE
        # ========================================================

        logger.info(
            "📊 [EDA] Starting Bivariate Analysis..."
        )

        if target_exists:

            try:

                bivariate_targets = (
                    self.bivariate.analyze_features_by_target(
                        self.df,
                        numeric_cols,
                        target_col=target_col,
                    )
                )

            except Exception as error:

                logger.exception(
                    "❌ Bivariate target analysis failed: %s",
                    error,
                )

                bivariate_targets = {}

        else:

            bivariate_targets = {}

        # ========================================================
        # Protocol vs Target
        # ========================================================

        if (
            "protocol" in self.df.columns
            and target_exists
        ):

            try:

                protocol_distribution = (
                    self.bivariate
                    .analyze_protocol_distribution_by_target(
                        self.df,
                        protocol_col="protocol",
                        target_col=target_col,
                    )
                )

            except Exception as error:

                logger.exception(
                    "❌ Protocol distribution analysis failed: %s",
                    error,
                )

                protocol_distribution = []

        else:

            protocol_distribution = []

        # ========================================================
        # 3. MULTIVARIATE
        # ========================================================

        logger.info(
            "📊 [EDA] Starting Multivariate Correlation Analysis..."
        )

        if len(numeric_cols) >= 2:

            try:

                correlation_results = (
                    self.multivariate.compute_correlation_matrix(
                        self.df,
                        numeric_cols,
                    )
                )

            except Exception as error:

                logger.exception(
                    "❌ Correlation analysis failed: %s",
                    error,
                )

                correlation_results = {}

        else:

            logger.warning(
                "⚠️ Not enough numeric features "
                "for correlation analysis."
            )

            correlation_results = {}

        # ========================================================
        # 4. SOC KPIs
        # ========================================================

        logger.info(
            "📊 [EDA] Calculating SOC operational KPIs..."
        )

        if target_exists:

            try:

                soc_kpis = (
                    self.business_kpis.compute_soc_metrics(
                        self.df,
                        target_col=target_col,
                    )
                )

            except Exception as error:

                logger.exception(
                    "❌ SOC KPI calculation failed: %s",
                    error,
                )

                soc_kpis = {}

        else:

            soc_kpis = {}

        # ========================================================
        # 5. Build Report
        # ========================================================

        eda_summary_report: Dict[str, Any] = {

            "metadata": {
                "total_columns": len(
                    self.df.columns
                ),

                "columns": self.df.columns,

                "numeric_features": numeric_cols,

                "numeric_feature_count": len(
                    numeric_cols
                ),

                "categorical_features": categorical_cols,

                "categorical_feature_count": len(
                    categorical_cols
                ),

                "target_column": (
                    target_col
                    if target_exists
                    else None
                ),
            },

            "univariate_numerical_analysis":
                univariate_numeric,

            "univariate_categorical_analysis":
                univariate_categorical,

            "bivariate_target_comparison":
                bivariate_targets,

            "protocol_attack_distribution":
                protocol_distribution,

            "multivariate_correlation":
                correlation_results,

            "soc_operational_kpis":
                soc_kpis,
        }

        # ========================================================
        # 6. Save JSON
        # ========================================================

        try:

            os.makedirs(
                cfg.REPORTS_DIR,
                exist_ok=True,
            )

            report_file_path = os.path.join(
                cfg.REPORTS_DIR,
                "eda_summary_report.json",
            )

            with open(
                report_file_path,
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    eda_summary_report,
                    file,
                    indent=4,
                    ensure_ascii=False,
                    default=self._json_serializer,
                )

            logger.info("=" * 70)

            logger.info(
                "✅ EDA report saved successfully:"
            )

            logger.info(
                "📄 %s",
                report_file_path,
            )

            logger.info("=" * 70)

        except Exception as error:

            logger.exception(
                "❌ Failed to save EDA report: %s",
                error,
            )

        # ========================================================
        # 7. Finish
        # ========================================================

        logger.info(
            "🎯 Enterprise EDA analysis completed successfully."
        )

        return eda_summary_report

    # ============================================================
    # JSON Serializer
    # ============================================================

    @staticmethod
    def _json_serializer(value: Any):

        if value is None:
            return None

        if isinstance(
            value,
            (
                int,
                float,
                str,
                bool,
            ),
        ):

            if isinstance(value, float):
                if (
                    math.isnan(value)
                    or math.isinf(value)
                ):
                    return 0.0

            return value

        try:

            if hasattr(value, "item"):
                return value.item()

        except Exception:
            pass

        return str(value)