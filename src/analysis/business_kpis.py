"""
Enterprise Cybersecurity SOC Business KPI Engine.

محرك مؤشرات الأداء التشغيلي لمركز عمليات الأمن SOC.

Calculates:
- Total traffic
- Benign traffic
- Malicious traffic
- Threat density
- Packets per second
- Throughput
- Estimated analyst hours saved
"""

from typing import Any, Dict, Optional

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col,
    count,
    sum as s_sum,
    when,
)

from src.common.logger import get_logger


logger = get_logger(__name__)


class BusinessKPIAnalyzer:
    """
    Calculates operational cybersecurity KPIs.
    """

    # ============================================================
    # Main KPI Engine
    # ============================================================

    def compute_soc_metrics(
        self,
        df: DataFrame,
        target_col: str = "label",
    ) -> Dict[str, Any]:
        """
        Calculate SOC operational metrics.

        The implementation is schema-aware and does not assume
        packet_length or timestamp columns always exist.
        """

        logger.info(
            "🛡️ [SOC KPIs] Starting SOC operational metrics..."
        )

        # --------------------------------------------------------
        # Validate target
        # --------------------------------------------------------

        if target_col not in df.columns:

            logger.warning(
                "⚠️ Target column '%s' does not exist.",
                target_col,
            )

            return {}

        # --------------------------------------------------------
        # Count total rows
        # --------------------------------------------------------

        total_packets = df.count()

        if total_packets == 0:

            logger.warning(
                "⚠️ DataFrame is empty."
            )

            return {
                "total_packets_inspected": 0,
                "benign_traffic_count": 0,
                "malicious_traffic_count": 0,
                "threat_density_percentage": 0.0,
                "network_throughput_mbps": 0.0,
                "traffic_density_pps": 0.0,
                "estimated_soc_triaging_hours_saved": 0.0,
            }

        # --------------------------------------------------------
        # Detect target type / values
        # --------------------------------------------------------

        target_values = (
            df.select(target_col)
            .filter(col(target_col).isNotNull())
            .distinct()
            .limit(100)
            .collect()
        )

        normalized_values = [
            self._normalize_value(row[target_col])
            for row in target_values
        ]

        # --------------------------------------------------------
        # Detect malicious and benign classes
        # --------------------------------------------------------

        malicious_condition = self._build_malicious_condition(
            target_col,
            normalized_values,
        )

        benign_condition = self._build_benign_condition(
            target_col,
            normalized_values,
        )

        # --------------------------------------------------------
        # Detect byte column
        # --------------------------------------------------------

        byte_column = self._find_first_existing_column(
            df,
            [
                "packet_length",
                "totlen_fwd_pkts",
                "totlen_bwd_pkts",
                "flow_bytes",
                "total_bytes",
                "tot_bytes",
            ],
        )

        # --------------------------------------------------------
        # Calculate malicious / benign counts
        # --------------------------------------------------------

        aggregation_expressions = [
            count(
                when(
                    malicious_condition,
                    1,
                )
            ).alias("malicious_count"),
        ]

        if benign_condition is not None:

            aggregation_expressions.append(
                count(
                    when(
                        benign_condition,
                        1,
                    )
                ).alias("benign_count")
            )

        else:

            aggregation_expressions.append(
                count(
                    when(
                        ~malicious_condition,
                        1,
                    )
                ).alias("benign_count")
            )

        # --------------------------------------------------------
        # Optional byte aggregation
        # --------------------------------------------------------

        if byte_column is not None:

            aggregation_expressions.append(
                s_sum(
                    col(byte_column)
                ).alias("total_bytes")
            )

        summary = (
            df.select(
                *aggregation_expressions
            )
            .first()
        )

        malicious_count = int(
            summary["malicious_count"] or 0
        )

        benign_count = int(
            summary["benign_count"] or 0
        )

        total_bytes = 0.0

        if byte_column is not None:

            total_bytes = float(
                summary["total_bytes"] or 0.0
            )

        # --------------------------------------------------------
        # Detect temporal column
        # --------------------------------------------------------

        time_column = self._find_first_existing_column(
            df,
            [
                "timestamp",
                "Timestamp",
                "date",
                "Date",
                "datetime",
                "flow_start",
                "flow_start_time",
            ],
        )

        # --------------------------------------------------------
        # Calculate duration
        # --------------------------------------------------------

        time_span_seconds = self._calculate_time_span(
            df,
            time_column,
        )

        # --------------------------------------------------------
        # Threat density
        # --------------------------------------------------------

        threat_density = (
            malicious_count
            / total_packets
            * 100
        )

        # --------------------------------------------------------
        # Packets per second
        # --------------------------------------------------------

        packets_per_second = (
            total_packets
            / time_span_seconds
        )

        # --------------------------------------------------------
        # Throughput
        # --------------------------------------------------------

        throughput_mbps = (
            total_bytes
            * 8
            / 1_000_000
            / time_span_seconds
        )

        # --------------------------------------------------------
        # Estimated SOC analyst hours saved
        # --------------------------------------------------------

        # Assumption:
        # 3 minutes manual triage per malicious alert.
        analyst_hours_saved = (
            malicious_count
            * 3.0
            / 60.0
        )

        kpis = {
            "total_packets_inspected": total_packets,

            "benign_traffic_count": benign_count,

            "malicious_traffic_count": malicious_count,

            "threat_density_percentage": round(
                threat_density,
                2,
            ),

            "network_throughput_mbps": round(
                throughput_mbps,
                2,
            ),

            "traffic_density_pps": round(
                packets_per_second,
                2,
            ),

            "estimated_soc_triaging_hours_saved": round(
                analyst_hours_saved,
                2,
            ),

            "supporting_columns": {
                "byte_column": byte_column,
                "time_column": time_column,
            },

            "time_span_seconds": round(
                time_span_seconds,
                4,
            ),
        }

        logger.info(
            "✅ SOC KPIs completed | "
            "Malicious=%d | Threat Density=%.2f%% | "
            "Analyst Hours Saved=%.2f",
            malicious_count,
            threat_density,
            analyst_hours_saved,
        )

        return kpis

    # ============================================================
    # Malicious Condition
    # ============================================================

    @staticmethod
    def _build_malicious_condition(
        target_col: str,
        values: list,
    ):
        """
        Build a Spark condition that identifies malicious records.
        """

        numeric_malicious = [
            value
            for value in values
            if value in (1, True)
        ]

        if numeric_malicious:

            return col(target_col) == 1

        # CIC-IDS datasets commonly contain labels such as:
        # BENIGN
        # DoS Hulk
        # DDoS
        # PortScan
        # Bot
        # etc.

        return (
            col(target_col).isNotNull()
            & (
                ~col(target_col)
                .cast("string")
                .isin(
                    "BENIGN",
                    "Benign",
                    "benign",
                    "0",
                )
            )
        )

    # ============================================================
    # Benign Condition
    # ============================================================

    @staticmethod
    def _build_benign_condition(
        target_col: str,
        values: list,
    ):
        """
        Build a Spark condition for benign traffic.
        """

        if any(
            value in (0, False)
            for value in values
        ):

            return col(target_col) == 0

        return (
            col(target_col)
            .cast("string")
            .isin(
                "BENIGN",
                "Benign",
                "benign",
                "0",
            )
        )

    # ============================================================
    # Detect Existing Column
    # ============================================================

    @staticmethod
    def _find_first_existing_column(
        df: DataFrame,
        candidates: list,
    ) -> Optional[str]:
        """
        Return the first candidate column that exists.
        """

        for candidate in candidates:

            if candidate in df.columns:
                return candidate

        return None

    # ============================================================
    # Time Span
    # ============================================================

    @staticmethod
    def _calculate_time_span(
        df: DataFrame,
        time_column: Optional[str],
    ) -> float:
        """
        Calculate capture duration in seconds.

        If no usable timestamp exists, returns 1.0 to prevent
        division by zero and clearly indicate that PPS/throughput
        are not based on an actual capture duration.
        """

        if time_column is None:

            logger.warning(
                "⚠️ No timestamp column found. "
                "Using 1 second fallback for rate calculations."
            )

            return 1.0

        try:

            schema_type = dict(
                df.dtypes
            ).get(time_column)

            # ----------------------------------------------------
            # Timestamp / Date
            # ----------------------------------------------------

            if schema_type in (
                "timestamp",
                "date",
            ):

                from pyspark.sql.functions import (
                    max as s_max,
                    min as s_min,
                )

                row = (
                    df.select(
                        s_min(
                            col(time_column)
                        ).alias("start"),
                        s_max(
                            col(time_column)
                        ).alias("end"),
                    )
                    .first()
                )

                if (
                    row is not None
                    and row["start"] is not None
                    and row["end"] is not None
                ):

                    seconds = (
                        row["end"]
                        - row["start"]
                    ).total_seconds()

                    return (
                        float(seconds)
                        if seconds > 0
                        else 1.0
                    )

            # ----------------------------------------------------
            # Numeric timestamp
            # ----------------------------------------------------

            if schema_type in (
                "int",
                "bigint",
                "double",
                "float",
                "long",
                "short",
            ):

                from pyspark.sql.functions import (
                    max as s_max,
                    min as s_min,
                )

                row = (
                    df.select(
                        s_min(
                            col(time_column)
                        ).alias("start"),
                        s_max(
                            col(time_column)
                        ).alias("end"),
                    )
                    .first()
                )

                if (
                    row is not None
                    and row["start"] is not None
                    and row["end"] is not None
                ):

                    difference = (
                        float(row["end"])
                        - float(row["start"])
                    )

                    return (
                        difference
                        if difference > 0
                        else 1.0
                    )

        except Exception as error:

            logger.warning(
                "⚠️ Failed to calculate time span from '%s': %s",
                time_column,
                error,
            )

        return 1.0

    # ============================================================
    # Value Normalization
    # ============================================================

    @staticmethod
    def _normalize_value(value: Any) -> Any:

        if value is None:
            return None

        try:

            if hasattr(value, "item"):
                return value.item()

        except Exception:
            pass

        return value