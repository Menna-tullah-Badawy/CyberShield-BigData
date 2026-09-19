"""
Decision Probability Threshold Optimizer
=========================================

Enterprise threshold optimization for cybersecurity models.

The underlying Spark ML model may be multiclass
(e.g. 9 attack/benign classes).

For binary security evaluation:

    0 = Benign
    1 = Attack

The attack probability is calculated as:

    P(Attack) = 1 - P(Benign)

The threshold is then optimized against F2, F1,
Recall, Precision, or MCC.
"""

from typing import Any, Dict, Tuple, Union, List, Optional

import numpy as np

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, udf, when, lower
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType
)

from src.common.logger import get_logger


# ==============================================================================
# Performance decorator
# ==============================================================================

try:
    from src.common.decorators import (
        measure_performance as time_execution
    )
except ImportError:

    def time_execution(func):
        return func


logger = get_logger(__name__)


# ==============================================================================
# Probability extraction
# ==============================================================================

def build_prob_extractor_udf(index: int):
    """
    Build a Spark UDF that extracts one probability from
    a Spark ML probability vector.

    Parameters
    ----------
    index : int
        Probability index to extract.

    Returns
    -------
    pyspark.sql.functions.udf
        Spark UDF.
    """

    @udf(returnType=DoubleType())
    def _extract(v):

        if v is None:
            return 0.0

        try:

            # Spark DenseVector / SparseVector
            if hasattr(v, "toArray"):

                arr = np.asarray(
                    v.toArray(),
                    dtype=float
                )

            # Python / NumPy vector
            elif isinstance(
                v,
                (list, tuple, np.ndarray)
            ):

                arr = np.asarray(
                    v,
                    dtype=float
                )

            # Scalar
            else:

                return float(v)

            if len(arr) == 0:
                return 0.0

            if index < 0 or index >= len(arr):
                return 0.0

            return float(arr[index])

        except Exception as exc:

            logger.warning(
                "⚠️ Probability extraction failed: "
                f"{exc}"
            )

            return 0.0

    return _extract


# ==============================================================================
# Attack Probability
# ==============================================================================

def extract_attack_probability(
    probability,
    benign_probability_index: int = 0
) -> float:
    """
    Convert a multiclass probability vector into binary
    attack probability.

    Formula:

        P(Attack) = 1 - P(Benign)

    Example:

        probability =
            [0.96, 0.03, 0.01, ...]

        if index 0 = Benign:

            P(Benign) = 0.96
            P(Attack) = 0.04

    This function is kept as a public helper because other
    modules such as model_selector.py may import it.
    """

    if probability is None:
        return 0.0

    try:

        # ------------------------------------------------------------------
        # Spark ML Vector
        # ------------------------------------------------------------------

        if hasattr(probability, "toArray"):

            arr = np.asarray(
                probability.toArray(),
                dtype=float
            )

            if (
                benign_probability_index < 0
                or benign_probability_index >= len(arr)
            ):
                return 0.0

            benign_probability = float(
                arr[benign_probability_index]
            )

            return float(
                np.clip(
                    1.0 - benign_probability,
                    0.0,
                    1.0
                )
            )

        # ------------------------------------------------------------------
        # Python / NumPy array
        # ------------------------------------------------------------------

        if isinstance(
            probability,
            (list, tuple, np.ndarray)
        ):

            arr = np.asarray(
                probability,
                dtype=float
            )

            if len(arr) == 0:
                return 0.0

            if (
                benign_probability_index < 0
                or benign_probability_index >= len(arr)
            ):
                return 0.0

            benign_probability = float(
                arr[benign_probability_index]
            )

            return float(
                np.clip(
                    1.0 - benign_probability,
                    0.0,
                    1.0
                )
            )

        # ------------------------------------------------------------------
        # Scalar probability
        # ------------------------------------------------------------------

        return float(
            np.clip(
                float(probability),
                0.0,
                1.0
            )
        )

    except Exception as exc:

        logger.warning(
            "⚠️ Failed to extract attack probability: "
            f"{exc}"
        )

        return 0.0


# ==============================================================================
# Threshold Optimizer
# ==============================================================================

class ThresholdOptimizer:
    """
    Enterprise threshold optimizer for binary cybersecurity decisions.

    The underlying model may be multiclass.

    Security interpretation:

        Benign = 0
        Attack = 1

    Probability conversion:

        P(Attack) = 1 - P(Benign)
    """

    def __init__(
        self,
        label_col: str = "label",
        probability_col: str = "probability",
        benign_probability_index: int = 0,
        attack_idx: Optional[int] = None,
        **kwargs
    ):

        self.label_col = label_col

        self.probability_col = probability_col

        self.benign_probability_index = (
            benign_probability_index
        )

        # ------------------------------------------------------------------
        # Backward compatibility
        #
        # Older modules may still pass attack_idx.
        # We don't use attack_idx for multiclass security probability,
        # but we keep the attribute so existing code doesn't break.
        # ------------------------------------------------------------------

        self.attack_idx = attack_idx

    # ==========================================================================
    # Label conversion
    # ==========================================================================

    def _to_binary_expr(
        self,
        df: DataFrame,
        column_name: str
    ):
        """
        Convert original labels into:

            0 = Benign
            1 = Attack

        String labels:
            benign / normal / safe -> 0
            everything else -> 1
        """

        data_type = (
            df.schema[column_name].dataType
        )

        if isinstance(
            data_type,
            StringType
        ):

            return when(
                lower(
                    col(column_name)
                ).isin(
                    "benign",
                    "normal",
                    "safe",
                    "0",
                    "0.0"
                ),
                0
            ).otherwise(1)

        return when(
            col(column_name) == 0,
            0
        ).otherwise(1)

    # ==========================================================================
    # Threshold generation
    # ==========================================================================

    def _build_thresholds(
        self,
        threshold_range: Union[
            Tuple[float, float, int],
            List[float],
            np.ndarray
        ]
    ) -> np.ndarray:
        """
        Build threshold search values.
        """

        if (
            isinstance(
                threshold_range,
                tuple
            )
            and len(threshold_range) == 3
        ):

            start, stop, count = (
                threshold_range
            )

            return np.linspace(
                float(start),
                float(stop),
                int(count)
            )

        return np.asarray(
            threshold_range,
            dtype=float
        )

    # ==========================================================================
    # Metric calculation
    # ==========================================================================

    @staticmethod
    def _calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate binary classification metrics.
        """

        # ------------------------------------------------------------------
        # Confusion matrix
        # ------------------------------------------------------------------

        tp = int(
            np.sum(
                (y_true == 1)
                &
                (y_pred == 1)
            )
        )

        fp = int(
            np.sum(
                (y_true == 0)
                &
                (y_pred == 1)
            )
        )

        tn = int(
            np.sum(
                (y_true == 0)
                &
                (y_pred == 0)
            )
        )

        fn = int(
            np.sum(
                (y_true == 1)
                &
                (y_pred == 0)
            )
        )

        # ------------------------------------------------------------------
        # Accuracy
        # ------------------------------------------------------------------

        total = (
            tp + fp + tn + fn
        )

        accuracy = (
            (tp + tn) / total
            if total > 0
            else 0.0
        )

        # ------------------------------------------------------------------
        # Precision
        # ------------------------------------------------------------------

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        # ------------------------------------------------------------------
        # Recall
        # ------------------------------------------------------------------

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        # ------------------------------------------------------------------
        # F1
        # ------------------------------------------------------------------

        f1 = (
            2.0
            * precision
            * recall
            /
            (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # ------------------------------------------------------------------
        # F2
        #
        # F2 gives more importance to Recall.
        # This is appropriate for IDS/security detection.
        # ------------------------------------------------------------------

        f2 = (
            5.0
            * precision
            * recall
            /
            (
                4.0 * precision
                + recall
            )
            if (
                4.0 * precision
                + recall
            ) > 0
            else 0.0
        )

        # ------------------------------------------------------------------
        # MCC
        # ------------------------------------------------------------------

        denominator = np.sqrt(
            float(
                (tp + fp)
                * (tp + fn)
                * (tn + fp)
                * (tn + fn)
            )
        )

        mcc = (
            (
                (tp * tn)
                -
                (fp * fn)
            )
            / denominator
            if denominator > 0
            else 0.0
        )

        # ------------------------------------------------------------------
        # False Positive Rate
        # ------------------------------------------------------------------

        fpr = (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0.0
        )

        # ------------------------------------------------------------------
        # False Negative Rate
        # ------------------------------------------------------------------

        fnr = (
            fn / (fn + tp)
            if (fn + tp) > 0
            else 0.0
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "f2_score": f2,
            "mcc": mcc,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn
        }

    # ==========================================================================
    # Find optimal threshold
    # ==========================================================================

    @time_execution
    def find_optimal_threshold(
        self,
        val_predictions_df: DataFrame,
        metric_target: str = "f2",
        threshold_range: Union[
            Tuple[float, float, int],
            List[float],
            np.ndarray
        ] = (0.01, 0.99, 99)
    ) -> Dict[str, Any]:
        """
        Search for the threshold that maximizes the requested metric.

        Default target:

            F2

        because F2 emphasizes Recall, which is important for
        intrusion detection systems.
        """

        metric_target = (
            metric_target.lower()
        )

        logger.info(
            "🎯 [Threshold Optimizer] "
            "Starting threshold optimization "
            f"for {metric_target.upper()}..."
        )

        # ------------------------------------------------------------------
        # Validate metric
        # ------------------------------------------------------------------

        supported_metrics = {
            "f2",
            "f1",
            "recall",
            "precision",
            "mcc"
        }

        if metric_target not in supported_metrics:

            raise ValueError(
                f"Unsupported metric_target: "
                f"{metric_target}. "
                f"Supported metrics: "
                f"{sorted(supported_metrics)}"
            )

        # ------------------------------------------------------------------
        # Convert labels
        # ------------------------------------------------------------------

        target_expr = (
            self._to_binary_expr(
                val_predictions_df,
                self.label_col
            )
        )

        # ------------------------------------------------------------------
        # Collect required data
        # ------------------------------------------------------------------

        pdf = (
            val_predictions_df
            .select(
                col(
                    self.probability_col
                ),

                target_expr
                .cast(IntegerType())
                .alias(
                    "actual_label"
                )
            )
            .toPandas()
        )

        if pdf.empty:

            raise ValueError(
                "Validation DataFrame is empty."
            )

        # ------------------------------------------------------------------
        # Ground truth
        # ------------------------------------------------------------------

        y_true = (
            pdf[
                "actual_label"
            ]
            .to_numpy(
                dtype=int
            )
        )

        # ------------------------------------------------------------------
        # Extract P(Benign)
        # ------------------------------------------------------------------

        benign_probabilities = []

        for probability in (
            pdf[
                self.probability_col
            ]
        ):

            if hasattr(
                probability,
                "toArray"
            ):

                vector = np.asarray(
                    probability.toArray(),
                    dtype=float
                )

            elif isinstance(
                probability,
                (
                    list,
                    tuple,
                    np.ndarray
                )
            ):

                vector = np.asarray(
                    probability,
                    dtype=float
                )

            else:

                raise TypeError(
                    "Probability column must contain "
                    "a Spark ML Vector, list, tuple, "
                    "or NumPy array."
                )

            if (
                self.benign_probability_index
                < 0
                or
                self.benign_probability_index
                >= len(vector)
            ):

                raise ValueError(
                    "Invalid benign_probability_index="
                    f"{self.benign_probability_index}. "
                    f"Probability vector contains "
                    f"{len(vector)} classes."
                )

            benign_probability = float(
                vector[
                    self.benign_probability_index
                ]
            )

            benign_probabilities.append(
                benign_probability
            )

        benign_probability = np.asarray(
            benign_probabilities,
            dtype=float
        )

        # ------------------------------------------------------------------
        # Convert to Attack Probability
        # ------------------------------------------------------------------

        attack_probability = (
            1.0
            -
            benign_probability
        )

        attack_probability = np.clip(
            attack_probability,
            0.0,
            1.0
        )

        # ------------------------------------------------------------------
        # Threshold candidates
        # ------------------------------------------------------------------

        thresholds = (
            self._build_thresholds(
                threshold_range
            )
        )

        if len(thresholds) == 0:

            raise ValueError(
                "No thresholds supplied."
            )

        # ------------------------------------------------------------------
        # Optimization
        # ------------------------------------------------------------------

        best_threshold = 0.50
        best_score = -np.inf
        best_metrics: Dict[str, Any] = {}

        for threshold in thresholds:

            y_pred = (
                attack_probability
                >= float(threshold)
            ).astype(int)

            metrics = (
                self._calculate_metrics(
                    y_true,
                    y_pred
                )
            )

            if metric_target == "f2":

                current_score = (
                    metrics[
                        "f2_score"
                    ]
                )

            elif metric_target == "f1":

                current_score = (
                    metrics[
                        "f1_score"
                    ]
                )

            elif metric_target == "recall":

                current_score = (
                    metrics[
                        "recall"
                    ]
                )

            elif metric_target == "precision":

                current_score = (
                    metrics[
                        "precision"
                    ]
                )

            elif metric_target == "mcc":

                current_score = (
                    metrics[
                        "mcc"
                    ]
                )

            else:

                current_score = 0.0

            # --------------------------------------------------------------
            # Update best
            # --------------------------------------------------------------

            if current_score > best_score:

                best_score = (
                    float(current_score)
                )

                best_threshold = round(
                    float(threshold),
                    4
                )

                best_metrics = {
                    "optimal_threshold":
                        best_threshold,

                    "target_metric":
                        metric_target,

                    "target_score":
                        round(
                            float(
                                current_score
                            ),
                            4
                        ),

                    "accuracy":
                        round(
                            float(
                                metrics[
                                    "accuracy"
                                ]
                            ),
                            4
                        ),

                    "precision":
                        round(
                            float(
                                metrics[
                                    "precision"
                                ]
                            ),
                            4
                        ),

                    "recall":
                        round(
                            float(
                                metrics[
                                    "recall"
                                ]
                            ),
                            4
                        ),

                    "f1_score":
                        round(
                            float(
                                metrics[
                                    "f1_score"
                                ]
                            ),
                            4
                        ),

                    "f2_score":
                        round(
                            float(
                                metrics[
                                    "f2_score"
                                ]
                            ),
                            4
                        ),

                    "mcc":
                        round(
                            float(
                                metrics[
                                    "mcc"
                                ]
                            ),
                            4
                        ),

                    "false_positive_rate":
                        round(
                            float(
                                metrics[
                                    "false_positive_rate"
                                ]
                            ),
                            4
                        ),

                    "false_negative_rate":
                        round(
                            float(
                                metrics[
                                    "false_negative_rate"
                                ]
                            ),
                            4
                        ),

                    "true_positives":
                        metrics[
                            "true_positives"
                        ],

                    "false_positives":
                        metrics[
                            "false_positives"
                        ],

                    "true_negatives":
                        metrics[
                            "true_negatives"
                        ],

                    "false_negatives":
                        metrics[
                            "false_negatives"
                        ],

                    "benign_probability_index":
                        self.benign_probability_index,

                    "classes":
                        (
                            len(
                                vector
                            )
                            if "vector" in locals()
                            else None
                        )
                }

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------

        logger.info(
            "🥇 [Threshold Optimizer] "
            f"Optimal threshold = "
            f"{best_threshold:.4f} | "
            f"{metric_target.upper()} = "
            f"{best_score:.4f} | "
            f"Benign index = "
            f"{self.benign_probability_index}"
        )

        return best_metrics

    # ==========================================================================
    # Apply threshold
    # ==========================================================================

    def apply_custom_threshold(
        self,
        predictions_df: DataFrame,
        threshold: float = 0.50
    ) -> DataFrame:
        """
        Apply binary security threshold.

        Creates:

            benign_probability
            attack_probability
            prediction

        where:

            prediction = 0 -> Benign
            prediction = 1 -> Attack
        """

        if not 0.0 <= threshold <= 1.0:

            raise ValueError(
                f"Threshold must be between "
                f"0 and 1. Received: {threshold}"
            )

        logger.info(
            "⚙️ [Threshold Optimizer] "
            f"Applying threshold={threshold:.4f} "
            f"with benign index="
            f"{self.benign_probability_index}"
        )

        # ------------------------------------------------------------------
        # Extract P(Benign)
        # ------------------------------------------------------------------

        extract_benign_probability = (
            build_prob_extractor_udf(
                self.benign_probability_index
            )
        )

        df = (
            predictions_df
            .withColumn(
                "benign_probability",
                extract_benign_probability(
                    col(
                        self.probability_col
                    )
                )
            )
        )

        # ------------------------------------------------------------------
        # P(Attack) = 1 - P(Benign)
        # ------------------------------------------------------------------

        df = (
            df.withColumn(
                "attack_probability",
                1.0
                -
                col(
                    "benign_probability"
                )
            )
        )

        # ------------------------------------------------------------------
        # Binary security prediction
        # ------------------------------------------------------------------

        df = (
            df.withColumn(
                "prediction",

                when(
                    col(
                        "attack_probability"
                    )
                    >= float(threshold),

                    1.0

                ).otherwise(0.0)
            )
        )

        return df