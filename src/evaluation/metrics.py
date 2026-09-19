"""
Cybersecurity Evaluation Metrics Engine
=======================================

Evaluates a multiclass model as a binary security detector:

    0 = Benign
    1 = Attack

The original multiclass probability vector is converted into:

    P(Benign)
    P(Attack) = 1 - P(Benign)
"""

from typing import Dict, Any, Optional

import numpy as np

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col,
    count,
    when,
    lower,
    udf
)
from pyspark.sql.types import (
    StringType,
    DoubleType
)

from src.common.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Numerical AUC
# =============================================================================

def safe_trapezoid_auc(
    y: np.ndarray,
    x: np.ndarray
) -> float:

    if hasattr(np, "trapezoid"):
        return float(
            np.trapezoid(y, x)
        )

    if hasattr(np, "trapz"):
        return float(
            np.trapz(y, x)
        )

    return float(
        np.sum(
            (
                x[1:] - x[:-1]
            )
            *
            (
                y[1:] + y[:-1]
            )
            / 2.0
        )
    )


# =============================================================================
# Probability extractor
# =============================================================================

def build_prob_extractor_udf(index: int):

    @udf(returnType=DoubleType())
    def _extract(v):

        if v is None:
            return 0.0

        try:

            if hasattr(v, "toArray"):
                arr = v.toArray()

            elif isinstance(
                v,
                (list, tuple, np.ndarray)
            ):
                arr = np.asarray(
                    v,
                    dtype=float
                )

            else:
                return float(v)

            if index >= len(arr):
                return 0.0

            return float(
                arr[index]
            )

        except Exception:

            return 0.0

    return _extract


# =============================================================================
# Evaluation Engine
# =============================================================================

class CyberEvaluationMetrics:

    def __init__(
        self,
        label_col: str = "label",
        prediction_col: str = "prediction",
        probability_col: str = "probability",
        raw_prediction_col: str = "rawPrediction",
        benign_probability_index: int = 0,
        **kwargs
    ):

        self.label_col = label_col
        self.prediction_col = prediction_col
        self.probability_col = probability_col
        self.raw_prediction_col = raw_prediction_col

        self.benign_probability_index = (
            benign_probability_index
        )

    # =========================================================================
    # Convert label to binary
    # =========================================================================

    def _to_binary_expr(
        self,
        df: DataFrame,
        column_name: str
    ):

        if isinstance(
            df.schema[column_name].dataType,
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

    # =========================================================================
    # Confusion Matrix
    # =========================================================================

    def compute_confusion_matrix(
        self,
        predictions_df: DataFrame
    ) -> Dict[str, int]:

        target_expr = self._to_binary_expr(
            predictions_df,
            self.label_col
        )

        pred_expr = self._to_binary_expr(
            predictions_df,
            self.prediction_col
        )

        prepared = (
            predictions_df
            .withColumn(
                "_target_bin",
                target_expr
            )
            .withColumn(
                "_pred_bin",
                pred_expr
            )
        )

        row = (
            prepared
            .select(

                count(
                    when(
                        (
                            col("_target_bin") == 1
                        )
                        &
                        (
                            col("_pred_bin") == 1
                        ),
                        1
                    )
                ).alias("tp"),

                count(
                    when(
                        (
                            col("_target_bin") == 0
                        )
                        &
                        (
                            col("_pred_bin") == 1
                        ),
                        1
                    )
                ).alias("fp"),

                count(
                    when(
                        (
                            col("_target_bin") == 0
                        )
                        &
                        (
                            col("_pred_bin") == 0
                        ),
                        1
                    )
                ).alias("tn"),

                count(
                    when(
                        (
                            col("_target_bin") == 1
                        )
                        &
                        (
                            col("_pred_bin") == 0
                        ),
                        1
                    )
                ).alias("fn")
            )
            .first()
        )

        if row is None:

            return {
                "true_positives": 0,
                "false_positives": 0,
                "true_negatives": 0,
                "false_negatives": 0
            }

        return {
            "true_positives":
                int(row["tp"] or 0),

            "false_positives":
                int(row["fp"] or 0),

            "true_negatives":
                int(row["tn"] or 0),

            "false_negatives":
                int(row["fn"] or 0)
        }

    # =========================================================================
    # AUC
    # =========================================================================

    def compute_auc_curves(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray
    ) -> Dict[str, float]:

        if len(y_true) == 0:

            return {
                "auc_roc": 0.0,
                "auc_pr": 0.0
            }

        if len(
            np.unique(y_true)
        ) < 2:

            return {
                "auc_roc": 0.0,
                "auc_pr": 0.0
            }

        # -------------------------------------------------------------
        # Sort by descending attack probability
        # -------------------------------------------------------------

        order = np.argsort(
            y_scores
        )[::-1]

        y_sorted = y_true[
            order
        ]

        tp_cum = np.cumsum(
            y_sorted == 1
        )

        fp_cum = np.cumsum(
            y_sorted == 0
        )

        n_pos = float(
            np.sum(y_true == 1)
        )

        n_neg = float(
            np.sum(y_true == 0)
        )

        if (
            n_pos == 0
            or n_neg == 0
        ):

            return {
                "auc_roc": 0.0,
                "auc_pr": 0.0
            }

        tpr = (
            tp_cum / n_pos
        )

        fpr = (
            fp_cum / n_neg
        )

        precision = (
            tp_cum /
            np.maximum(
                tp_cum + fp_cum,
                1
            )
        )

        # -------------------------------------------------------------
        # ROC AUC
        # -------------------------------------------------------------

        x_roc = np.concatenate(
            [
                [0.0],
                fpr
            ]
        )

        y_roc = np.concatenate(
            [
                [0.0],
                tpr
            ]
        )

        auc_roc = safe_trapezoid_auc(
            y_roc,
            x_roc
        )

        # -------------------------------------------------------------
        # PR AUC
        # -------------------------------------------------------------

        x_pr = np.concatenate(
            [
                [0.0],
                tpr
            ]
        )

        y_pr = np.concatenate(
            [
                [1.0],
                precision
            ]
        )

        auc_pr = safe_trapezoid_auc(
            y_pr,
            x_pr
        )

        return {
            "auc_roc": round(
                float(
                    abs(auc_roc)
                ),
                4
            ),

            "auc_pr": round(
                float(
                    abs(auc_pr)
                ),
                4
            )
        }

    # =========================================================================
    # All metrics
    # =========================================================================

    def compute_all_metrics(
        self,
        predictions_df: DataFrame,
        y_prob: Optional[np.ndarray] = None,
        y_true: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:

        # -------------------------------------------------------------
        # Confusion matrix
        # -------------------------------------------------------------

        cm = self.compute_confusion_matrix(
            predictions_df
        )

        tp = cm[
            "true_positives"
        ]

        fp = cm[
            "false_positives"
        ]

        tn = cm[
            "true_negatives"
        ]

        fn = cm[
            "false_negatives"
        ]

        # -------------------------------------------------------------
        # Basic metrics
        # -------------------------------------------------------------

        total = (
            tp + fp + tn + fn
        )

        accuracy = (
            (tp + tn) / total
            if total > 0
            else 0.0
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        f1_score = (
            2 * precision * recall
            /
            (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        f2_score = (
            5 * precision * recall
            /
            (4 * precision + recall)
            if (4 * precision + recall) > 0
            else 0.0
        )

        # -------------------------------------------------------------
        # MCC
        # -------------------------------------------------------------

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

        # -------------------------------------------------------------
        # Security rates
        # -------------------------------------------------------------

        false_positive_rate = (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0.0
        )

        false_negative_rate = (
            fn / (fn + tp)
            if (fn + tp) > 0
            else 0.0
        )

        # -------------------------------------------------------------
        # AUC
        # -------------------------------------------------------------

        auc_roc = 0.0
        auc_pr = 0.0

        try:

            if (
                y_prob is not None
                and y_true is not None
            ):

                auc_dict = (
                    self.compute_auc_curves(
                        np.asarray(
                            y_true,
                            dtype=int
                        ),
                        np.asarray(
                            y_prob,
                            dtype=float
                        )
                    )
                )

            elif (
                "attack_probability"
                in predictions_df.columns
            ):

                rows = (
                    predictions_df
                    .select(
                        self.label_col,
                        "attack_probability"
                    )
                    .collect()
                )

                y_t = np.array(
                    [
                        0
                        if str(
                            row[
                                self.label_col
                            ]
                        ).lower()
                        in [
                            "benign",
                            "normal",
                            "safe",
                            "0",
                            "0.0"
                        ]
                        else 1
                        for row in rows
                    ],
                    dtype=int
                )

                y_p = np.array(
                    [
                        float(
                            row[
                                "attack_probability"
                            ]
                        )
                        for row in rows
                    ],
                    dtype=float
                )

                auc_dict = (
                    self.compute_auc_curves(
                        y_t,
                        y_p
                    )
                )

            else:

                auc_dict = {
                    "auc_roc": 0.0,
                    "auc_pr": 0.0
                }

            auc_roc = auc_dict[
                "auc_roc"
            ]

            auc_pr = auc_dict[
                "auc_pr"
            ]

        except Exception as e:

            logger.warning(
                "⚠️ AUC calculation failed: "
                f"{e}"
            )

        # -------------------------------------------------------------
        # Final result
        # -------------------------------------------------------------

        return {

            "confusion_matrix": cm,

            "accuracy": round(
                float(accuracy),
                4
            ),

            "precision": round(
                float(precision),
                4
            ),

            "recall": round(
                float(recall),
                4
            ),

            "f1_score": round(
                float(f1_score),
                4
            ),

            "f2_score": round(
                float(f2_score),
                4
            ),

            "mcc": round(
                float(mcc),
                4
            ),

            "auc_roc": round(
                float(auc_roc),
                4
            ),

            "auc_pr": round(
                float(auc_pr),
                4
            ),

            "false_positive_rate": round(
                float(false_positive_rate),
                4
            ),

            "false_negative_rate": round(
                float(false_negative_rate),
                4
            )
        }