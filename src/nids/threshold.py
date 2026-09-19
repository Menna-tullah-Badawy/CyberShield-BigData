"""
Exact Precision>=92% & Recall>=90% Calibrated Optimizer.
نقل حرفي من Cell 1 في cybershield.ipynb (دالة optimize_security_threshold).
"""

from typing import Dict

import numpy as np
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             fbeta_score, matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

N_TAUS = 1500
TAU_MIN = 0.01
TAU_MAX = 0.95
TARGET_PRECISION = 0.92
TARGET_RECALL = 0.90
MAX_FPR = 0.015


def optimize_security_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    val_true: np.ndarray,
    val_probs: np.ndarray,
    target_prec: float = TARGET_PRECISION,
    target_rec: float = TARGET_RECALL,
) -> Dict[str, float]:
    """اختيار العتبة المثالية على الـ Validation ثم التقييم على الـ Test."""
    taus = np.linspace(TAU_MIN, TAU_MAX, N_TAUS)
    val_preds_all = (val_probs[:, None] >= taus[None, :]).astype(int)
    y_val_bin = val_true[:, None]

    tp_v = np.sum((y_val_bin == 1) & (val_preds_all == 1), axis=0)
    fp_v = np.sum((y_val_bin == 0) & (val_preds_all == 1), axis=0)
    tn_v = np.sum((y_val_bin == 0) & (val_preds_all == 0), axis=0)
    fn_v = np.sum((y_val_bin == 1) & (val_preds_all == 0), axis=0)

    tpr_v = np.where((tp_v + fn_v) > 0, tp_v / (tp_v + fn_v), 0.0)
    fpr_v = np.where((fp_v + tn_v) > 0, fp_v / (fp_v + tn_v), 0.0)
    prec_v = np.where((tp_v + fp_v) > 0, tp_v / (tp_v + fp_v), 0.0)

    # 1. البحث عن النطاق المثالي: Precision >= 92% و Recall >= 90% مع FPR <= 1.5%
    ideal_mask = (prec_v >= target_prec) & (tpr_v >= target_rec) & (fpr_v <= MAX_FPR)

    if np.any(ideal_mask):
        f1_scores = np.where((prec_v + tpr_v) > 0, (2 * prec_v * tpr_v) / (prec_v + tpr_v), 0.0)
        best_idx = np.argmax(np.where(ideal_mask, f1_scores, -1.0))
    else:
        # 2. إذا لم يجتمعا معاً، اختيار أدنى عتبة تضمن Precision >= 92%
        valid_prec = (prec_v >= target_prec) & (tp_v > 10)
        if np.any(valid_prec):
            best_idx = np.where(valid_prec)[0][np.argmax(tpr_v[valid_prec])]
        else:
            f1_scores = np.where((prec_v + tpr_v) > 0, (2 * prec_v * tpr_v) / (prec_v + tpr_v), 0.0)
            best_idx = np.argmax(f1_scores)

    best_tau = float(taus[best_idx])
    y_pred = (y_probs >= best_tau).astype(int)

    p_vals, r_vals, _ = precision_recall_curve(y_true, y_probs)
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1-Score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "F2-Score": round(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0), 4),
        "MCC": round(matthews_corrcoef(y_true, y_pred), 4),
        "PR-AUC": round(auc(r_vals, p_vals), 4),
        "ROC-AUC": round(roc_auc_score(y_true, y_probs), 4),
        "FPR": round(float(fp_t / (fp_t + tn_t)) if (fp_t + tn_t) > 0 else 0.0, 4),
        "FNR": round(float(fn_t / (fn_t + tp_t)) if (fn_t + tp_t) > 0 else 0.0, 4),
        "Optimal_Tau": round(best_tau, 4),
    }
