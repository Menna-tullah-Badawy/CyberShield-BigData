"""
Balanced High-Precision Hybrid Ensemble.
نقل حرفي من Cell 1 في cybershield.ipynb (الأوزان بالنص).
"""

from typing import Dict, Mapping

import numpy as np

# الأوزان بالنص من النوت بوك
HYBRID_WEIGHTS: Dict[str, float] = {
    "MAMBA_SSM": 0.40,
    "XGBoost": 0.35,
    "LightGBM": 0.15,
    "CNN_BiLSTM": 0.10,
}


def hybrid_ensemble_predict(
    predictions: Mapping[str, np.ndarray],
    weights: Mapping[str, float] = HYBRID_WEIGHTS,
) -> np.ndarray:
    """متوسط مرجّح لاحتمالات الأعضاء (نفس سطر النوت بوك)."""
    return sum(weights[k] * predictions[k] for k in weights)
