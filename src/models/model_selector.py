"""
Model Selector: champion by F1 + Balanced High-Precision Hybrid Ensemble.
منطق Cell 1 من cybershield.ipynb (اختيار الأفضل + الأوزان بالنص).
"""

from typing import Dict, Mapping

import numpy as np
import pandas as pd

# الأوزان بالنص من النوت بوك
HYBRID_WEIGHTS: Dict[str, float] = {
    "MAMBA_SSM": 0.40,
    "XGBoost": 0.35,
    "LightGBM": 0.15,
    "CNN_BiLSTM": 0.10,
}


class ModelSelector:
    """اختيار البطل وبناء الـ Ensemble الهجين."""

    @staticmethod
    def hybrid_ensemble_predict(
        predictions: Mapping[str, np.ndarray],
        weights: Mapping[str, float] = HYBRID_WEIGHTS,
    ) -> np.ndarray:
        """متوسط مرجّح لاحتمالات الأعضاء (نفس سطر النوت بوك)."""
        return sum(weights[k] * predictions[k] for k in weights)

    @staticmethod
    def select_champion_by_f1(df: pd.DataFrame) -> str:
        """اختيار النموذج البطل بأعلى F1-Score (نفس سطر النوت بوك)."""
        return df["F1-Score"].idxmax()
