"""
Feature Selection (pandas/numpy track).
منطق Cell 0: قائمة الحظر STRICT_EXCLUDE + فلتر التباين الصفري (يُدرّب على Train فقط).
"""

from typing import List, Tuple

import numpy as np
import pandas as pd

# قائمة الحظر الصارمة لمنع تسريب الهوية والمنافذ (بالنص من النوت بوك)
STRICT_EXCLUDE = [
    'label', 'target', 'timestamp', 'ts_parsed', 'flow id',
    'src ip', 'dst ip', 'source ip', 'destination ip',
    'src port', 'dst port', 'source port', 'destination port',
    'protocol', 'unnamed: 0',
]

ZERO_VAR_EPS = 1e-5


class FeatureSelector:
    """اختيار الخصائص الآمنة من التسريب."""

    @staticmethod
    def apply_strict_exclude(df: pd.DataFrame) -> List[str]:
        """استبعاد المنافذ والعناوين التعريفية (سطر النوت بوك بالنص)."""
        return [c for c in df.columns if c.lower() not in STRICT_EXCLUDE]

    @staticmethod
    def filter_zero_variance(
        X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
        feature_names: List[str], eps: float = ZERO_VAR_EPS,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """تصفية الميزات عديمة التباين بناءً على التدريب فقط لمنع الـ Leakage."""
        stds = np.std(X_train.reshape(-1, X_train.shape[2]), axis=0)
        valid_feats = stds > eps
        kept = [n for n, k in zip(feature_names, valid_feats) if k]
        return (X_train[:, :, valid_feats], X_val[:, :, valid_feats],
                X_test[:, :, valid_feats], kept)
