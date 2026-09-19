"""
Feature Scaling (sklearn track).
منطق Cell 1: RobustScaler(5, 95) يُدرّب على Train فقط + Clip.
"""

from typing import Tuple

import numpy as np
from sklearn.preprocessing import RobustScaler

SCALER_QMIN, SCALER_QMAX = 5.0, 95.0
SCALER_CLIP = 6.0


class FeatureScaler:
    """التقييس القوي مع القص (نفس سطور النوت بوك)."""

    def __init__(self):
        self.scaler = RobustScaler(quantile_range=(SCALER_QMIN, SCALER_QMAX))

    def fit_transform_train(self, X_train: np.ndarray) -> np.ndarray:
        num_features = X_train.shape[2]
        flat = np.clip(self.scaler.fit_transform(X_train.reshape(-1, num_features)),
                       -SCALER_CLIP, SCALER_CLIP)
        return flat.reshape(X_train.shape[0], X_train.shape[1], num_features)

    def transform(self, X: np.ndarray, clip: float = SCALER_CLIP) -> np.ndarray:
        num_features = X.shape[2]
        flat = np.clip(self.scaler.transform(X.reshape(-1, num_features)), -clip, clip)
        return flat.reshape(X.shape[0], X.shape[1], num_features)

    def scale_all(
        self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """تقييس الثلاثة معاً (يُدرّب على Train فقط)."""
        return self.fit_transform_train(X_train), self.transform(X_val), self.transform(X_test)
