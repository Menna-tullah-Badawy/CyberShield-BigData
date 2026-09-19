"""
Data Splitter (numpy track).
منطق Cell 1: موازنة فئات التدريب بنسبة 1:4 (Undersampling للـ Negatives).
"""

from typing import Tuple

import numpy as np

NEG_POS_RATIO = 4


class DataSplitter:
    """موازنة وتقسيم بيانات التدريب."""

    @staticmethod
    def balance_train_1_to_4(
        X_train_raw: np.ndarray, y_train_raw: np.ndarray, seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Undersampling للـ Negatives بنسبة 1:4 (نفس كود النوت بوك)."""
        rng = np.random.RandomState(seed)
        pos_idx = np.where(y_train_raw == 1)[0]
        neg_idx = np.where(y_train_raw == 0)[0]
        sampled_neg = rng.choice(
            neg_idx, size=min(len(neg_idx), len(pos_idx) * NEG_POS_RATIO), replace=False)
        train_idx = np.concatenate([pos_idx, sampled_neg])
        rng.shuffle(train_idx)
        return X_train_raw[train_idx], y_train_raw[train_idx]
