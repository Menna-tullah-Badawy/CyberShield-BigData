"""
Feature Aggregations (numpy track).
منطق Cell 0 + Cell 1: بناء السلاسل الزمنية + إحصائيات الـ Flow للنماذج الجدولية.
"""

from typing import Tuple

import numpy as np

SEQ_LEN = 10
SEQ_ATTACK_VOTE = 0.2  # السلسلة Attack لو متوسط الليبل >= 0.2


class FeatureAggregator:
    """التجميعات الزمنية والإحصائية (نفس دوال النوت بوك)."""

    @staticmethod
    def create_day_sequences(
        X_mat: np.ndarray, y_arr: np.ndarray, seq_len: int = SEQ_LEN
    ) -> Tuple[np.ndarray, np.ndarray]:
        """تقطيع مصفوفة اليوم لسلاسل زمنية (دالة النوت بوك بالنص)."""
        num_seq = len(X_mat) // seq_len
        if num_seq == 0:
            return np.empty((0, seq_len, X_mat.shape[1])), np.empty((0,))
        X_cut = X_mat[:num_seq * seq_len].reshape(num_seq, seq_len, X_mat.shape[1])
        y_cut = y_arr[:num_seq * seq_len].reshape(num_seq, seq_len)
        y_final = (np.mean(y_cut, axis=1) >= SEQ_ATTACK_VOTE).astype(np.int64)
        return X_cut, y_final

    @staticmethod
    def extract_flow_stats(X_3d: np.ndarray) -> np.ndarray:
        """آخر قيمة + mean + max + min + std لكل سلسلة (دالة extract_features بالنص)."""
        f_last = X_3d[:, -1, :]
        f_mean = np.mean(X_3d, axis=1)
        f_max = np.max(X_3d, axis=1)
        f_min = np.min(X_3d, axis=1)
        f_std = np.std(X_3d, axis=1)
        return np.hstack([f_last, f_mean, f_max, f_min, f_std])
