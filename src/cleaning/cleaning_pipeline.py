"""
Data Cleaning Pipeline (pandas/numpy track).
منطق التنظيف من Cell 0/Cell 1: to_numeric + fillna(0) + nan_to_num.
"""

from typing import List

import numpy as np
import pandas as pd


class CleaningPipeline:
    """المنسق الشامل للتنظيف (نفس سطور النوت بوك)."""

    @staticmethod
    def sanitize_dataframe(df: pd.DataFrame, valid_cols: List[str]) -> pd.DataFrame:
        """تحويل رقمي آمن + ملء النواقص بصفر (سطر النوت بوك بالنص)."""
        df[valid_cols] = df[valid_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        return df

    @staticmethod
    def sanitize_array(arr: np.ndarray) -> np.ndarray:
        """تنظيف NaN/Inf من المصفوفات (سطر النوت بوك بالنص)."""
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def run_cleaning_workflow(self, df: pd.DataFrame, valid_cols: List[str]) -> pd.DataFrame:
        """الواجهة الموحدة لتنظيف إطار اليوم الواحد."""
        return self.sanitize_dataframe(df, valid_cols)
