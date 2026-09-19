"""
Datetime Features (pandas track).
منطق Cell 0: تحليل التايمستامب + الترتيب الزمني + العينة المنتظمة.
"""

import pandas as pd

SAMPLE_STEP = 5  # أخذ عينة متساوية عبر كامل اليوم (تمثيل 20% دون تشويه السلاسل)


class DatetimeFeatureGenerator:
    """تجهيز المحور الزمني لإطار اليوم الواحد."""

    @staticmethod
    def parse_and_sort(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
        """تحليل التايمستامب وإسقاط الفاشل والترتيب زمنياً (نفس سطور النوت بوك)."""
        df["ts_parsed"] = pd.to_datetime(df[ts_col], errors="coerce", dayfirst=True)
        return df.dropna(subset=["ts_parsed"]).sort_values("ts_parsed").reset_index(drop=True)

    @staticmethod
    def uniform_sample(df: pd.DataFrame, sample_step: int = SAMPLE_STEP) -> pd.DataFrame:
        """أخذ عينة زمنية ممتدة على مدار الـ 24 ساعة."""
        return df.iloc[::sample_step].reset_index(drop=True)
