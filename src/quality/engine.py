"""
Data Quality Engine (pandas/numpy track).
فحوصات الجودة الخفيفة المقابلة لمنطق Cell 0: عمود الليبل + سلامة التايمستامب + تدقيق النواقص.
"""

from typing import Any, Dict, Optional

import pandas as pd


class DataQualityEngine:
    """بوابة الجودة قبل التنظيف (تُستخدم داخل FeaturePipelineBuilder)."""

    @staticmethod
    def find_label_column(df: pd.DataFrame) -> Optional[str]:
        return next((c for c in df.columns if 'label' in c.lower()), None)

    @staticmethod
    def find_timestamp_column(df: pd.DataFrame) -> Optional[str]:
        return next((c for c in df.columns if 'timestamp' in c.lower()), None)

    def audit_day_frame(self, df: pd.DataFrame, day_name: str = "") -> Dict[str, Any]:
        """تدقيق سريع لإطار اليوم الواحد وإرجاع تقرير."""
        lbl_col = self.find_label_column(df)
        ts_col = self.find_timestamp_column(df)
        missing = {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()}
        report = {
            "day": day_name,
            "rows": len(df),
            "columns": len(df.columns),
            "label_column": lbl_col,
            "timestamp_column": ts_col,
            "missing_values": missing,
            "status": "PASS" if lbl_col else "FAIL_NO_LABEL",
        }
        return report

    def run_all_checks(self, df: pd.DataFrame, day_name: str = "") -> Dict[str, Any]:
        """الواجهة الموحدة: تدقيق + رفع خطأ عند غياب عمود الليبل."""
        report = self.audit_day_frame(df, day_name)
        if report["status"] == "FAIL_NO_LABEL":
            raise ValueError(f"No label column in {day_name or 'frame'}")
        return report
