"""
Statistical Data Drift and Distribution Shift Detection Engine.
محرك كشف انحراف التوزيعات الإحصائية باستخدام اختبار Kolmogorov-Smirnov (KS-Test) ومؤشر PSI.
"""

from typing import Dict, Any, List, Tuple  # استيراد أدوات التوثيق النوعي
import numpy as np  # استيراد مكتبة العمليات الرياضية
from scipy.stats import ks_2samp  # استيراد اختبار كولموجوروف-سميرنوف الإحصائي للعينتين
from pyspark.sql import DataFrame  # استيراد نوع جدول بيانات سبارك
from src.common.logger import get_logger  # استيراد نظام التسجيل
from src.common.decorators import time_execution  # استيراد مصمم قياس زمن التنفيذ

logger = get_logger(__name__)  # تهيئة المسجل لهذا الموديول


class DataDriftDetector:
    """
    Detects covariate shift and statistical data drift by comparing baseline training data
    with incoming live production traffic micro-batches.
    """

    def __init__(self, p_value_threshold: float = 0.05, ks_stat_threshold: float = 0.15):
        self.p_value_threshold = p_value_threshold  # عتبة الدلالة الإحصائية (P-value < 0.05 تعني وجود انحراف)
        self.ks_stat_threshold = ks_stat_threshold  # عتبة مسافة الانحراف الأقصى (D-statistic)

    def calculate_ks_drift(
        self,
        baseline_series: np.ndarray,
        current_series: np.ndarray
    ) -> Tuple[float, float, bool]:
        """
        حساب إحصائية KS-Test وقيمة P-value بين توزيعين رقميين لتحديد ما إذا كان هناك انحراف ذو دلالة إحصائية.
        """
        # التأكد من وجود عينات كافية في التوزيعين
        if len(baseline_series) == 0 or len(current_series) == 0:
            return 0.0, 1.0, False

        # تطبيق اختبار Kolmogorov-Smirnov بين عينتي البيانات
        ks_result = ks_2samp(baseline_series, current_series)
        ks_stat = float(ks_result.statistic)  # قيمة مسافة كولموجوروف (D)
        p_val = float(ks_result.pvalue)  # القيمة الاحتمالية

        # يعتبر العمود منحرفاً إذا كانت القيمة الاحتمالية أقل من العتبة وتجاوزت المسافة حد الأمان
        is_drifted = bool(p_val < self.p_value_threshold and ks_stat > self.ks_stat_threshold)

        return round(ks_stat, 4), round(p_val, 6), is_drifted

    def calculate_psi(
        self,
        baseline_series: np.ndarray,
        current_series: np.ndarray,
        num_buckets: int = 10
    ) -> float:
        """
        حساب مؤشر ثبات التوزيع (Population Stability Index - PSI) لقياس التغير الإجمالي في التوزيع.
        """
        if len(baseline_series) == 0 or len(current_series) == 0:
            return 0.0

        # إنشاء فئات متساوية العرض بناءً على بيانات التدريب المرجعية
        quantiles = np.linspace(0, 100, num_buckets + 1)
        bins = np.percentile(baseline_series, quantiles)
        bins[0] -= 1e-5
        bins[-1] += 1e-5

        # حساب النسب المئوية للعينات في كل فئة
        baseline_counts, _ = np.histogram(baseline_series, bins=bins)
        current_counts, _ = np.histogram(current_series, bins=bins)

        # إضافة ثابت صغير جداً لمنع القسمة على صفر أو اللوغاريتم لصفر
        b_pct = (baseline_counts / len(baseline_series)) + 1e-5
        c_pct = (current_counts / len(current_series)) + 1e-5

        # معادلة حساب الـ PSI
        psi_value = np.sum((c_pct - b_pct) * np.log(c_pct / b_pct))
        return round(float(psi_value), 4)

    @time_execution
    def audit_features_drift(
        self,
        baseline_df: DataFrame,
        current_df: DataFrame,
        features_to_monitor: List[str]
    ) -> Dict[str, Any]:
        """
        فحص انحراف قائمة الخصائص المستهدفة بالكامل بين بيانات الأساس وبيانات الإنتاج الحالية.
        """
        logger.info(f"🔍 [Drift Detector] فحص انحراف البيانات لـ {len(features_to_monitor)} خاصية...")
        drift_report = {}
        drifted_features_count = 0

        # استخراج عينات عشوائية سريعة لتحويلها إلى مصفوفات محلية دون استهلاك ذاكرة السبارك
        sample_size = min(baseline_df.count(), 10000)
        base_pd = baseline_df.select(features_to_monitor).sample(fraction=1.0, seed=42).limit(sample_size).toPandas()
        curr_pd = current_df.select(features_to_monitor).sample(fraction=1.0, seed=42).limit(sample_size).toPandas()

        for feature in features_to_monitor:
            if feature in base_pd.columns and feature in curr_pd.columns:
                base_vals = base_pd[feature].dropna().values
                curr_vals = curr_pd[feature].dropna().values

                # حساب KS-Test
                ks_stat, p_val, is_drifted = self.calculate_ks_drift(base_vals, curr_vals)
                
                # حساب PSI
                psi_score = self.calculate_psi(base_vals, curr_vals)

                if is_drifted or psi_score >= 0.2:  # PSI >= 0.2 يدل على انحراف جوهري
                    drifted_features_count += 1
                    logger.warning(f"⚠️ [Drift Alert] تم رصد انحراف في الخاصية [{feature}]: KS-Stat={ks_stat}, P-Val={p_val}, PSI={psi_score}")

                drift_report[feature] = {
                    "ks_statistic": ks_stat,
                    "p_value": p_val,
                    "psi_score": psi_score,
                    "drift_detected": is_drifted or (psi_score >= 0.2),
                    "status": "DRIFTED" if (is_drifted or psi_score >= 0.2) else "STABLE"
                }

        # تحديد الحالة الكلية للبيانات
        drift_percentage = (drifted_features_count / len(features_to_monitor)) * 100 if features_to_monitor else 0.0
        overall_status = "CRITICAL_DRIFT" if drift_percentage > 30.0 else ("WARNING_DRIFT" if drift_percentage > 0 else "HEALTHY")

        summary = {
            "overall_drift_status": overall_status,
            "monitored_features_count": len(features_to_monitor),
            "drifted_features_count": drifted_features_count,
            "drifted_features_percentage": round(drift_percentage, 2),
            "feature_details": drift_report
        }

        logger.info(f"✅ اكتمل فحص الانحراف: الحالة الكلية = [{overall_status}] ({drifted_features_count} خصائص منحرفة).")
        return summary