"""
Outlier detection analyzer module using IQR.
محرك اكتشاف القيم الشاذة إحصائياً في بيانات الشبكات باستخدام المدى الربيعي (IQR).
"""

from typing import Dict, Any  # استيراد أدوات التوثيق النوعي
from pyspark.sql import DataFrame  # استيراد جدول بيانات سبارك[cite: 8]
from pyspark.sql.functions import col  # استيراد دالة col للإشارة للأعمدة[cite: 8]
from src.quality.base import BaseQualityAnalyzer  # استيراد الكلاس الأساسي[cite: 8]
from src.quality.exceptions import InvalidColumnError  # استيراد استثناء الأعمدة غير الصحيحة[cite: 8]
from src.common.logger import get_logger  # استيراد المسجل

logger = get_logger(__name__)  # تهيئة المسجل


class OutlierAnalyzer(BaseQualityAnalyzer):  # وراثة الفئة الأساسية[cite: 8]
    """
    Detect statistical outliers in numerical features using Interquartile Range (IQR).
    يحسب الربيع الأول Q1 والربيع الثالث Q3 ويحدد حدود الشذوذ السفلى والعليا.
    """

    def analyze(self, df: DataFrame, column_name: str = None) -> Dict[str, Any]:  #[cite: 8]
        """
        حساب القيم الشاذة لعمود رقمي محدد.
        """
        # التأكد من وجود العمود في الجدول[cite: 8]
        if column_name not in df.columns:  #[cite: 8]
            raise InvalidColumnError(f"Target column '{column_name}' does not exist.")  #[cite: 8]

        logger.info(f"🔍 [Data Quality] فحص القيم الشاذة (Outliers) للعمود: [{column_name}]...")

        # حساب الربيعين 25% و 75% بدقة تقريبية 1% لتسريع المعالجة الموزعة[cite: 8]
        q1, q3 = df.approxQuantile(column_name, [0.25, 0.75], 0.01)  #[cite: 8]

        iqr = q3 - q1  # حساب المدى الربيعي (Interquartile Range)[cite: 8]
        lower_bound = q1 - (1.5 * iqr)  # حساب الحد الأدنى المسموح به[cite: 8]
        upper_bound = q3 + (1.5 * iqr)  # حساب الحد الأقصى المسموح به[cite: 8]

        # عد القيم التي تقع خارج النطاق الطبيعي (أقل من الأدنى أو أعلى من الأقصى)[cite: 8]
        outlier_count = df.filter(
            (col(column_name) < lower_bound) | (col(column_name) > upper_bound)  #[cite: 8]
        ).count()  #[cite: 8]

        result = {  # بناء تقرير الشذوذ الإحصائي[cite: 8]
            "column": column_name,  # اسم العمود[cite: 8]
            "lower_bound": round(float(lower_bound), 2),  # الحد الأدنى[cite: 8]
            "upper_bound": round(float(upper_bound), 2),  # الحد الأقصى[cite: 8]
            "outliers": int(outlier_count)  # عدد الشواذ المرصودة[cite: 8]
        }

        logger.info(f"✅ العمود [{column_name}]: تم رصد {outlier_count} قيمة شاذة خارج النطاق [{result['lower_bound']} إلى {result['upper_bound']}].")
        return result  # إرجاع نتيجة الفحص[cite: 8]