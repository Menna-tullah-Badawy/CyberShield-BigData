"""
Outlier handling module using Interquartile Range (IQR).
معالج وتصفية القيم الشاذة والمتطرفة باستخدام المدى الربيعي الإحصائي الموزع.
"""

from pyspark.sql import DataFrame  # استيراد جدول بيانات سبارك[cite: 13]
from pyspark.sql.functions import col, when  # استيراد دوال معالجة الأعمدة والشروط[cite: 13]
from src.common.logger import get_logger  # استيراد المسجل

logger = get_logger(__name__)  # تهيئة المسجل


class OutlierHandler:  # تعريف كلاس معالجة الشواذ[cite: 13]
    """
    Remove or clip outliers using statistical Interquartile Range (IQR).
    يوفر خيارات لحذف السجلات الشاذة أو قصها عند الحدود الطبيعية (Winsorizing/Clipping).
    """

    def remove_iqr(self, df: DataFrame, column: str, factor: float = 1.5) -> DataFrame:  # دالة إزالة الشواذ عبر IQR[cite: 13]
        """
        فلترة وحذف الصفوف التي تقع خارج النطاق الطبيعي [Q1 - factor*IQR, Q3 + factor*IQR].
        """
        if column not in df.columns:  # التحقق من وجود العمود
            logger.warning(f"⚠️ العمود '{column}' غير موجود لمعالجة الشواذ.")  # تسجيل تحذير
            return df  # إعادة الجدول الأصلي

        logger.info(f"🧹 [Cleaning] معالجة وتصفية الشواذ إحصائياً للعمود: [{column}]...")  # تسجيل العملية

        # حساب الربيع الأول والربيع الثالث بدقة 1% لتسريع الحساب الموزع[cite: 13]
        q1, q3 = df.approxQuantile(column, [0.25, 0.75], 0.01)  #[cite: 13]
        iqr = q3 - q1  # حساب المدى الربيعي[cite: 13]
        lower_bound = q1 - (factor * iqr)  # حساب الحد الأدنى المسموح به[cite: 13]
        upper_bound = q3 + (factor * iqr)  # حساب الحد الأقصى المسموح به[cite: 13]

        # تصفية الجدول والاحتفاظ بالقيم التي تقع داخل الحدود فقط[cite: 13]
        filtered_df = df.filter(
            (col(column) >= lower_bound) & (col(column) <= upper_bound)  #[cite: 13]
        )
        return filtered_df  # إرجاع الجدول المصفى[cite: 13]

    def clip_outliers(self, df: DataFrame, column: str, factor: float = 1.5) -> DataFrame:  # دالة قص الشواذ دون حذف الصفوف
        """
        تعديل القيم الشاذة وقصها عند الحدود (Clipping) بدلاً من حذف الصف كاملاً.
        """
        if column not in df.columns:  # فحص وجود العمود
            return df

        logger.info(f"🧹 [Cleaning] قص وتعديل القيم الشاذة (Clipping) للعمود: [{column}]...")  # تسجيل العملية
        q1, q3 = df.approxQuantile(column, [0.25, 0.75], 0.01)  # حساب الربيعات
        iqr = q3 - q1  # حساب المدى الربيعي
        lower_bound = q1 - (factor * iqr)  # الحد الأدنى
        upper_bound = q3 + (factor * iqr)  # الحد الأقصى

        # استبدال القيمة بالحد الأدنى إذا كانت أصغر منه، وبالحد الأقصى إذا كانت أكبر منه
        return df.withColumn(
            column,
            when(col(column) < lower_bound, lower_bound)
            .when(col(column) > upper_bound, upper_bound)
            .otherwise(col(column))
        )