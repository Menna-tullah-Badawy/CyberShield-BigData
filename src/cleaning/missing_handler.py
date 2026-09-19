"""
Missing value handling module.
معالج تطهير وتعويض القيم المفقودة (Null / NaN) بطرق إحصائية موزعة.
"""

from typing import List, Any  # استيراد أدوات التوثيق النوعي
from pyspark.sql import DataFrame  # استيراد جدول بيانات سبارك[cite: 12]
from pyspark.sql.functions import col, mean  # استيراد دوال معالجة الأعمدة وحساب المتوسط[cite: 12]
from src.common.logger import get_logger  # استيراد المسجل

logger = get_logger(__name__)  # تهيئة المسجل


class MissingValueHandler:  # تعريف كلاس معالجة القيم المفقودة[cite: 12]
    """
    Handle missing values via dropping or statistical imputation (Mean / Constant).
    يتيح حذف الصفوف الناقصة أو تعويضها بقيم ثابتة أو بالمتوسط الحسابي الموزع.
    """

    def drop_rows(self, df: DataFrame, subset: List[str] = None) -> DataFrame:  # دالة إسقاط الصفوف الفارغة[cite: 12]
        """
        حذف أي صف يحتوي على قيم مفقودة في كامل الجدول أو في أعمدة محددة.
        """
        logger.info("🧹 [Cleaning] جاري حذف الصفوف التي تحتوي على قيم فارغة (Nulls)...")  # تسجيل العملية
        return df.na.drop(subset=subset)  # إسقاط الصفوف باستخدام واجهة سبارك na.drop[cite: 12]

    def fill_constant(self, df: DataFrame, value: Any, columns: List[str]) -> DataFrame:  # دالة التعويض بقيمة ثابتة[cite: 12]
        """
        تعويض القيم الفارغة بقيمة افتراضية ثابتة (مثل 0 للأرقام أو UNKNOWN للنصوص).
        """
        logger.info(f"🧹 [Cleaning] تعويض القيم الفارغة في الأعمدة {columns} بالقيمة الثابتة: [{value}]...")  # تسجيل التعويض
        return df.na.fill(value, subset=columns)  # تعويض القيم المفقودة بالقيمة المحددة[cite: 12]

    def fill_mean(self, df: DataFrame, column: str) -> DataFrame:  # دالة التعويض بالمتوسط الحسابي[cite: 12]
        """
        حساب المتوسط الحسابي لعمود رقمي وتعويض القيم المفقودة به تلقائياً.
        """
        if column not in df.columns:  # التحقق من وجود العمود في الجدول
            logger.warning(f"⚠️ العمود '{column}' غير موجود للتعويض بالمتوسط.")  # تسجيل تحذير
            return df  # إعادة الجدول دون تعديل

        logger.info(f"🧹 [Cleaning] حساب وتعويض المتوسط الحسابي للعمود: [{column}]...")  # تسجيل بدء الحساب
        avg_val = df.select(mean(col(column))).first()[0]  # حساب المتوسط الموزع للعمود واستخراج قيمته[cite: 12]

        if avg_val is not None:  # التأكد من أن المتوسط تم حسابه بنجاح
            return df.na.fill(avg_val, subset=[column])  # تعويض القيم الفارغة بالمتوسط المحسوب[cite: 12]
        return df  # إرجاع الجدول[cite: 12]