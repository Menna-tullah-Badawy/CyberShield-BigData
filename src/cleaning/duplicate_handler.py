"""
Duplicate record handling module.
معالج إزالة السجلات المكررة في بيئات البيانات الضخمة الموزعة.
"""

from typing import List  # استيراد أدوات التوثيق النوعي للقوائم
from pyspark.sql import DataFrame  # استيراد نوع جدول بيانات سبارك[cite: 11]
from src.common.logger import get_logger  # استيراد مسجل الأحداث المركزي

logger = get_logger(__name__)  # تهيئة المسجل لهذا الموديول


class DuplicateHandler:  # تعريف كلاس معالجة التكرارات[cite: 11]
    """
    Remove duplicate records from Spark DataFrames across all columns or specific keys.
    يوفر وظائف لحذف التكرار الكامل أو التكرار القائم على مفاتيح الشبكة المنطقية.
    """

    def remove_all(self, df: DataFrame) -> DataFrame:  # دالة حذف التكرار المتطابق بالكامل[cite: 11]
        """
        حذف الصفوف المتطابقة بنسبة 100% في كافة الأعمدة.
        """
        logger.info("🧹 [Cleaning] جاري إزالة كافة السجلات المكررة بالكامل...")  # تسجيل بدء العملية
        initial_count = df.count()  # حساب عدد الصفوف قبل الحذف
        cleaned_df = df.dropDuplicates()  # إزالة الصفوف المكررة تماماً عبر سبارك[cite: 11]
        removed_count = initial_count - cleaned_df.count()  # حساب عدد الصفوف المحذوفة
        logger.info(f"✅ تم حذف {removed_count} سجل مكرر بنجاح.")  # تسجيل النتيجة
        return cleaned_df  # إرجاع جدول البيانات النظيف[cite: 11]

    def remove_by_keys(self, df: DataFrame, keys: List[str]) -> DataFrame:  # دالة حذف التكرار بناءً على أعمدة محددة[cite: 11]
        """
        حذف التكرارات بناءً على قائمة محددة من مفاتيح الأعمال (Business Keys).
        """
        logger.info(f"🧹 [Cleaning] جاري إزالة التكرارات بناءً على المفاتيح: {keys}...")  # تسجيل المفاتيح المستهدفة
        cleaned_df = df.dropDuplicates(keys)  # حذف التكرار وفق المفاتيح المحددة[cite: 11]
        return cleaned_df  # إرجاع الجدول المعالج[cite: 11]