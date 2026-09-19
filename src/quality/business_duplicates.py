"""
Business duplicate analyzer module.
محرك فحص التكرار المنطقي بناءً على مفاتيح الأعمال وحركة الشبكة (IP, Ports, Timestamps).
"""

from typing import List, Dict, Any  # استيراد أدوات التوثيق النوعي
from pyspark.sql import DataFrame  # استيراد جدول بيانات سبارك[cite: 2]
from src.quality.base import BaseQualityAnalyzer  # استيراد الكلاس الأساسي[cite: 2]
from src.quality.exceptions import InvalidBusinessKeyError  # استيراد استثناء المفاتيح غير الصالحة[cite: 2]
from src.common.logger import get_logger  # استيراد المسجل

logger = get_logger(__name__)  # تهيئة المسجل


class BusinessDuplicateAnalyzer(BaseQualityAnalyzer):  # وراثة فئة الفاحصين[cite: 2]
    """
    Analyze duplicates using business keys (Composite Security Keys).
    يفحص تكرار الحزم عبر مفاتيح محددة (مثل المصدر والهدف والبروتوكول) لرصد تكرار الاتصالات.
    """

    def analyze(self, df: DataFrame, business_keys: List[str] = None) -> Dict[str, Any]:  #[cite: 2]
        """
        تنفيذ الفحص على الأعمدة المحددة كمفاتيح منطقية.
        """
        # التحقق من أن قائمة المفاتيح غير فارغة[cite: 2]
        if not business_keys:
            raise InvalidBusinessKeyError("Business keys cannot be empty.")  #[cite: 2]

        # التأكد من وجود كل مفتاح ضمن أعمدة الـ DataFrame[cite: 2]
        for key in business_keys:  #[cite: 2]
            if key not in df.columns:  #[cite: 2]
                raise InvalidBusinessKeyError(f"Business key column '{key}' does not exist in DataFrame.")  #[cite: 2]

        logger.info(f"🔍 [Data Quality] فحص التكرار المنطقي لمفاتيح الأعمال: {business_keys}...")

        total_rows = df.count()  # حساب إجمالي عدد الصفوف[cite: 2]

        # حساب الصفوف الفريدة بناءً على الأعمدة المفتاحية فقط[cite: 2]
        unique_rows = df.dropDuplicates(business_keys).count()  #[cite: 2]

        # حساب التكرارات المنطقية[cite: 2]
        duplicate_rows = total_rows - unique_rows  #[cite: 2]

        # حساب النسبة المئوية لتكرار المفاتيح[cite: 2]
        percentage = (duplicate_rows / total_rows * 100) if total_rows else 0.0  #[cite: 2]

        result = {  # هيكلة النتيجة[cite: 2]
            "business_keys": business_keys,  # قائمة المفاتيح المختبرة[cite: 2]
            "duplicate_rows": int(duplicate_rows),  # عدد الصفوف المكررة منطقياً[cite: 2]
            "percentage": round(percentage, 2)  # نسبة التكرار[cite: 2]
        }

        logger.info(f"✅ اكتمل فحص تكرار مفاتيح الأعمال: تم رصد {duplicate_rows} تكرار.")
        return result  # إرجاع نتيجة الفحص[cite: 2]