"""
Duplicate row analyzer module.
محرك فحص السجلات المكررة بالكامل عبر كافة خصائص الحزم.
"""

from typing import Dict, Any  # استيراد أدوات التوثيق النوعي
from pyspark.sql import DataFrame  # استيراد جدول بيانات سبارك[cite: 3]
from src.quality.base import BaseQualityAnalyzer  # استيراد الكلاس الأساسي للفاحصين[cite: 3]
from src.common.logger import get_logger  # استيراد المسجل

logger = get_logger(__name__)  # تهيئة المسجل


class DuplicateAnalyzer(BaseQualityAnalyzer):  # وراثة الفئة الأساسية لفحص التكرارات[cite: 3]
    """
    Analyze exact duplicate rows across all features in a distributed dataset.
    يحسب عدد ونسبة الصفوف المكررة تماماً لمراقبة تضخم البيانات والتحيز.
    """

    def analyze(self, df: DataFrame) -> Dict[str, Any]:  # تنفيذ دالة التحليل[cite: 3]
        """
        فحص التكرارات وإرجاع إحصائيات الصفوف الفريدة والمكررة.
        """
        logger.info("🔍 [Data Quality] فحص السجلات المكررة بالكامل (Duplicate Rows)...")

        total_rows = df.count()  # حساب إجمالي عدد الصفوف الكلي[cite: 3]

        # حساب عدد الصفوف الفريدة بعد حذف التكرارات المتطابقة[cite: 3]
        unique_rows = df.dropDuplicates().count()  #[cite: 3]

        # حساب عدد الصفوف المكررة بطرح الفريدة من الإجمالي[cite: 3]
        duplicate_rows = total_rows - unique_rows  #[cite: 3]

        # حساب النسبة المئوية للتكرار[cite: 3]
        percentage = (duplicate_rows / total_rows * 100) if total_rows else 0.0  #[cite: 3]

        result = {  # بناء القاموس النهائي لنتائج التكرار[cite: 3]
            "total_rows": int(total_rows),  # إجمالي الصفوف[cite: 3]
            "unique_rows": int(unique_rows),  # الصفوف الفريدة[cite: 3]
            "duplicate_rows": int(duplicate_rows),  # الصفوف المكررة[cite: 3]
            "percentage": round(percentage, 2)  # نسبة التكرار المئوية[cite: 3]
        }

        logger.info(f"✅ اكتمل فحص التكرارات: تم رصد {duplicate_rows} صف مكرر ({result['percentage']}%).")
        return result  # إرجاع النتائج[cite: 3]