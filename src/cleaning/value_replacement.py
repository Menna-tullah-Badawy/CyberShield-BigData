"""
Value replacement module.
مستبدل ومصحح القيم الخاطئة أو الشاذة منطقياً في السجلات.
"""

from typing import Any, Dict  # استيراد أدوات التوثيق النوعي
from pyspark.sql import DataFrame  # استيراد جدول بيانات سبارك[cite: 15]
from src.common.logger import get_logger  # استيراد المسجل

logger = get_logger(__name__)  # تهيئة المسجل


class ValueReplacement:  # تعريف كلاس استبدال القيم[cite: 15]
    """
    Replace specific column values or mapped dictionaries to sanitize categorical/numerical noise.
    يقوم باستبدال الرموز الخاطئة أو النصوص غير المفهومة بقيم قياسية موحدة.
    """

    def replace(self, df: DataFrame, column: str, old_value: Any, new_value: Any) -> DataFrame:  # دالة استبدال قيمة مفردة[cite: 15]
        """
        استبدال قيمة قديمة بقيمة جديدة داخل عمود محدد.
        """
        if column in df.columns:  # فحص وجود العمود المستهدف
            logger.info(f"🧹 [Cleaning] استبدال القيمة [{old_value}] بـ [{new_value}] في العمود [{column}]...")  # تسجيل الاستبدال
            return df.replace(old_value, new_value, subset=[column])  # استبدال القيمة عبر دالة replace في سبارك[cite: 15]
        return df  # إرجاع الجدول[cite: 15]

    def replace_mapping(self, df: DataFrame, column: str, mapping_dict: Dict[Any, Any]) -> DataFrame:  # دالة استبدال خريطة قيم
        """
        استبدال مجموعة من القيم القديمة بما يقابلها عبر قاموس استبدال كامل.
        """
        if column in df.columns:  # التحقق من وجود العمود
            logger.info(f"🧹 [Cleaning] تطبيق خريطة استبدال القيم للعمود [{column}]...")  # تسجيل العملية
            return df.replace(mapping_dict, subset=[column])  # تنفيذ الاستبدال المتعدد
        return df  # إرجاع الجدول