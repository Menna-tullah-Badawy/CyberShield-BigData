"""
Invalid value analyzer module.
محرك فحص القيم غير الصالحة أو المنتهكة للشروط والقواعد الأمنية المخصصة.
"""

from typing import Dict, Any, Callable  # استيراد أدوات التوثيق والدوال القابلة للاستدعاء[cite: 6]
from pyspark.sql import DataFrame  # استيراد جدول بيانات سبارك[cite: 6]
from pyspark.sql.functions import col  # استيراد دالة col[cite: 6]
from src.quality.base import BaseQualityAnalyzer  # استيراد الكلاس الأساسي[cite: 6]
from src.quality.exceptions import InvalidColumnError  # استيراد استثناء الأعمدة[cite: 6]
from src.common.logger import get_logger  # استيراد المسجل

logger = get_logger(__name__)  # تهيئة المسجل


class InvalidValueAnalyzer(BaseQualityAnalyzer):  # وراثة الفئة الأساسية[cite: 6]
    """
    Detect invalid or corrupted values using user-defined logical rules/conditions.
    يسمح بتمرير دالة شرطية وفحص عدد الحقول التي تخالف هذا الشرط المنطقي.
    """

    def analyze(self, df: DataFrame, column_name: str = None, condition: Callable = None) -> Dict[str, Any]:  #[cite: 6]
        """
        فحص القيم المنتهكة للشرط المحدد.
        """
        # التحقق من وجود العمود[cite: 6]
        if column_name not in df.columns:  #[cite: 6]
            raise InvalidColumnError(f"Target column '{column_name}' does not exist.")  #[cite: 6]

        logger.info(f"🔍 [Data Quality] فحص القيم غير الصالحة للعمود: [{column_name}]...")

        # تصفية وعد السجلات التي لا ينطبق عليها الشرط الصحيح (~condition)[cite: 6]
        invalid_count = df.filter(~condition(col(column_name))).count()  #[cite: 6]

        result = {  # هيكلة النتيجة[cite: 6]
            "column": column_name,  # اسم العمود[cite: 6]
            "invalid_values": int(invalid_count)  # عدد السجلات غير المطابقة[cite: 6]
        }

        logger.info(f"✅ العمود [{column_name}]: تم رصد {invalid_count} قيمة غير صالحة تخالف القاعدة.")
        return result  # إرجاع النتيجة[cite: 6]