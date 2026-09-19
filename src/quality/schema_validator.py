"""
Schema and Data Contract Validator.
محرك التحقق من ثبات وصحة المخطط الهيكلي وتطابق أنواع الأعمدة مع شروط العقد البرمجي.
"""

from typing import Dict, Any, List  # استيراد أدوات التوثيق النوعي
from pyspark.sql import DataFrame  # استيراد جدول بيانات سبارك
from pyspark.sql.types import StructType  # استيراد نوع مخطط البيانات
from src.quality.base import BaseQualityAnalyzer  # استيراد الكلاس الأساسي
from src.common.logger import get_logger  # استيراد المسجل

logger = get_logger(__name__)  # تهيئة المسجل


class SchemaValidator(BaseQualityAnalyzer):
    """
    Verifies that the incoming DataFrame strictly complies with expected column names and data types.
    يضمن منع تسرب بيانات ذات أعمدة مفقودة أو أنواع بيانات خاطئة إلى خطوط المعالجة.
    """

    def analyze(self, df: DataFrame, expected_schema: StructType = None) -> Dict[str, Any]:
        """
        فحص تطابق أعمدة الـ DataFrame والأنواع مع المخطط المتوقع.
        """
        logger.info("🔍 [Data Quality] فحص وتدقيق الـ Schema ومطابقة أنواع الأعمدة...")

        actual_fields = {field.name: field.dataType.simpleString() for field in df.schema.fields}
        missing_columns: List[str] = []
        type_mismatches: List[Dict[str, str]] = []

        if expected_schema:
            for expected_field in expected_schema.fields:
                field_name = expected_field.name
                expected_type = expected_field.dataType.simpleString()

                # فحص الأعمدة المفقودة
                if field_name not in actual_fields:
                    missing_columns.append(field_name)
                else:
                    # فحص توافق نوع البيانات
                    actual_type = actual_fields[field_name]
                    if actual_type != expected_type:
                        type_mismatches.append({
                            "column": field_name,
                            "expected_type": expected_type,
                            "actual_type": actual_type
                        })

        is_valid = (len(missing_columns) == 0 and len(type_mismatches) == 0)

        result = {
            "is_schema_valid": is_valid,
            "missing_columns": missing_columns,
            "type_mismatches": type_mismatches,
            "total_actual_columns": len(actual_fields)
        }

        if is_valid:
            logger.info("✅ المخطط الهيكلي (Schema) مطابق وصحيح 100%.")
        else:
            logger.warning(f"⚠️ تم رصد عدم تطابق في الـ Schema: أعمدة مفقودة={missing_columns} | أنواع غير متطابقة={type_mismatches}")

        return result