"""
Missing value analyzer module.
محرك فحص وحساب نسب وتوزيع القيم المفقودة (Null/NaN) في البيانات الموزعة.
"""

from typing import List, Dict, Any  # استيراد أدوات التوثيق النوعي للبيانات
from pyspark.sql import DataFrame  # استيراد جدول بيانات سبارك[cite: 7]
from pyspark.sql.functions import col, count, when  # استيراد دوال العد والشرط في سبارك[cite: 7]
from src.quality.base import BaseQualityAnalyzer  # استيراد الكلاس الأساسي للفاحصين[cite: 7]
from src.common.logger import get_logger  # استيراد نظام التسجيل

logger = get_logger(__name__)  # تهيئة المسجل لهذا الموديول


class MissingValueAnalyzer(BaseQualityAnalyzer):  # وراثة الفئة الأساسية[cite: 7]
    """
    Analyze missing values across all columns in a distributed DataFrame.
    يقوم بحساب عدد ونسبة القيم الفارغة في كل عمود باستخدام استعلام موزع واحد.
    """

    def analyze(self, df: DataFrame) -> List[Dict[str, Any]]:  # تنفيذ دالة التحليل[cite: 7]
        """
        فحص كافة الأعمدة وحساب عدد ونسب القيم المفقودة.
        """
        logger.info("🔍 [Data Quality] فحص وتحليل القيم المفقودة (Missing Values)...")

        total_rows = df.count()  # حساب إجمالي عدد الصفوف في الجدول[cite: 7]

        # تجميع استعلام عد القيم الفارغة لكافة الأعمدة في عملية واحدة مجمعة[cite: 7]
        missing_counts = (
            df.select([
                count(when(col(column).isNull(), column)).alias(column)  # عد الحقول الفارغة[cite: 7]
                for column in df.columns  # المرور على كافة الأعمدة الموجودة[cite: 7]
            ])
            .first()  # استرجاع النتيجة الأولى والمجمعة[cite: 7]
            .asDict()  # تحويل النتيجة لقاموس بايثون لتسهيل القراءة[cite: 7]
        )

        report = []  # قائمة لتخزين تفاصيل الفحص لكل عمود[cite: 7]

        for column in df.columns:  # المرور على كل عمود لتنسيق نسبته المئوية[cite: 7]
            missing = missing_counts[column]  # عدد القيم المفقودة في العمود[cite: 7]
            
            # حساب النسبة المئوية للقيم المفقودة مع الحماية من القسمة على صفر[cite: 7]
            percentage = (missing / total_rows * 100) if total_rows else 0.0  #[cite: 7]

            report.append({  # إضافة التقرير التفصيلي للعمود[cite: 7]
                "column": column,  # اسم العمود[cite: 7]
                "missing": int(missing),  # عدد الخانات الفارغة[cite: 7]
                "percentage": round(percentage, 2)  # النسبة المئوية مقربة لرقمين عشريين[cite: 7]
            })

        logger.info(f"✅ اكتمل فحص القيم المفقودة لـ {len(df.columns)} عمود.")
        return report  # إرجاع التقرير الإحصائي النهائي[cite: 7]