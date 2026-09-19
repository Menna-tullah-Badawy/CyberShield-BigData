"""
Enterprise Data Quality Engine and Audit Orchestrator.
محرك الجودة الشامل الذي يدير كافة الفحوصات ويصدر تقرير تدقيق الجودة النهائي quality_report.json.
"""

import os  # استيراد مكتبة التعامل مع نظام التشغيل
import json  # استيراد مكتبة التعامل مع ملفات JSON لتصدير التقارير
from typing import List, Dict, Any  # استيراد أدوات التوثيق النوعي
from pyspark.sql import DataFrame  # استيراد جدول بيانات سبارك[cite: 4, 9]
import configs.settings as cfg  # استيراد الإعدادات ومسارات التقارير
from src.quality.missing import MissingValueAnalyzer  # استيراد فاحص القيم المفقودة[cite: 4, 9]
from src.quality.duplicates import DuplicateAnalyzer  # استيراد فاحص التكرارات[cite: 4, 9]
from src.quality.business_duplicates import BusinessDuplicateAnalyzer  # استيراد فاحص مفاتيح الأعمال[cite: 4, 9]
from src.quality.outliers import OutlierAnalyzer  # استيراد كاشف الشواذ[cite: 4, 9]
from src.quality.invalid_values import InvalidValueAnalyzer  # استيراد فاحص القيم غير الصالحة[cite: 4, 9]
from src.quality.schema_validator import SchemaValidator  # استيراد فاحص المخطط
from src.common.logger import get_logger  # استيراد نظام التسجيل
from src.common.decorators import time_execution  # استيراد مصمم قياس الزمن

logger = get_logger(__name__)  # تهيئة المسجل


class DataQualityEngine:  # تعريف محرك الجودة الرئيسي للمنصة[cite: 4, 9]
    """
    Central Quality Gate Engine.
    يجمع كافة الفاحصين وينسق عملية التدقيق الشاملة للبيانات الموزعة قبل بدء المعالجة.
    """

    def __init__(self) -> None:  # مشيد المحرك وتهيئة جميع المحللات[cite: 4, 9]
        self.missing = MissingValueAnalyzer()  # تهيئة فاحص القيم المفقودة[cite: 4, 9]
        self.duplicates = DuplicateAnalyzer()  # تهيئة فاحص التكرارات[cite: 4, 9]
        self.business_duplicates = BusinessDuplicateAnalyzer()  # تهيئة فاحص تكرار المفاتيح[cite: 4, 9]
        self.invalid_values = InvalidValueAnalyzer()  # تهيئة فاحص القيم غير الصالحة[cite: 4, 9]
        self.outliers = OutlierAnalyzer()  # تهيئة فاحص القيم الشاذة[cite: 4, 9]
        self.schema_validator = SchemaValidator()  # تهيئة فاحص المخطط

    @time_execution  # قياس زمن تنفيذ فحص الجودة بالكامل
    def run_all_checks(
        self,
        df: DataFrame,
        dataset_name: str = "Network_Traffic_Dataset",
        business_keys: List[str] = None
    ) -> Dict[str, Any]:
        """
        تشغيل كافة اختبارات الجودة وحفظ التقرير بصيغة JSON.
        """
        logger.info("=" * 70)
        logger.info(f"🛡️ بدء تدقيق الجودة الشامل لمجموعة البيانات: [{dataset_name}]")
        logger.info("=" * 70)

        # 1. تشغيل فحص القيم المفقودة[cite: 4, 9]
        missing_report = self.missing.analyze(df)  #[cite: 4, 9]

        # 2. تشغيل فحص السجلات المكررة بالكامل[cite: 4, 9]
        duplicates_report = self.duplicates.analyze(df)  #[cite: 4, 9]

        # 3. تشغيل فحص التكرار لمفاتيح الأعمال إن وجدت[cite: 4, 9]
        business_dup_report = {}
        if business_keys:  #[cite: 4, 9]
            business_dup_report = self.business_duplicates.analyze(df, business_keys)  #[cite: 4, 9]

        # 4. تشغيل فحص الشواذ للخصائص الرقمية الأساسية لحزم الشبكة
        outliers_report = {}
        for col_name in ["packet_length", "time_delta", "header_length"]:
            if col_name in df.columns:
                outliers_report[col_name] = self.outliers.analyze(df, col_name)

        # 5. تشغيل فحص القيم غير الصالحة (مثال: التأكد أن طول الحزمة دائماً موجب)
        invalid_report = {}
        if "packet_length" in df.columns:
            invalid_report["packet_length"] = self.invalid_values.analyze(
                df, "packet_length", condition=lambda c: c >= 0
            )

        # 6. تلخيص مقاييس جودة البيانات المجمعة[cite: 4, 9]
        columns_with_missing = sum(1 for item in missing_report if item["missing"] > 0)  #[cite: 4, 9]
        quality_status = "PASSED" if (duplicates_report["percentage"] < 10.0 and columns_with_missing == 0) else "WARNING"

        full_quality_report = {
            "dataset_name": dataset_name,
            "quality_status": quality_status,
            "summary": {  #[cite: 4, 9]
                "total_rows": duplicates_report["total_rows"],
                "total_columns": len(df.columns),
                "columns_with_missing": columns_with_missing,  #[cite: 4, 9]
                "duplicate_rows": duplicates_report["duplicate_rows"],  #[cite: 4, 9]
                "duplicate_percentage": duplicates_report["percentage"]
            },
            "missing_values_audit": missing_report,
            "exact_duplicates_audit": duplicates_report,
            "business_duplicates_audit": business_dup_report,
            "outliers_audit": outliers_report,
            "invalid_values_audit": invalid_report
        }

        # 7. حفظ التقرير في مجلد التقارير reports/quality_report.json
        report_file_path = os.path.join(cfg.REPORTS_DIR, "quality_report.json")
        with open(report_file_path, "w", encoding="utf-8") as file:
            json.dump(full_quality_report, file, indent=4, ensure_ascii=False)

        logger.info("=" * 70)
        logger.info(f"✅ تم حفظ تقرير جودة البيانات بنجاح في: {report_file_path}")
        logger.info(f"🎯 حالة الجودة النهائية: [{quality_status}]")
        logger.info("=" * 70)

        return full_quality_report