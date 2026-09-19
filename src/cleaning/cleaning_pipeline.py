"""
Enterprise Data Cleaning Pipeline Orchestrator.
المنسق العام لخط أنابيب تنظيف وتطهير البيانات الموزعة بالكامل.
"""

from pyspark.sql import DataFrame  # استيراد جدول بيانات سبارك[cite: 10]
from src.cleaning.duplicate_handler import DuplicateHandler  # استيراد معالج التكرارات[cite: 10]
from src.cleaning.missing_handler import MissingValueHandler  # استيراد معالج المفقودات[cite: 10]
from src.cleaning.outlier_handler import OutlierHandler  # استيراد معالج الشواذ[cite: 10]
from src.cleaning.type_casting import TypeCasting  # استيراد محول الأنواع[cite: 10]
from src.cleaning.value_replacement import ValueReplacement  # استيراد مستبدل القيم[cite: 10]
from src.common.logger import get_logger  # استيراد نظام التسجيل
from src.common.decorators import time_execution  # استيراد مصمم قياس زمن التنفيذ

logger = get_logger(__name__)  # تهيئة المسجل


class CleaningPipeline:  # تعريف المنسق الشامل للتنظيف[cite: 10]
    """
    Execute end-to-end data cleaning operations on distributed cybersecurity traffic datasets.
    ينسق تسلسل العمليات: إزالة التكرارات، توحيد الأنواع، تعويض المفقودات، معالجة الشواذ، واستبدال القيم.
    """

    def __init__(self):  # مشيد خط الأنابيب وتهيئة كافة المعالجات الفرعية[cite: 10]
        self.duplicates = DuplicateHandler()  # تهيئة معالج التكرارات[cite: 10]
        self.missing = MissingValueHandler()  # تهيئة معالج المفقودات[cite: 10]
        self.outliers = OutlierHandler()  # تهيئة معالج الشواذ[cite: 10]
        self.types = TypeCasting()  # تهيئة محول الأنواع[cite: 10]
        self.replace = ValueReplacement()  # تهيئة مستبدل القيم[cite: 10]

    def remove_duplicates(self, df: DataFrame) -> DataFrame:  # واجهة إزالة التكرارات[cite: 10]
        """إزالة السجلات المكررة بالكامل."""
        return self.duplicates.remove_all(df)  #[cite: 10]

    def drop_missing(self, df: DataFrame) -> DataFrame:  # واجهة إسقاط الصفوف الفارغة[cite: 10]
        """إسقاط الصفوف التي تحتوي على قيم مفقودة."""
        return self.missing.drop_rows(df)  #[cite: 10]

    @time_execution  # قياس زمن تنفيذ خط الأنابيب بالكامل
    def run_cleaning_workflow(self, df: DataFrame) -> DataFrame:
        """
        تشغيل تسلسل التنظيف المؤسسي الكامل على مجموعة البيانات.
        """
        logger.info("=" * 70)
        logger.info("🚀 بدء تشغيل خط أنابيب تنظيف البيانات الموزعة الكامل (Cleaning Pipeline)...")
        logger.info("=" * 70)

        # 1. إزالة السجلات المكررة بالكامل لمنع تضخم البيانات[cite: 10]
        df_cleaned = self.duplicates.remove_all(df)

        # 2. توحيد وتحويل أنواع البيانات للأعمدة الأساسية لحزم الشبكة[cite: 10]
        type_mapping = {
            "packet_length": "double",
            "time_delta": "double",
            "header_length": "double",
            "window_size": "double",
            "protocol": "integer",
            "src_port": "integer",
            "dst_port": "integer",
            "label": "integer"
        }
        df_cleaned = self.types.cast_multiple(df_cleaned, type_mapping)

        # 3. استبدال النصوص الفارغة أو المعطوبة في عمود الـ Payload بقيمة قياسية
        if "payload" in df_cleaned.columns:
            df_cleaned = self.replace.replace(df_cleaned, "payload", "", "EMPTY_PAYLOAD")

        # 4. تعويض القيم المفقودة بالأصفار للأعمدة الرقمية لحركة المرور[cite: 10]
        numeric_cols_to_fill = ["packet_length", "time_delta", "header_length", "window_size"]
        existing_numeric_cols = [c for c in numeric_cols_to_fill if c in df_cleaned.columns]
        df_cleaned = self.missing.fill_constant(df_cleaned, 0.0, existing_numeric_cols)

        # 5. معالجة وتصفية القيم الشاذة المتطرفة لطول الحزم عبر IQR[cite: 10]
        if "packet_length" in df_cleaned.columns:
            df_cleaned = self.outliers.clip_outliers(df_cleaned, "packet_length", factor=3.0)

        logger.info("=" * 70)
        logger.info("✅ اكتملت كافة مراحل تنظيف وتطهير البيانات بنجاح، والبيانات جاهزة للتحليل.")
        logger.info("=" * 70)

        return df_cleaned  # إرجاع جدول البيانات النظيف والجاهز للـ EDA