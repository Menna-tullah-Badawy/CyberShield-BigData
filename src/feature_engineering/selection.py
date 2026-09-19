"""
Distributed Feature Selection Module.
موديول اختيار الخصائص وحذف التباين الصفري (Variance Threshold & Column Filtering).
"""

from typing import List  # استيراد أدوات التوثيق للقوائم
from pyspark.sql import DataFrame  # استيراد جدول بيانات سبارك
from pyspark.ml.feature import VarianceThresholdSelector  # استيراد فاحص التباين الموزع في سبارك
from src.common.logger import get_logger  # استيراد المسجل

logger = get_logger(__name__)  # تهيئة المسجل


class FeatureSelector:  # تعريف كلاس اختيار الخصائص[cite: 20]
    """
    Remove low-variance or redundant features to reduce noise and dimensionality.
    يقوم بحذف الأعمدة الثابتة عديمة الفائدة أو تصفية المتجهات ذات التباين المنعدم.
    """

    def remove_columns(self, df: DataFrame, columns_to_drop: List[str]) -> DataFrame:  # دالة حذف الأعمدة المحددة[cite: 20]
        """
        إسقاط قائمة من الأعمدة غير المرغوب فيها من جدول البيانات.
        """
        existing_to_drop = [c for c in columns_to_drop if c in df.columns]  # تصفية الأعمدة الموجودة
        if existing_to_drop:  # في حال وجود أعمدة للإسقاط
            logger.info(f"⚙️ [Feature Engineering] حذف الأعمدة غير المؤثرة: {existing_to_drop}...")  # تسجيل الحذف
            return df.drop(*existing_to_drop)  # حذف الأعمدة عبر دالة drop في سبارك[cite: 20]
        return df  # إرجاع الجدول كما هو[cite: 20]

    def filter_low_variance(
        self,
        df: DataFrame,
        features_col: str = "scaled_features",
        output_col: str = "selected_features",
        variance_threshold: float = 0.001
    ) -> DataFrame:
        """
        استبعاد الميزات التي يقل تباينها عن العتبة المحددة لمنع إدخال الضوضاء للنموذج.
        """
        if features_col not in df.columns:  # التحقق من وجود عمود المتجهات
            return df

        logger.info(f"⚙️ [Feature Engineering] تصفية المتجه [{features_col}] بعتبة تباين: [{variance_threshold}]...")  # تسجيل الفلترة

        try:
            # تهيئة مرشح التباين الموزع
            selector = VarianceThresholdSelector(
                varianceThreshold=variance_threshold,
                featuresCol=features_col,
                outputCol=output_col
            )
            # تدريب وتطبيق الفلترة
            selected_df = selector.fit(df).transform(df)  # تصفية الميزات الضعيفة
            return selected_df  # إرجاع الجدول المصفى
        except Exception as error:
            logger.warning(f"⚠️ تعذر تطبيق مرشح التباين التلقائي، سيتم استخدام المتجه الأصلي: {str(error)}")  # تسجيل تحذير
            return df.withColumn(output_col, df[features_col])  # الاعتماد على المتجه الأصلي