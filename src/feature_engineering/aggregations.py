"""
Distributed Feature Aggregation Module using PySpark.
موديول التجميعات والحسابات الإحصائية السلوكية للشبكة بناءً على عناوين IP أو البروتوكولات.
"""

from typing import List  # استيراد أدوات التوثيق النوعي
from pyspark.sql import DataFrame  # استيراد جدول بيانات سبارك
from pyspark.sql.functions import avg, count, max as s_max, min as s_min, sum as s_sum, col  # استيراد دوال التجميع[cite: 16]
from src.common.logger import get_logger  # استيراد المسجل

logger = get_logger(__name__)  # تهيئة المسجل لهذا الموديول


class FeatureAggregator:  # تعريف كلاس التجميعات الإحصائية[cite: 16]
    """
    Compute distributed aggregations and behavioral summary features.
    حساب تجميعات إحصائية (المجموع، المتوسط، العد) لحزم الاتصال.
    """

    def aggregate_by_key(self, df: DataFrame, group_col: str, target_metric_col: str = "packet_length") -> DataFrame:  # دالة التجميع بمفتاح[cite: 16]
        """
        تجميع البيانات بناءً على مفتاح محدد (مثل src_ip أو protocol) وحساب مقاييس السلوك.
        """
        if group_col not in df.columns or target_metric_col not in df.columns:  # التحقق من وجود الأعمدة
            logger.warning(f"⚠️ الأعمدة المطلوبة للتجميع ({group_col}, {target_metric_col}) غير موجودة بالكامل.")  # تسجيل تحذير
            return df  # إعادة الجدول الأصلي

        logger.info(f"⚙️ [Feature Engineering] تجميع البيانات سلوكياً بناءً على [{group_col}] للمقياس [{target_metric_col}]...")  # تسجيل العملية

        # حساب التجميعات الموزعة (العدد، المتوسط، الإجمالي، والقيمة العظمى)[cite: 16]
        aggregated_stats = (
            df.groupBy(group_col)  # التجميع حسب المفتاح المحدد[cite: 16]
            .agg(
                count("*").alias(f"{group_col}_flow_count"),  # حساب عدد الحزم في هذا التدفق[cite: 16]
                avg(target_metric_col).alias(f"{group_col}_avg_{target_metric_col}"),  # متوسط الحجم[cite: 16]
                s_sum(target_metric_col).alias(f"{group_col}_total_{target_metric_col}"),  # إجمالي الحجم المنقول[cite: 16]
                s_max(target_metric_col).alias(f"{group_col}_max_{target_metric_col}")  # أقصى حجم حزمة
            )
        )

        # دمج الإحصاءات المحسوبة مع الجدول الأصلي عبر عملية Join موزعة
        joined_df = df.join(aggregated_stats, on=group_col, how="left")  # دمج الميزات المجمعة
        return joined_df  # إرجاع الجدول المزود بالخصائص السلوكية