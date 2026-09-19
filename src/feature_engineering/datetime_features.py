"""
Distributed Datetime Feature Engineering.
استخراج الميزات والأنماط الزمنية من طوابع الوقت الشبكية (Timestamps).
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, from_unixtime, hour, minute, dayofweek, when, to_timestamp, coalesce, lit
from pyspark.sql.types import TimestampType
from src.common.logger import get_logger

logger = get_logger("Datetime-Feature-Extractor")


class DatetimeFeatureExtractor:
    def __init__(self):
        pass

    def extract_time_features(self, df: DataFrame, timestamp_col: str = "timestamp") -> DataFrame:
        """استخراج الخصائص الزمنية (الساعة، اليوم، نهاية الأسبوع) من عمود الوقت."""
        if timestamp_col not in df.columns:
            logger.warn(f"⚠️ Column '{timestamp_col}' not found in DataFrame. Skipping datetime extraction.")
            return df

        logger.info(f"⚙️ [Feature Engineering] Extracting temporal features from column: [{timestamp_col}]...")

        col_type = dict(df.dtypes).get(timestamp_col, "")

        # تحويل الطابع الزمني الرقمي (Unix Timestamp) إلى تاريخ ووقت قياسي
        if any(t in col_type for t in ["double", "int", "bigint", "long"]):
            df_with_ts = df.withColumn(
                "temp_dt_col",
                from_unixtime(col(timestamp_col)).cast(TimestampType())
            )
        elif "string" in col_type:
            # محاولة تحويل النص إلى timestamp مع دعم عدة صيغ
            # صيغة DD/MM/YYYY HH:mm:ss (الصيغة في CIC-IDS2018)
            # to_timestamp يعطي NULL للقيم الفاسدة تلقائياً
            df_with_ts = df.withColumn(
                "temp_dt_col",
                coalesce(
                    to_timestamp(col(timestamp_col), "dd/MM/yyyy HH:mm:ss"),
                    to_timestamp(col(timestamp_col), "dd/MM/yyyy HH:mm"),
                    to_timestamp(col(timestamp_col), "yyyy-MM-dd HH:mm:ss"),
                    to_timestamp(col(timestamp_col), "yyyy-MM-dd'T'HH:mm:ss"),
                    to_timestamp(col(timestamp_col))
                )
            )
        else:
            df_with_ts = df.withColumn("temp_dt_col", col(timestamp_col))

        # تصفية الصفوف التي فشل تحويل تاريخها (temp_dt_col = NULL)
        initial_count = df_with_ts.count()
        df_with_ts = df_with_ts.filter(col("temp_dt_col").isNotNull())
        filtered_count = df_with_ts.count()
        
        if initial_count > filtered_count:
            dropped = initial_count - filtered_count
            logger.warn(f"⚠️ Dropped {dropped:,} rows with invalid/malformed timestamps.")

        # استخراج الخصائص الزمنية الأساسية
        df_featured = (
            df_with_ts
            .withColumn("flow_hour", hour(col("temp_dt_col")))
            .withColumn("flow_minute", minute(col("temp_dt_col")))
            .withColumn("flow_dayofweek", dayofweek(col("temp_dt_col")))
            .withColumn(
                "is_weekend",
                when(col("flow_dayofweek").isin(1, 7), 1).otherwise(0)
            )
            .drop("temp_dt_col")
        )

        logger.info("✅ Datetime features extracted: ['flow_hour', 'flow_minute', 'flow_dayofweek', 'is_weekend'].")
        return df_featured

    def extract_datetime_features(self, df: DataFrame, timestamp_col: str = "timestamp") -> DataFrame:
        """دالة بديلة متوافقة بنفس الوظيفة."""
        return self.extract_time_features(df, timestamp_col)


# أسماء مستعارة لتغطية كافة الاستيرادات في __init__.py وباقي الموديولات
DatetimeFeatureGenerator = DatetimeFeatureExtractor
DatetimeFeatures = DatetimeFeatureExtractor
TemporalFeatureExtractor = DatetimeFeatureExtractor