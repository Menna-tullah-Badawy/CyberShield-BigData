"""
Real-time Network Traffic Streaming Engine (Spark Structured Streaming).
محرك معالجة تدفقات الحزم الحية في الوقت الحقيقي مع حساب النوافذ الزمنية ونقاط الاسترجاع.
"""

import os  # استيراد مكتبة نظام التشغيل
from pyspark.sql import DataFrame  # استيراد كائن DataFrame
from pyspark.sql.functions import col, window, count, avg, max as s_max  # استيراد دوال المعالجة اللحظية
from pyspark.sql.types import (  # استيراد Schema التدفق اللحظي
    StructType,
    StructField,
    StringType,
    DoubleType,
    IntegerType,
    TimestampType
)
import configs.settings as cfg  # استيراد الإعدادات المركزية
from src.common.spark_manager import SparkManager  # استيراد مدير سبارك
from src.common.logger import get_logger  # استيراد المسجل
from src.common.exceptions import DataPipelineError  # استيراد الاستثناءات

logger = get_logger(__name__)  # تهيئة المسجل


class SparkStreamingPipeline:
    """
    مسؤول عن استقبال تدفقات حزم الشبكة الحية (Micro-batches)، وتطبيق التجميعات
    في نوافذ زمنية منزلقة لكشف هجمات حجب الخدمة (DDoS) والتسلل فور وقوعه.
    """
    def __init__(self):
        # الحصول على جلسة Spark
        self.spark = SparkManager.get_spark_session()
        # تعريف مخطط بيانات التدفق الحي (Streaming Schema)
        self.stream_schema = StructType([
            StructField("timestamp", TimestampType(), True),
            StructField("src_ip", StringType(), True),
            StructField("dst_ip", StringType(), True),
            StructField("protocol", IntegerType(), True),
            StructField("packet_length", DoubleType(), True),
            StructField("header_length", DoubleType(), True),
            StructField("flags", StringType(), True),
            StructField("payload", StringType(), True)
        ])

    def create_file_stream_source(self, monitoring_directory: str) -> DataFrame:
        """
        إنشاء مصدر بث حي يراقب مجلداً معيناً لمعالجة أي ملفات حزم شبكية جديدة تسقط فيه فوراً.
        """
        # التأكد من وجود مجلد المراقبة
        os.makedirs(monitoring_directory, exist_ok=True)
        logger.info(f"📡 بدء الاستماع والمراقبة لتدفق الملفات الحية في: {monitoring_directory}")

        try:
            # بدء قراءة التدفق المستمر بصيغة CSV أو JSON
            streaming_df = (
                self.spark.readStream
                .schema(self.stream_schema)
                .option("maxFilesPerTrigger", 1)  # معالجة ملف واحد في كل دفعة ميكروية
                .csv(monitoring_directory)
            )
            return streaming_df

        except Exception as error:
            logger.error(f"❌ فشل إنشاء مصدر البث الحي: {str(error)}")
            raise DataPipelineError(f"Streaming Source Creation Failed: {str(error)}")

    def apply_windowed_aggregations(self, streaming_df: DataFrame, window_duration: str = "10 seconds", slide_duration: str = "5 seconds") -> DataFrame:
        """
        تطبيق تجميع إحصائي لحظي لحساب كثافة حركة المرور ومعدل تدفق الحزم عبر نوافذ زمنية منزلقة.
        """
        logger.info(f"⚙️ تطبيق النوافذ الزمنية للتدفق الحي: نافذة [{window_duration}] بانزلاق [{slide_duration}]...")

        # تجميع الحزم بناءً على النافذة الزمنية وعنوان IP الوجهة لاكتشاف الهجمات الموجهة
        windowed_stream = (
            streaming_df
            .withWatermark("timestamp", "10 seconds")  # تحديد مهلة زمنية للبيانات المتأخرة لمنع استهلاك الذاكرة
            .groupBy(
                window(col("timestamp"), window_duration, slide_duration),
                col("dst_ip"),
                col("protocol")
            )
            .agg(
                count("*").alias("packets_per_window"),             # عدد الحزم في النافذة
                avg("packet_length").alias("avg_window_packet_len"), # متوسط طول الحزم
                s_max("packet_length").alias("max_window_packet_len") # أقصى طول حزمة
            )
        )
        return windowed_stream

    def start_console_sink(self, streaming_df: DataFrame, query_name: str = "LiveTrafficMonitoring"):
        """
        تشغيل حلقة الاستماع وإخراج النتائج اللحظية مباشرة في التيرمنال لأغراض المراقبة الحية.
        """
        logger.info(f"🚀 تشغيل استعلام البث الحي: [{query_name}]...")
        checkpoint_location = os.path.join(cfg.CHECKPOINTS_DIR, query_name)

        try:
            # كتابة تدفق النتائج في التيرمنال
            query = (
                streaming_df.writeStream
                .outputMode("complete")  # إخراج الجدول المجمع كاملاً عند كل تحديث
                .format("console")
                .queryName(query_name)
                .option("checkpointLocation", checkpoint_location)  # حفظ نقاط الاسترجاع
                .start()
            )
            logger.info(f"✅ استعلام البث الحي [{query_name}] يعمل الآن بنجاح.")
            return query

        except Exception as error:
            logger.error(f"❌ خطأ أثناء بدء استعلام البث الحي: {str(error)}")
            raise DataPipelineError(f"Streaming Sink Failed: {str(error)}")