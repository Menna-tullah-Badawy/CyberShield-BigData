"""
Distributed Feature Store Management.
حفظ واسترجاع الخصائص المهندسة بصيغة Parquet / Delta Lake.
"""

import os
import shutil
from pyspark.sql import DataFrame
from src.common.logger import get_logger
from src.common.spark_manager import SparkManager

logger = get_logger("Feature-Store")


class FeatureStoreManager:
    def __init__(self, base_path: str = "data/feature_store"):
        self.base_path = base_path
        self.spark = SparkManager.get_spark_session()
        os.makedirs(self.base_path, exist_ok=True)

    def save_features_to_store(
        self,
        df: DataFrame,
        table_name: str = "nids_features_latest",
        mode: str = "overwrite"
    ) -> str:
        """حفظ إطار البيانات المجهز في مخزن الخصائص الموزع."""
        target_path = os.path.join(self.base_path, table_name).replace("\\", "/")
        logger.info(f"💾 حفظ الخصائص في Feature Store المسار: {target_path}...")
        
        # حذف المجلد القديم إذا كان موجوداً (في حالة overwrite)
        if mode == "overwrite" and os.path.exists(target_path):
            logger.info(f"🗑️ حذف البيانات القديمة من: {target_path}")
            shutil.rmtree(target_path)
        
        try:
            # استخدام Spark's native parquet writer مع repartition(1) 
            # لإنشاء ملف واحد فقط وتجنب مشاكل Hadoop في إنشاء المجلدات المتعددة
            logger.info("💾 كتابة البيانات باستخدام Spark (repartitioned to 1 file)...")
            
            # إعادة التقسيم إلى partition واحد لإنشاء ملف واحد فقط
            df_single = df.repartition(1)
            
            # كتابة مباشرة باستخدام Spark مع تعطيل Hadoop permissions
            df_single.write \
                .mode(mode) \
                .option("compression", "snappy") \
                .option("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false") \
                .option("parquet.enable.summary-metadata", "false") \
                .option("mapreduce.fileoutputcommitter.algorithm.version", "2") \
                .parquet(target_path)
            
            row_count = df.count()
            logger.info(f"✅ تم حفظ الجدول بنجاح: {table_name} ({row_count:,} rows)")
            return target_path
            
        except Exception as e:
            logger.error(f"❌ فشل حفظ الخصائص: {e}")
            raise

    def load_features_from_store(self, table_name: str = "nids_features_latest") -> DataFrame:
        """استرجاع الخصائص من الـ Feature Store."""
        target_path = os.path.join(self.base_path, table_name).replace("\\", "/")
        
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"❌ الجدول غير موجود في المسار: {target_path}")
        
        logger.info(f"📥 قراءة الخصائص من: {target_path}...")
        return self.spark.read.parquet(target_path)