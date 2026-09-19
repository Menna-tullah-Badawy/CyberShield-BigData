"""
Distributed Feature Store Management (Enterprise Windows-Compatible).
حفظ واسترجاع الخصائص بصيغة Parquet عبر PyArrow لتجاوز أخطاء ويندوز ونظام الملفات.
"""

import os
import shutil
import pandas as pd
from typing import Optional
from pyspark.sql import DataFrame, SparkSession
from src.common.logger import get_logger
from src.common.spark_manager import SparkManager

logger = get_logger("Feature-Store")


class FeatureStoreManager:
    def __init__(self, base_path: str = "data/feature_store", spark: Optional[SparkSession] = None):
        self.base_path = os.path.abspath(base_path)
        self.spark = spark if spark is not None else SparkManager.get_spark_session()
        os.makedirs(self.base_path, exist_ok=True)

    def save_features_to_store(
        self,
        df: DataFrame,
        table_name: str = "nids_features_latest",
        mode: str = "overwrite"
    ) -> str:
        """حفظ الخصائص محلياً باستخدام PyArrow بدون استدعاء Hadoop NativeIO."""
        target_dir = os.path.join(self.base_path, table_name)
        target_file = os.path.join(target_dir, "features.parquet")
        logger.info(f"💾 حفظ الخصائص في Feature Store المسار: {target_file}...")

        try:
            if mode == "overwrite" and os.path.exists(target_dir):
                logger.info(f"🗑️ حذف البيانات القديمة من: {target_dir}")
                shutil.rmtree(target_dir, ignore_errors=True)

            os.makedirs(target_dir, exist_ok=True)

            logger.info("💾 جاري تصدير البيانات وحفظها عبر PyArrow...")
            pdf = df.toPandas()

            # تحويل أي أعمدة متجهات Spark Vectors إلى قوائم لتوافق PyArrow
            for col_name in pdf.columns:
                if len(pdf) > 0 and hasattr(pdf[col_name].iloc[0], "toArray"):
                    pdf[col_name] = pdf[col_name].apply(
                        lambda v: v.toArray().tolist() if hasattr(v, "toArray") else v
                    )

            pdf.to_parquet(target_file, engine="pyarrow", index=False)
            logger.info(f"✅ تم حفظ الجدول بنجاح: {table_name} ({len(pdf):,} rows)")
            return target_file

        except Exception as e:
            logger.error(f"❌ فشل حفظ الخصائص: {e}")
            raise e

    def load_features_from_store(self, table_name: str = "nids_features_latest") -> DataFrame:
        """استرجاع الخصائص كـ Spark DataFrame."""
        target_file = os.path.join(self.base_path, table_name, "features.parquet")
        
        if not os.path.exists(target_file):
            raise FileNotFoundError(f"❌ الجدول غير موجود في المسار: {target_file}")

        logger.info(f"📥 قراءة الخصائص من: {target_file}...")
        pdf = pd.read_parquet(target_file, engine="pyarrow")
        return self.spark.createDataFrame(pdf)