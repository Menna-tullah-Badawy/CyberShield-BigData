"""
Apache Spark Distributed Session & S3 Cloud Configuration Manager.
إدارة جلسة Spark وضبط التوافقية الكاملة لنظام ويندوز مع تعطيل NativeIO المعطوب.
"""

import os
import sys
from pyspark.sql import SparkSession
from src.common.logger import get_logger
from src.common.exceptions import CyberShieldException

logger = get_logger("Spark-Manager")

# توجيه مفسر بايثون للبيئة الافتراضية
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

if sys.platform.startswith('win'):
    hadoop_dir = "C:\\hadoop"
    if os.path.exists(hadoop_dir):
        os.environ["HADOOP_HOME"] = hadoop_dir
        os.environ["hadoop.home.dir"] = hadoop_dir
        os.environ["PATH"] = f"{hadoop_dir}\\bin;" + os.environ.get("PATH", "")


class SparkManager:
    _instance: SparkSession = None

    @classmethod
    def get_spark_session(cls, app_name: str = "CyberShield-BigData-NIDS") -> SparkSession:
        """تهيئة جلسة Spark مع تعطيل NativeIO للويندوز وضبط بروتوكول S3A."""
        if cls._instance is None:
            try:
                logger.info("⚡ Initializing Apache Spark Session with AWS S3 connector...")

                cls._instance = (
                    SparkSession.builder
                    .appName(app_name)
                    .master("local[*]")
                    .config("spark.driver.memory", "4g")
                    .config("spark.executor.memory", "4g")
                    .config(
                        "spark.jars.packages",
                        "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262"
                    )
                    # 1. إيقاف NativeIO الخاص بويندوز لتفادي خطأ access0 UnsatisfiedLinkError
                    .config("spark.hadoop.io.native.lib.available", "false")

                    # 2. المصادقة المفتوحة العامة لـ CSE-CIC-IDS2018
                    .config(
                        "spark.hadoop.fs.s3a.aws.credentials.provider",
                        "org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider"
                    )
                    .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com")
                    .config("spark.hadoop.fs.s3a.path.style.access", "true")
                    .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "true")

                    # 3. إعدادات رقمية صريحة لبروتوكول S3A
                    .config("spark.hadoop.fs.s3a.multipart.purge", "false")
                    .config("spark.hadoop.fs.s3a.multipart.purge.age", "86400")
                    .config("spark.hadoop.fs.s3a.threads.keepalivetime", "60")
                    .config("spark.hadoop.fs.s3a.connection.timeout", "60000")
                    .config("spark.hadoop.fs.s3a.connection.establish.timeout", "60000")
                    .config("spark.hadoop.fs.s3a.socket.timeout", "60000")
                    .config("spark.hadoop.fs.s3a.readahead.range", "65536")
                    .config("spark.hadoop.fs.s3a.multipart.size", "104857600")
                    .config("spark.hadoop.fs.s3a.multipart.threshold", "2147483647")
                    .config("spark.hadoop.fs.s3a.block.size", "33554432")

                    # 4. تحسين استهلاك الذاكرة وتوزيع المعالجة
                    .config("spark.hadoop.fs.s3a.threads.max", "20")
                    .config("spark.hadoop.fs.s3a.threads.core", "10")
                    .config("spark.hadoop.fs.s3a.connection.maximum", "30")
                    .config("spark.hadoop.fs.s3a.fast.upload", "true")
                    .config("spark.hadoop.fs.s3a.fast.upload.buffer", "bytebuffer")
                    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
                    .config("spark.sql.shuffle.partitions", "8")
                    
                    # 5. إصلاح مشكلة Python Worker Timeout على ويندوز
                    .config("spark.python.worker.timeout", "600")
                    .config("spark.executor.heartbeatInterval", "60s")
                    .config("spark.network.timeout", "600s")
                    .config("spark.python.worker.reuse", "false")
                    .config("spark.rpc.message.maxSize", "512")
                    .config("spark.driver.maxResultSize", "2g")
                    .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000")
                    
                    # 6. تحسينات لتقليل حجم المهام والأداء
                    .config("spark.sql.adaptive.enabled", "true")
                    .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
                    .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "64m")
                    .config("spark.sql.autoBroadcastJoinThreshold", "10m")
                    .config("spark.default.parallelism", "8")
                    .config("spark.sql.inMemoryColumnarStorage.compressed", "true")
                    .config("spark.sql.inMemoryColumnarStorage.batchSize", "10000")
                    
                    # 7. تجاوز مشاكل Hadoop على Windows عند الكتابة
                    .config("spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
                    .config("spark.hadoop.parquet.enable.summary-metadata", "false")
                    .config("spark.sql.parquet.writeLegacyFormat", "true")
                    .config("spark.sql.sources.commitProtocolClass", 
                            "org.apache.spark.sql.execution.datasources.SQLHadoopMapReduceCommitProtocol")
                    .config("spark.sql.parquet.output.committer.class", 
                            "org.apache.parquet.hadoop.ParquetOutputCommitter")
                    
                    .getOrCreate()
                )
                cls._instance.sparkContext.setLogLevel("WARN")
                logger.info("✅ Apache Spark successfully connected to S3 Cloud.")

            except Exception as error:
                logger.error(f"❌ Spark Initialization Failed: {str(error)}")
                raise CyberShieldException(f"Spark Initialization Failed: {str(error)}")

        return cls._instance