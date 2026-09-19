"""
PyTest Master Configuration and Spark Test Fixtures.
إعداد جلسة Spark معزولة وخفيفة ومشاركتها بين كافة الاختبارات لتسريع الفحص البرمجي.
"""

import pytest
import os
import shutil

try:
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
    HAS_PYSPARK = True
except ImportError:
    HAS_PYSPARK = False


@pytest.fixture(scope="session")
def spark_session():
    """تهيئة جلسة Spark Session مخصصة لبيئة الاختبارات."""
    if not HAS_PYSPARK:
        pytest.skip("pyspark not installed — skipping Spark-backed tests")
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("CyberShield_PyTest_Suite")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()


@pytest.fixture(scope="function")
def sample_network_df(spark_session):
    """توليد جدول بيانات شبكي نموذجي لاختبارات الوحدة."""
    schema = StructType([
        StructField("timestamp", DoubleType(), True),
        StructField("src_ip", StringType(), True),
        StructField("dst_ip", StringType(), True),
        StructField("src_port", IntegerType(), True),
        StructField("dst_port", IntegerType(), True),
        StructField("protocol", IntegerType(), True),
        StructField("packet_length", DoubleType(), True),
        StructField("time_delta", DoubleType(), True),
        StructField("header_length", DoubleType(), True),
        StructField("window_size", DoubleType(), True),
        StructField("flags", StringType(), True),
        StructField("payload", StringType(), True),
        StructField("label", IntegerType(), True)
    ])

    data = [
        (1718000001.0, "192.168.1.10", "10.0.0.5", 443, 80, 6, 520.0, 0.002, 32.0, 64240.0, "SYN", "GET / HTTP/1.1", 0),
        (1718000002.0, "192.168.1.15", "10.0.0.5", 8080, 80, 6, 1420.0, 0.001, 32.0, 32120.0, "ACK", "UNION SELECT * FROM users", 1),
        (1718000003.0, "192.168.1.20", "10.0.0.5", 5353, 53, 17, 85.0, 0.050, 8.0, 0.0, "NONE", "DNS_QUERY", 0),
        (1718000004.0, "192.168.1.25", "10.0.0.8", 1234, 445, 6, 3200.0, 0.0001, 40.0, 1024.0, "SYN-ACK", "/bin/sh -i", 1),
        (1718000001.0, "192.168.1.10", "10.0.0.5", 443, 80, 6, 520.0, 0.002, 32.0, 64240.0, "SYN", "GET / HTTP/1.1", 0) # سجل مكرر لاختبار التنظيف
    ]

    return spark_session.createDataFrame(data, schema)