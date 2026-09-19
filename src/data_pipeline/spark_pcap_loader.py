"""
Distributed Network PCAP Loader and Packet Extraction Engine.
محرك استخراج وتفكيك حزم الشبكة من ملفات PCAP وتحويلها إلى بيانات مهيكلة عبر Spark.
"""

import os  # استيراد مكتبة التعامل مع نظام التشغيل
from typing import List, Dict, Any  # استيراد أدوات التوثيق النوعي للبيانات
from scapy.all import rdpcap, IP, TCP, UDP  # استيراد أدوات قراءة حزم الشبكة من Scapy
from pyspark.sql import DataFrame  # استيراد كائن جدول بيانات Spark
from pyspark.sql.types import (  # استيراد أنواع البيانات المخصصة لبناء Schema الجدول
    StructType,
    StructField,
    StringType,
    DoubleType,
    IntegerType,
    LongType
)
from src.common.spark_manager import SparkManager  # استيراد مدير جلسات Spark
from src.common.logger import get_logger  # استيراد نظام التسجيل
from src.common.exceptions import IngestionFileNotFoundError, DataPipelineError  # استيراد الاستثناءات
from src.common.decorators import time_execution  # استيراد مصمم حساب زمن التنفيذ

logger = get_logger(__name__)  # تهيئة مسجل الأحداث لهذا الموديول


class SparkPCAPLoader:
    """
    مسؤول عن قراءة ملفات التقاط الشبكة (.pcap)، تفكيك طبقات البروتوكولات،
    واستخراج الترويسات (Headers) والحمولة (Payloads) كـ DataFrame موزع.
    """
    def __init__(self):
        # الحصول على جلسة Spark الموزعة الموحدة
        self.spark = SparkManager.get_spark_session()
        # تعريف المخطط الهيكلي الثابت لسجلات الحزم الشبكية (Packet Schema)
        self.packet_schema = StructType([
            StructField("timestamp", DoubleType(), True),         # الطابع الزمني للوصول
            StructField("src_ip", StringType(), True),             # عنوان IP المصدر
            StructField("dst_ip", StringType(), True),             # عنوان IP الوجهة
            StructField("src_port", IntegerType(), True),          # منفذ المصدر
            StructField("dst_port", IntegerType(), True),          # منفذ الوجهة
            StructField("protocol", IntegerType(), True),          # رقم البروتوكول (TCP=6, UDP=17)
            StructField("packet_length", DoubleType(), True),      # طول الحزمة بالبايت
            StructField("time_delta", DoubleType(), True),         # الفارق الزمني عن الحزمة السابقة
            StructField("header_length", DoubleType(), True),      # طول الترويسة
            StructField("window_size", DoubleType(), True),        # حجم نافذة الاستقبال TCP
            StructField("flags", StringType(), True),              # رايات التحكم TCP Flags
            StructField("payload", StringType(), True),            # النص المستخرج من الحمولة
            StructField("label", IntegerType(), True)              # تسمية الفئة (0 عادي / 1 هجوم)
        ])

    def _parse_pcap_file(self, file_path: str, default_label: int = 0) -> List[Dict[str, Any]]:
        """
        دالة داخلية لقراءة ملف PCAP واستخراج الخصائص الفيزيائية والدلالية لكل حزمة.
        """
        # التحقق من وجود الملف على القرص
        if not os.path.exists(file_path):
            logger.error(f"❌ لم يتم العثور على ملف التقاط الشبكة: {file_path}")
            raise IngestionFileNotFoundError(f"PCAP file not found: {file_path}")

        records = []  # قائمة لتجميع الحزم المستخرجة
        prev_time = 0.0  # متغير لحساب الفارق الزمني بين الحزم المتتالية

        logger.info(f"🔍 جاري تفكيك وفحص حزم الملف: {file_path}...")
        try:
            # قراءة كافة الحزم المخزنة في ملف الـ PCAP
            packets = rdpcap(file_path)
            
            for index, packet in enumerate(packets):
                # فحص ما إذا كانت الحزمة تحتوي على طبقة بروتوكول الإنترنت (IP Layer)
                if IP in packet:
                    current_time = float(packet.time)  # توقيت وصول الحزمة
                    # حساب الفارق الزمني (Time Delta) مع وضع 0 لأول حزمة
                    time_delta = (current_time - prev_time) if prev_time > 0 else 0.0
                    prev_time = current_time

                    # استخراج طول الترويسة وطول الحزمة الكلي
                    pkt_len = float(len(packet))
                    ip_layer = packet[IP]
                    src_ip = str(ip_layer.src)
                    dst_ip = str(ip_layer.dst)
                    proto = int(ip_layer.proto)
                    header_len = float(ip_layer.ihl * 4) if hasattr(ip_layer, "ihl") else 20.0

                    src_port = 0
                    dst_port = 0
                    window_size = 0.0
                    flags = "NONE"
                    payload_text = ""

                    # فحص واستخراج خصائص بروتوكول TCP
                    if TCP in packet:
                        tcp_layer = packet[TCP]
                        src_port = int(tcp_layer.sport)
                        dst_port = int(tcp_layer.dport)
                        window_size = float(tcp_layer.window)
                        flags = str(tcp_layer.flags)
                        # استخراج الحمولة الخام وتحويلها لنص آمن لمعالجتها لاحقاً عبر SecBERT
                        raw_payload = bytes(tcp_layer.payload)
                        payload_text = raw_payload.decode("utf-8", errors="ignore").strip()

                    # فحص واستخراج خصائص بروتوكول UDP
                    elif UDP in packet:
                        udp_layer = packet[UDP]
                        src_port = int(udp_layer.sport)
                        dst_port = int(udp_layer.dport)
                        raw_payload = bytes(udp_layer.payload)
                        payload_text = raw_payload.decode("utf-8", errors="ignore").strip()

                    # بناء سجل الحزمة المتكامل
                    packet_record = {
                        "timestamp": current_time,
                        "src_ip": src_ip,
                        "dst_ip": dst_ip,
                        "src_port": src_port,
                        "dst_port": dst_port,
                        "protocol": proto,
                        "packet_length": pkt_len,
                        "time_delta": time_delta,
                        "header_length": header_len,
                        "window_size": window_size,
                        "flags": flags,
                        "payload": payload_text if payload_text else "EMPTY_PAYLOAD",
                        "label": default_label
                    }
                    records.append(packet_record)

            logger.info(f"✅ تم تفكيك واستخراج {len(records)} حزمة بنجاح من {file_path}")
            return records

        except Exception as error:
            logger.error(f"❌ خطأ أثناء تفكيك حزم الـ PCAP: {str(error)}")
            raise DataPipelineError(f"Failed to parse PCAP file: {str(error)}")

    @time_execution
    def load_pcap_to_dataframe(self, file_path: str, default_label: int = 0) -> DataFrame:
        """
        تحويل سجلات الـ PCAP المستخرجة إلى DataFrame موزع داخل Apache Spark.
        """
        # تفكيك ملف الـ PCAP واسترجاع القوائم
        raw_records = self._parse_pcap_file(file_path, default_label)
        
        # إنشاء DataFrame موزع باستخدام الـ Schema المعرفة مسبقاً
        spark_df = self.spark.createDataFrame(data=raw_records, schema=self.packet_schema)
        
        # إعادة تقسيم البيانات على 4 أقسام موزعة لتسريع المعالجة
        return spark_df.repartition(4)