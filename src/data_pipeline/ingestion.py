"""
CyberShield-BigData: Enterprise Fast Ingestion Engine (High-Performance Parallel Sampler).
==========================================================================================

محرك استيعاب البيانات الموزع فائق السرعة المتوافق 100% مع بيئة ويندوز:
- سحب عينات طبقية ذكية من كافة الأيام الـ 10 بالتوازي عبر طلبات HTTP Byte-Range المباشرة.
- توحيد أسماء الأعمدة وحل مشاكل أنواع البيانات المختلطة (Mixed Data Types).
- التخزين المؤقت الموثوق بصيغة Parquet عبر PyArrow لتجاوز أخطاء Winutils و NativeIO في ويندوز.
- توزيع الـ Spark DataFrame تلقائياً على أنوية المعالج (repartition) لمنع أخطاء Large Task Size.
"""

import io
import os
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession

from src.common.decorators import measure_performance
from src.common.exceptions import CyberShieldException
from src.common.logger import get_logger
from src.common.spark_manager import SparkManager

logger = get_logger("Ingestion-Engine")


class DataIngestionEngine:
    """محرك استيعاب وتطبيع بيانات هجمات CSE-CIC-IDS2018 وتحويلها لـ Spark DataFrame."""

    def __init__(self, spark: Optional[SparkSession] = None):
        self.spark = (
            spark
            if spark is not None
            else SparkManager.get_spark_session()
        )

        self.cache_dir = os.path.join(
            os.getcwd(),
            "data",
            "processed"
        )

        self.cache_parquet_path = os.path.join(
            self.cache_dir,
            "cicids2018_sample.parquet"
        )

        self.base_url = (
            "https://cse-cic-ids2018.s3.amazonaws.com/"
            "Processed%20Traffic%20Data%20for%20ML%20Algorithms/"
        )

    # ============================================================
    # PUBLIC API
    # ============================================================

    @measure_performance
    def load_or_fetch_dataset(
        self,
        sample_fraction: float = 0.05,
        force_download: bool = False
    ) -> DataFrame:
        """
        تحميل مجموعة البيانات من الكاش المحلي السريع أو سحبها وتخزينها تلقائياً.
        """
        # 1. فحص وجود الكاش المحلي
        if os.path.exists(self.cache_parquet_path) and not force_download:
            logger.info(
                f"⚡ [Fast Cache Hit] Loading enterprise Parquet dataset from: {self.cache_parquet_path}"
            )
            try:
                pdf = pd.read_parquet(
                    self.cache_parquet_path,
                    engine="pyarrow"
                )
                logger.info(f"📦 Cached dataset loaded successfully: {len(pdf):,} records")
                return self._create_spark_dataframe(pdf)
            except Exception as error:
                logger.warning(
                    f"⚠️ Cached Parquet file could not be loaded. Rebuilding cache. Reason: {error}"
                )

        # 2. سحب البيانات من S3 بالتوازي عبر الـ Byte-Ranges
        logger.info(
            "🌐 [Fast Range Sampler] Fetching parallel multi-offset slices across all 10 attack days..."
        )
        pdf = self._fetch_all_days_fast_samples()

        if pdf.empty:
            raise CyberShieldException("The ingestion engine returned an empty dataset.")

        # 3. توحيد وضبط أنواع البيانات
        logger.info("🧹 Normalizing DataFrame schema before Parquet serialization...")
        pdf = self._normalize_dataframe(pdf)

        # 4. حفظ الكاش محلياً بصيغة Parquet بتجاوز winutils
        logger.info("💾 Caching sampled dataset locally to Parquet via PyArrow...")
        os.makedirs(self.cache_dir, exist_ok=True)

        try:
            pdf.to_parquet(
                self.cache_parquet_path,
                engine="pyarrow",
                index=False
            )
        except Exception as error:
            logger.error("❌ Failed to serialize dataset to Parquet.")
            self._log_dataframe_schema(pdf)
            raise CyberShieldException(f"Parquet serialization failed: {error}") from error

        logger.info(f"✅ Multi-day dataset successfully cached at: {self.cache_parquet_path}")

        # 5. بناء وتوزيع الـ Spark DataFrame
        return self._create_spark_dataframe(pdf)

    # ============================================================
    # BYTE-RANGE PARALLEL FETCHING
    # ============================================================

    def _fetch_file_slice(self, file_name: str) -> Optional[pd.DataFrame]:
        """سحب عينات مجزأة (بداية، منتصف، وآخر اليوم) من ملف S3 دون تحميله كاملاً."""
        encoded_name = urllib.parse.quote(file_name)
        url = self.base_url + encoded_name

        try:
            # 1. فحص الحجم الكلي للملف
            req_head = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req_head, timeout=15) as response:
                total_size = int(response.headers.get("Content-Length", 0))

            if total_size <= 0:
                logger.warning(f"⚠️ Could not determine file size: [{file_name}]")
                return None

            # 2. قراءة ترويسة الأعمدة (Header)
            req_header = urllib.request.Request(url, headers={"Range": "bytes=0-10240"})
            with urllib.request.urlopen(req_header, timeout=15) as response:
                header_bytes = response.read()

            header_text = header_bytes.decode("utf-8", errors="ignore")
            header_lines = header_text.splitlines()

            if not header_lines:
                logger.warning(f"⚠️ Empty CSV header: [{file_name}]")
                return None

            header_line = header_lines[0].strip()

            # 3. اقتطاع 3 شرائح موزعة (بحجم 1.5MB لكل شريحة)
            offsets = [
                int(total_size * 0.15),
                int(total_size * 0.50),
                int(total_size * 0.85)
            ]
            chunk_size = int(1.5 * 1024 * 1024)

            dataframes: List[pd.DataFrame] = []

            for offset in offsets:
                end_offset = min(offset + chunk_size - 1, total_size - 1)
                req_chunk = urllib.request.Request(
                    url,
                    headers={"Range": f"bytes={offset}-{end_offset}"}
                )

                with urllib.request.urlopen(req_chunk, timeout=30) as response:
                    chunk_bytes = response.read()

                chunk_text = chunk_bytes.decode("utf-8", errors="ignore")
                if not chunk_text.strip():
                    continue

                lines = chunk_text.splitlines()
                if len(lines) <= 1:
                    continue

                # استبعاد السطر الأول غير المكتمل ودمج الهيدر
                valid_lines = lines[1:]
                full_csv_slice = header_line + "\n" + "\n".join(valid_lines)

                try:
                    chunk_df = pd.read_csv(
                        io.StringIO(full_csv_slice),
                        on_bad_lines="skip",
                        low_memory=False
                    )
                    if not chunk_df.empty:
                        dataframes.append(chunk_df)
                except Exception as error:
                    logger.warning(f"⚠️ CSV slice parsing failed for [{file_name}] at offset {offset}: {error}")
                    continue

            if not dataframes:
                logger.warning(f"⚠️ No valid slices found for [{file_name}]")
                return None

            day_df = pd.concat(dataframes, ignore_index=True)
            logger.info(f"📥 Successfully sampled {len(day_df):,} records from: [{file_name[:35]}...]")
            return day_df

        except Exception as error:
            logger.warning(f"⚠️ Fast slice failed for [{file_name}]: {error}")
            return None

    def _fetch_all_days_fast_samples(self) -> pd.DataFrame:
        """سحب عينات الأيام العشرة كاملة بالتوازي."""
        day_files: List[str] = [
            "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",  # FTP & SSH Brute Force
            "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",   # DoS-GoldenEye & Slowloris
            "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",     # DoS-SlowHTTPTest & Hulk
            "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv",   # DDoS-LOIC-HTTP (مكتوبة Thuesday)
            "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",  # DDoS-HOIC & LOIC-UDP
            "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",   # Web Attacks
            "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",     # Web Attacks SQL
            "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",  # Infiltration
            "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",   # Infiltration
            "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"      # Botnet Traffic
        ]

        all_dfs: List[pd.DataFrame] = []

        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_file = {
                executor.submit(self._fetch_file_slice, file_name): file_name
                for file_name in day_files
            }

            for future in as_completed(future_to_file):
                file_name = future_to_file[future]
                try:
                    result = future.result()
                    if result is not None and not result.empty:
                        all_dfs.append(result)
                except Exception as error:
                    logger.warning(f"⚠️ Worker failed for [{file_name}]: {error}")
                    continue

        if not all_dfs:
            raise CyberShieldException("Failed to fetch any valid slices from AWS S3.")

        combined_pdf = pd.concat(all_dfs, ignore_index=True)
        combined_pdf = self._normalize_column_names(combined_pdf)
        logger.info(f"🎯 Total 10-Day Stratified Multi-Attack Sample: {len(combined_pdf):,} records.")
        return combined_pdf

    # ============================================================
    # DATAFRAME NORMALIZATION & PREPARATION
    # ============================================================

    def _normalize_dataframe(self, pdf: pd.DataFrame) -> pd.DataFrame:
        """تنظيف وتوحيد أنماط البيانات وتطهير القيم اللانهائية والمكررة."""
        pdf = pdf.copy()
        pdf = self._normalize_column_names(pdf)

        # 1. تنظيف النصوص وإزالة الفراغات
        object_columns = pdf.select_dtypes(include=["object"]).columns
        for col_name in object_columns:
            pdf[col_name] = pdf[col_name].map(
                lambda val: val.strip() if isinstance(val, str) else val
            )

        # 2. استبدال القيم اللانهائية بالقيم الفارغة NaN
        numeric_columns = pdf.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            pdf[numeric_columns] = pdf[numeric_columns].replace(
                [np.inf, -np.inf],
                np.nan
            )

        # 3. التحويل التلقائي للأعمدة الرقمية المصنفة كـ Object
        object_columns = pdf.select_dtypes(include=["object"]).columns
        for col_name in object_columns:
            series = pdf[col_name]
            if series.dropna().empty or col_name in ["label", "timestamp", "flow_id", "src_ip", "dst_ip"]:
                continue

            converted = pd.to_numeric(series, errors="coerce")
            non_null_count = series.notna().sum()
            if non_null_count > 0:
                numeric_ratio = converted.notna().sum() / non_null_count
                if numeric_ratio >= 0.95:
                    pdf[col_name] = converted

        # 4. تحويل باقي الأعمدة النصية إلى str صريح لسلامة PyArrow
        remaining_objects = pdf.select_dtypes(include=["object"]).columns
        for col_name in remaining_objects:
            pdf[col_name] = pdf[col_name].map(
                lambda val: str(val) if val is not None and not isinstance(val, str) else val
            )

        # 5. إزالة السجلات المكررة
        before_dedup = len(pdf)
        pdf = pdf.drop_duplicates(ignore_index=True)
        removed = before_dedup - len(pdf)
        if removed > 0:
            logger.info(f"🧹 Removed {removed:,} duplicate records.")

        pdf.reset_index(drop=True, inplace=True)
        return pdf

    @staticmethod
    def _normalize_column_names(pdf: pd.DataFrame) -> pd.DataFrame:
        """تنظيف وتوحيد أسماء الأعمدة (lowercase + underscore)."""
        pdf = pdf.copy()
        new_columns = []
        used_names = {}

        for col_name in pdf.columns:
            clean_name = (
                str(col_name)
                .strip()
                .lower()
                .replace(" ", "_")
                .replace("/", "_")
                .replace("-", "_")
                .replace(".", "_")
            )

            while "__" in clean_name:
                clean_name = clean_name.replace("__", "_")
            clean_name = clean_name.strip("_")

            if clean_name in used_names:
                used_names[clean_name] += 1
                clean_name = f"{clean_name}_{used_names[clean_name]}"
            else:
                used_names[clean_name] = 0

            new_columns.append(clean_name)

        pdf.columns = new_columns
        return pdf

    # ============================================================
    # SPARK CONVERSION
    # ============================================================

    def _create_spark_dataframe(self, pdf: pd.DataFrame) -> DataFrame:
        """تحويل Pandas DataFrame إلى Spark DataFrame مع توزيعه بالتوازي."""
        logger.info("⚡ Creating distributed Spark DataFrame...")
        try:
            num_cores = os.cpu_count() or 4
            spark_df = self.spark.createDataFrame(pdf).repartition(num_cores)
            logger.info(f"✅ Distributed Spark DataFrame successfully created across {num_cores} partitions.")
            return spark_df
        except Exception as error:
            logger.error(f"❌ Failed to create Spark DataFrame: {error}")
            self._log_dataframe_schema(pdf)
            raise CyberShieldException(f"Spark DataFrame creation failed: {error}") from error

    @staticmethod
    def _log_dataframe_schema(pdf: pd.DataFrame) -> None:
        """طباعة ملخص بنية الأعمدة للتتبع والتشخيص."""
        logger.info("📋 Final Pandas DataFrame schema:")
        for col_name, dtype in pdf.dtypes.items():
            logger.info(f"    {col_name:<40} -> {dtype}")

    # ============================================================
    # ADDITIONAL METHODS FOR PIPELINE COMPATIBILITY
    # ============================================================

    def ingest_csv_dataset(self, file_path: str) -> DataFrame:
        """تحميل ملف CSV محلي وتحويله إلى Spark DataFrame."""
        logger.info(f"📂 Loading CSV dataset from: {file_path}")
        try:
            pdf = pd.read_csv(file_path, on_bad_lines="skip", low_memory=False)
            pdf = self._normalize_dataframe(pdf)
            logger.info(f"✅ Loaded {len(pdf):,} records from CSV file.")
            return self._create_spark_dataframe(pdf)
        except Exception as error:
            raise CyberShieldException(f"Failed to load CSV file: {error}") from error

    def generate_synthetic_stream(self, num_records: int = 10000) -> DataFrame:
        """إنشاء بيانات شبكية اصطناعية لأغراض الاختبار."""
        logger.info(f"🧪 Generating synthetic network stream with {num_records:,} records...")
        
        np.random.seed(42)
        
        # إنشاء بيانات اصطناعية مشابهة لبيانات الشبكة
        data = {
            "timestamp": np.random.randint(1518000000, 1520000000, num_records),
            "src_ip": [f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}" for _ in range(num_records)],
            "dst_ip": [f"10.0.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}" for _ in range(num_records)],
            "protocol": np.random.choice([6, 17, 1], num_records),  # TCP, UDP, ICMP
            "packet_length": np.random.randint(40, 1500, num_records),
            "time_delta": np.random.uniform(0.001, 2.0, num_records),
            "header_length": np.random.randint(20, 60, num_records),
            "window_size": np.random.randint(1000, 65535, num_records),
            "flags": np.random.choice(["SYN", "ACK", "FIN", "PSH"], num_records),
            "flow_duration": np.random.uniform(0.1, 100.0, num_records),
            "tot_fwd_pkts": np.random.randint(1, 100, num_records),
            "tot_bwd_pkts": np.random.randint(1, 100, num_records),
            "totlen_fwd_pkts": np.random.randint(40, 10000, num_records),
            "totlen_bwd_pkts": np.random.randint(40, 10000, num_records),
            "label": np.random.choice([0, 1], num_records, p=[0.8, 0.2])  # 20% malicious
        }
        
        pdf = pd.DataFrame(data)
        logger.info(f"✅ Generated {len(pdf):,} synthetic network records.")
        return self._create_spark_dataframe(pdf)