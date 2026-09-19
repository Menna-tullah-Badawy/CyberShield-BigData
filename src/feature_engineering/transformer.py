"""
Custom Domain Feature Transformer and Augmentation Module.
محول الميزات المركبة المخصصة لبيانات الأمن السيبراني وحركة الشبكة الموزعة.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col
from src.common.logger import get_logger

logger = get_logger("Feature-Transformer")


class FeatureTransformer:
    """
    Generate domain-specific composite cybersecurity features from raw network attributes.
    توليد ميزات نسبية متقدمة لكل من بيانات CIC-IDS2018 الحقيقية وحزم الشبكة المصنعة.
    """

    def augment_network_features(self, df: DataFrame) -> DataFrame:
        """
        توليد ميزات أمنية متقدمة من الخصائص الخام.
        """
        logger.info("⚙️ [Feature Engineering] توليد الخصائص المركبة لبيانات الشبكة (Feature Augmentation)...")
        df_out = df

        # ============================================================
        # 1. خصائص بيانات CIC-IDS2018 الحقيقية
        # ============================================================

        # نسبة الحزم الأمامية إلى الخلفية (Fwd to Bwd Packets Ratio)
        if "tot_fwd_pkts" in df_out.columns and "tot_bwd_pkts" in df_out.columns:
            df_out = df_out.withColumn(
                "fwd_bwd_packet_ratio",
                col("tot_fwd_pkts") / (col("tot_bwd_pkts") + 1.0)
            )

        # نسبة حجم البايتات الأمامية إلى الخلفية (Fwd to Bwd Bytes Ratio)
        if "totlen_fwd_pkts" in df_out.columns and "totlen_bwd_pkts" in df_out.columns:
            df_out = df_out.withColumn(
                "fwd_bwd_bytes_ratio",
                col("totlen_fwd_pkts") / (col("totlen_bwd_pkts") + 1.0)
            )

        # متوسط حجم البايت لكل حزمة أمامية (Avg Bytes per Fwd Packet)
        if "totlen_fwd_pkts" in df_out.columns and "tot_fwd_pkts" in df_out.columns:
            df_out = df_out.withColumn(
                "avg_fwd_byte_per_packet",
                col("totlen_fwd_pkts") / (col("tot_fwd_pkts") + 1.0)
            )

        # ============================================================
        # 2. خصائص البيانات الاصطناعية (Synthetic Stream)
        # ============================================================

        # حساب معدل تدفق البايتس اللحظي (Byte Rate Proxy)
        if "packet_length" in df_out.columns and "time_delta" in df_out.columns:
            df_out = df_out.withColumn(
                "byte_rate_proxy",
                col("packet_length") / (col("time_delta") + 0.00001)
            )

        # نسبة طول الترويسة إلى إجمالي حجم الحزمة (Header-to-Payload Ratio)
        if "header_length" in df_out.columns and "packet_length" in df_out.columns:
            df_out = df_out.withColumn(
                "header_ratio",
                col("header_length") / (col("packet_length") + 1.0)
            )

        # كثافة الحزمة بالنسبة لنافذة الاستقبال (Window-to-Packet Ratio)
        if "window_size" in df_out.columns and "packet_length" in df_out.columns:
            df_out = df_out.withColumn(
                "window_to_packet_ratio",
                col("window_size") / (col("packet_length") + 1.0)
            )

        logger.info("✅ تم توليد الميزات المركبة بنجاح.")
        return df_out

    # دعم لكافة أسماء الاستدعاءات للتوافق مع خط الأنابيب
    def create_composite_features(self, df: DataFrame) -> DataFrame:
        return self.augment_network_features(df)

    def transform(self, df: DataFrame) -> DataFrame:
        return self.augment_network_features(df)