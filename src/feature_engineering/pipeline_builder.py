"""
Distributed Feature Engineering Pipeline Builder (NaN-Free Engine).
خط أنابيب هندسة الخصائص وتوليد المتجهات الموزعة مع تنظيف فوري للأرقام غير الصالحة.
"""

from typing import Tuple
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, isnan
from pyspark.ml.feature import VectorAssembler, StandardScalerModel

from src.common.logger import get_logger
from src.common.decorators import measure_performance
from src.feature_engineering.transformer import FeatureTransformer
from src.feature_engineering.datetime_features import DatetimeFeatureExtractor
from src.feature_engineering.numerical_features import NumericalFeatureTransformer
from src.feature_engineering.encoding import CategoricalEncoder
from src.feature_engineering.scaling import FeatureScaler

logger = get_logger("Feature-Pipeline-Builder")


class FeaturePipelineBuilder:
    def __init__(self):
        self.transformer = FeatureTransformer()
        self.dt_extractor = DatetimeFeatureExtractor()
        self.num_engine = NumericalFeatureTransformer()
        self.cat_encoder = CategoricalEncoder()
        self.scaler = FeatureScaler()

    @measure_performance
    def build_features(self, df: DataFrame) -> Tuple[DataFrame, StandardScalerModel]:
        logger.info("=" * 70)
        logger.info("🔬 Starting Distributed Feature Engineering Pipeline...")
        logger.info("=" * 70)

        df_trans = df

        # 1. الميزات المركبة
        if hasattr(self.transformer, "create_composite_features"):
            df_trans = self.transformer.create_composite_features(df_trans)
        elif hasattr(self.transformer, "transform"):
            df_trans = self.transformer.transform(df_trans)

        # 2. الميزات الزمنية
        if "timestamp" in df_trans.columns and hasattr(self.dt_extractor, "extract_time_features"):
            try:
                df_trans = self.dt_extractor.extract_time_features(df_trans, "timestamp")
            except Exception as e:
                logger.warning(f"⚠️ Datetime feature extraction skipped: {e}")

        # 3. الميزات الرقمية
        target_num_cols = ["packet_length", "time_delta", "byte_rate_proxy", "flow_duration", "tot_fwd_pkts"]
        if hasattr(self.num_engine, "apply_log_transform"):
            for col_name in target_num_cols:
                if col_name in df_trans.columns:
                    try:
                        df_trans = self.num_engine.apply_log_transform(df_trans, col_name)
                    except Exception as e:
                        logger.warning(f"⚠️ Log transform skipped for {col_name}: {e}")

        # 4. ترميز الأعمدة الفئوية
        cat_cols = ["protocol", "flags"]
        if hasattr(self.cat_encoder, "index_column"):
            for col_name in cat_cols:
                if col_name in df_trans.columns:
                    try:
                        df_trans = self.cat_encoder.index_column(df_trans, col_name, f"{col_name}_idx")
                    except Exception as e:
                        logger.warning(f"⚠️ Categorical indexing skipped for {col_name}: {e}")

        # 5. استخراج وتنظيف الأعمدة الرقمية من NaN / Inf
        exclude_cols = {
            "timestamp", "src_ip", "dst_ip", "payload", "label", 
            "flow_id", "flags", "protocol", "assembled_features", "final_features"
        }
        numeric_types = {"int", "bigint", "double", "float", "smallint", "tinyint"}
        
        feature_cols = [
            c for c, t in df_trans.dtypes 
            if c not in exclude_cols and t.lower() in numeric_types
        ]

        logger.info(f"🧹 Sanitizing {len(feature_cols)} numeric columns from NaN and Inf values...")
        for c in feature_cols:
            df_trans = df_trans.withColumn(
                c,
                when(col(c).isNull() | isnan(col(c)) | (col(c) == float("inf")) | (col(c) == float("-inf")), 0.0).otherwise(col(c))
            )

        # 6. تجميع الخصائص الرقمية في متجه
        logger.info(f"⚙️ [Feature Engineering] Assembling {len(feature_cols)} features into vector...")
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="assembled_features", handleInvalid="skip")
        df_assembled = assembler.transform(df_trans)

        # 7. المعايرة القياسية
        df_scaled, scaler_model = self.scaler.fit_and_scale(
            df_assembled,
            input_col="assembled_features",
            output_col="final_features"
        )

        logger.info("✅ Distributed Feature Engineering completed successfully.")
        return df_scaled, scaler_model