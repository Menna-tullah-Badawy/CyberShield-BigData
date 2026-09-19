"""
Unit Tests for 7-Component Feature Engineering Modules.
"""

from src.feature_engineering.transformer import FeatureTransformer
from src.feature_engineering.numerical_features import NumericalFeatureTransformer
from src.feature_engineering.pipeline_builder import FeaturePipelineBuilder


def test_custom_domain_transformer(sample_network_df):
    """اختبار توليد الخصائص المركبة (Byte Rate & Header Ratio)."""
    transformer = FeatureTransformer()
    transformed_df = transformer.augment_network_features(sample_network_df)
    
    assert "byte_rate_proxy" in transformed_df.columns
    assert "header_ratio" in transformed_df.columns
    assert "window_to_packet_ratio" in transformed_df.columns


def test_log_transformation(sample_network_df):
    """اختبار تطبيق التحويل اللوغاريتمي على الخصائص الرقمية."""
    num_trans = NumericalFeatureTransformer()
    log_df = num_trans.apply_log_transform(sample_network_df, ["packet_length"])
    
    assert "packet_length_log" in log_df.columns
    first_val = log_df.select("packet_length_log").first()[0]
    assert first_val > 0.0


def test_feature_pipeline_builder(sample_network_df):
    """اختبار بناء خط أنابيب هندسة الخصائص الكامل وتوليد متجه الميزات."""
    builder = FeaturePipelineBuilder()
    final_df, scaler_model = builder.build_features(sample_network_df)
    
    assert "final_features" in final_df.columns
    assert scaler_model is not None