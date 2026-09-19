"""
Unit Tests for Data Cleaning and Sanitization Modules.
"""

import pytest
from src.cleaning.duplicate_handler import DuplicateHandler
from src.cleaning.missing_handler import MissingValueHandler
from src.cleaning.outlier_handler import OutlierHandler


def test_remove_duplicates(sample_network_df):
    """اختبار حذف التكرارات الكاملة من جدول البيانات."""
    initial_count = sample_network_df.count()
    handler = DuplicateHandler()
    cleaned_df = handler.remove_all(sample_network_df)
    
    assert cleaned_df.count() == initial_count - 1
    assert cleaned_df.count() == 4


def test_fill_missing_values(spark_session):
    """اختبار تعويض القيم المفقودة بالأصفار أو المتوسط."""
    schema = sample_network_df.schema if "sample_network_df" in locals() else None
    test_data = [(100.0,), (None,), (300.0,)]
    df = spark_session.createDataFrame(test_data, ["packet_length"])

    handler = MissingValueHandler()
    filled_df = handler.fill_constant(df, value=0.0, columns=["packet_length"])
    
    assert filled_df.filter(filled_df.packet_length.isNull()).count() == 0
    assert filled_df.filter(filled_df.packet_length == 0.0).count() == 1


def test_outlier_clipping(sample_network_df):
    """اختبار قص القيم الشاذة المتطرفة دون حذف الصفوف."""
    handler = OutlierHandler()
    clipped_df = handler.clip_outliers(sample_network_df, "packet_length", factor=1.5)
    
    assert clipped_df.count() == sample_network_df.count()
    max_val = clipped_df.selectExpr("max(packet_length)").first()[0]
    assert max_val <= 3200.0