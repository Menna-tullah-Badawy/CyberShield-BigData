"""
Feature Engineering Package.
تصدير كافة موديولات هندسة الخصائص السبعة وباني خط الأنابيب الموزع.
"""

from src.feature_engineering.aggregations import FeatureAggregator  # استيراد موديول التجميعات[cite: 16]
from src.feature_engineering.datetime_features import DatetimeFeatureGenerator  # استيراد موديول الميزات الزمنية[cite: 17]
from src.feature_engineering.encoding import CategoricalEncoder  # استيراد موديول الترميز[cite: 18]
from src.feature_engineering.numerical_features import NumericalFeatureTransformer  # استيراد موديول التحويلات الرقمية
from src.feature_engineering.scaling import FeatureScaler  # استيراد موديول التحجيم[cite: 19]
from src.feature_engineering.selection import FeatureSelector  # استيراد موديول اختيار الخصائص[cite: 20]
from src.feature_engineering.transformer import FeatureTransformer  # استيراد موديول المحولات المخصصة[cite: 21]
from src.feature_engineering.pipeline_builder import FeaturePipelineBuilder  # استيراد باني خط الأنابيب

# إتاحة الكلاسات للاستدعاء المباشر
__all__ = [
    "FeatureAggregator",
    "DatetimeFeatureGenerator",
    "CategoricalEncoder",
    "NumericalFeatureTransformer",
    "FeatureScaler",
    "FeatureSelector",
    "FeatureTransformer",
    "FeaturePipelineBuilder"
]