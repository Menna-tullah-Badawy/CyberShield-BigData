"""
Explainable AI (XAI) and Threat Attribution Package.
تصدير كلاسات تفسير قرارات النماذج، إسناد أهمية الخصائص، وتحليل الرموز المشبوهة في الحمولة.
"""

from src.explainability.feature_importance import GlobalFeatureImportance  # استيراد محرك أهمية الخصائص
from src.explainability.token_attribution import PayloadTokenAttribution  # استيراد محرك إسناد الرموز
from src.explainability.xai_engine import ExplainabilityEngine  # استيراد المنسق الشامل للتفسير

# إتاحة الكلاسات للاستدعاء المباشر
__all__ = [
    "GlobalFeatureImportance",
    "PayloadTokenAttribution",
    "ExplainabilityEngine"
]