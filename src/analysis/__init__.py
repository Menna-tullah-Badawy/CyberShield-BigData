"""
Exploratory Data Analysis and SOC Metrics Package.
تصدير كلاسات التحليل الإحصائي الأحادي، الثنائي، المتعدد، ومؤشرات الـ SOC.
"""

from src.analysis.univariate import UnivariateAnalyzer  # استيراد محلل المتغيرات الفردية
from src.analysis.bivariate import BivariateAnalyzer  # استيراد محلل العلاقات الثنائية
from src.analysis.multivariate import MultivariateAnalyzer  # استيراد محلل الارتباط المتعدد
from src.analysis.business_kpis import BusinessKPIAnalyzer  # استيراد محلل مؤشرات الأداء السيبرانية
from src.analysis.analyzer import SparkEDAOrchestrator  # استيراد المنسق العام للتحليل

# إتاحة الكلاسات للاستدعاء المباشر عند استخدام (*)
__all__ = [
    "UnivariateAnalyzer",
    "BivariateAnalyzer",
    "MultivariateAnalyzer",
    "BusinessKPIAnalyzer",
    "SparkEDAOrchestrator"
]