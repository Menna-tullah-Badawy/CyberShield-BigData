"""
Base class for all data quality analyzers.
الكلاس الأساسي المجرد لجميع محللات فحص جودة البيانات الموزعة.
"""

from abc import ABC, abstractmethod  # استيراد مكتبة بناء الكلاسات المجردة (Abstract Base Classes)
from pyspark.sql import DataFrame  # استيراد نوع جدول بيانات سبارك[cite: 1]


class BaseQualityAnalyzer(ABC):  # تعريف الكلاس الأساسي المجرد لفاحصي الجودة[cite: 1]
    """
    Abstract base class for quality analyzers.
    يفرض تطبيق دالة analyze على كل موديول تدقيق فرعي لضمان توحيد الواجهات البرمجية.
    """

    @abstractmethod  # وسم الدالة بأنها مجردة ويجب بناؤها في الكلاسات المشتقة[cite: 1]
    def analyze(self, df: DataFrame):  # استقبال جدول بيانات سبارك كمدخل إلزامي[cite: 1]
        """
        Analyze the DataFrame and return the quality metrics.
        تنفيذ الفحص وإرجاع قاموس بالنتائج الإحصائية.
        """
        raise NotImplementedError  # إطلاق استثناء في حال استدعاء الدالة مباشرة دون بنائها[cite: 1]