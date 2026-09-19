"""
Abstract Base Estimator for Cybersecurity Classification Models.
الكلاس الأساسي المجرد لتوحيد واجهات التدريب، التنبؤ، وحفظ النماذج في المنصة.
"""

from abc import ABC, abstractmethod  # استيراد أدوات بناء الكلاسات المجردة
from pyspark.sql import DataFrame  # استيراد جدول بيانات سبارك
from src.common.logger import get_logger  # استيراد نظام التسجيل

logger = get_logger(__name__)  # تهيئة المسجل لهذا الموديول


class BaseCyberModel(ABC):
    """
    Abstract Base Class for all Machine Learning & Deep Learning Classifiers.
    يفرض تطبيق دوال التدريب (train) والتنبؤ (predict) وحفظ النموذج على كافة النماذج المخصصة.
    """

    def __init__(self, model_name: str, features_col: str = "final_features", label_col: str = "label"):
        self.model_name = model_name  # اسم النموذج
        self.features_col = features_col  # اسم عمود متجه الخصائص
        self.label_col = label_col  # اسم عمود الفئة المستهدفة
        self.fitted_model = None  # متغير لحفظ كائن النموذج بعد اكتمال تدريبه

    @abstractmethod
    def train(self, train_df: DataFrame):
        """
        تدريب النموذج على جدول بيانات التدريب الموزع.
        """
        raise NotImplementedError  # إطلاق خطأ في حال عدم تطبيق الدالة في الكلاس المشتق

    @abstractmethod
    def predict(self, test_df: DataFrame) -> DataFrame:
        """
        توليد التنبؤات والاحتماليات لبيانات الاختبار.
        """
        raise NotImplementedError  # إطلاق خطأ في حال عدم تطبيق الدالة

    def save(self, path: str):
        """
        حفظ النموذج المدرب على القرص بصيغة متوافقة مع محرك Spark.
        """
        if self.fitted_model is None:  # التحقق من أن النموذج تم تدريبه أولاً
            logger.error(f"❌ لا يمكن حفظ النموذج [{self.model_name}] لأنه لم يتم تدريبه بعد.")
            raise ValueError(f"Model {self.model_name} is not fitted yet.")
        
        logger.info(f"💾 [Model Storage] حفظ النموذج [{self.model_name}] في: {path}...")
        self.fitted_model.write().overwrite().save(path)  # كتابة النموذج مع الاستبدال
        logger.info(f"✅ تم حفظ النموذج [{self.model_name}] بنجاح.")