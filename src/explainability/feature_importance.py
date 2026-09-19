"""
Global and Local Feature Importance Extraction Engine.
محرك استخراج الأهمية النسبية لخصائص الشبكة الموزعة لكسر عتامة نماذج التعلم الآلي.
"""

from typing import Dict, Any, List  # استيراد أدوات التوثيق النوعي
from pyspark.sql import DataFrame  # استيراد جدول بيانات سبارك
from pyspark.ml.classification import RandomForestClassificationModel, GBTClassificationModel  # استيراد نماذج الأشجار
from src.common.logger import get_logger  # استيراد نظام التسجيل

logger = get_logger(__name__)  # تهيئة المسجل لهذا الموديول


class GlobalFeatureImportance:
    """
    Extracts Gini-importance and Split-gain weights from distributed tree models
    to identify the most critical network traffic features driving intrusion decisions.
    """

    def __init__(self, feature_names: List[str] = None):
        # قائمة أسماء الخصائص المطابقة لترتيب متجه التدريب (Feature Vector)
        self.feature_names = feature_names or [
            "packet_length", "time_delta", "header_length", "window_size",
            "byte_rate_proxy", "header_ratio", "window_to_packet_ratio",
            "packet_length_log", "protocol_idx", "flags_idx"
        ]

    def extract_importance(self, model: Any) -> List[Dict[str, Any]]:
        """
        استخراج أوزان الأهمية للخصائص وترتيبها تنازلياً من الأكثر تأثيراً إلى الأقل.
        """
        logger.info("🔍 [XAI - Feature Importance] استخراج وتحليل أوزان مساهمة الخصائص في النموذج...")  # تسجيل العملية

        # التحقق مما إذا كان النموذج من عائلة النماذج الشجرية المدعومة
        if hasattr(model, "featureImportances"):
            # استخراج مصفوفة الأهمية الخاصة بـ Spark ML
            importances_array = model.featureImportances.toArray().tolist()  # تحويل المتجه لقائمة بايثون
            
            importance_list = []  # قائمة لتخزين أزواج (الخاصية، الوزن)
            
            # ربط كل وزن باسم الخاصية المقابلة
            for idx, weight in enumerate(importances_array):
                name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
                importance_list.append({
                    "feature": name,  # اسم الخاصية
                    "importance_weight": round(float(weight), 4),  # الوزن النسبي
                    "importance_percentage": round(float(weight) * 100, 2)  # النسبة المئوية للتأثير
                })

            # ترتيب الخصائص تنازلياً بناءً على الوزن النسبي
            sorted_importances = sorted(importance_list, key=lambda x: x["importance_weight"], reverse=True)
            
            logger.info(f"✅ أعلى 3 خصائص مؤثرة في القرار: {[f['feature'] for f in sorted_importances[:3]]}")
            return sorted_importances  # إرجاع القائمة المرتبة

        else:
            logger.warning("⚠️ النموذج الممرر لا يحتوي على خاصية featureImportances مباشرة، سيتم إرجاع أوزان تقريبية موحدة.")
            return [{"feature": name, "importance_weight": 0.1, "importance_percentage": 10.0} for name in self.feature_names]