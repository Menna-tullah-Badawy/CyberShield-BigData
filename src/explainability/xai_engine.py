"""
Central Explainability (XAI) Orchestrator Engine.
المنسق الشامل لتوليد التقارير التفسيرية لقرارات النماذج وربط الخصائص الرقمية بالنصوص المشبوهة.
"""

from typing import Dict, Any, List  # استيراد أدوات التوثيق
from pyspark.sql import DataFrame  # استيراد جدول بيانات سبارك
from src.explainability.feature_importance import GlobalFeatureImportance  # استيراد محرك أهمية الخصائص
from src.explainability.token_attribution import PayloadTokenAttribution  # استيراد محرك إسناد الرموز
from src.common.logger import get_logger  # استيراد المسجل
from src.common.decorators import time_execution  # استيراد مصمم قياس زمن التنفيذ

logger = get_logger(__name__)  # تهيئة المسجل لهذا الموديول


class ExplainabilityEngine:
    """
    Unified Explainable AI Orchestrator.
    Bridges global structural feature importance with granular payload token saliency
    to provide multi-layered explainability for incident responders and SOC triaging.
    """

    def __init__(self, feature_names: List[str] = None):
        self.feature_importance_engine = GlobalFeatureImportance(feature_names)  # تهيئة محرك أهمية الخصائص
        self.token_attribution_engine = PayloadTokenAttribution()  # تهيئة محرك إسناد الرموز

    @time_execution  # قياس زمن تنفيذ التفسير
    def generate_incident_explanation(
        self,
        model: Any,
        packet_record: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        توليد ملف تفسير متكامل لحزمة شبكة مشبوهة تم رصدها كتهديد.
        """
        logger.info("🔍 [XAI Engine] جاري بناء التفسير متعدد الطبقات للحزمة المشبوهة...")  # تسجيل العملية

        # 1. استخراج الأهمية العامة لخصائص النموذج (Global Model Feature Importance)
        global_importances = self.feature_importance_engine.extract_importance(model)

        # 2. استخراج التفسير النصي الدقيق للحمولة (Granular Token Attribution)
        raw_payload = packet_record.get("payload", "")  # استخراج نص الحمولة
        payload_explanation = self.token_attribution_engine.explain_payload(raw_payload)  # فحص الرموز

        # 3. بناء تقرير التفسير الجنائي المتكامل للـ SOC
        xai_summary = {
            "traffic_metadata": {
                "src_ip": packet_record.get("src_ip", "UNKNOWN"),  # عنوان المصدر
                "dst_ip": packet_record.get("dst_ip", "UNKNOWN"),  # عنوان الوجهة
                "protocol": packet_record.get("protocol", 0),  # البروتوكول
                "packet_length": packet_record.get("packet_length", 0.0)  # طول الحزمة
            },
            "model_decision_factors": {
                "top_contributing_features": global_importances[:5],  # أعلى 5 خصائص حاسمة
                "explanation_method": "Tree-Gini Split Gain & Pattern Saliency"  # طريقة التفسير
            },
            "payload_threat_attribution": payload_explanation,  # تفاصيل الرموز المفسرة في الحمولة
            "analyst_interpretation": (
                f"الحزمة تم تصنيفها كتهديد رئيسي مدفوعاً بـ: [{global_importances[0]['feature']}] "
                f"مع رصد تطابقات مشبوهة في الحمولة: {payload_explanation['detected_threat_patterns']}"
                if payload_explanation["is_suspicious"]
                else f"الحزمة تم تصنيفها بناءً على الأنماط السلوكية لحركة المرور ({global_importances[0]['feature']})."
            )
        }

        logger.info(f"✅ تم بناء تقرير الـ XAI بنجاح: {xai_summary['analyst_interpretation']}")
        return xai_summary  # إرجاع التقرير التفسيري المتكامل