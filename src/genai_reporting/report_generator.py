
"""
Enterprise Incident Report Orchestrator Module.
المنسق العام لإنشاء وحفظ تقارير الحوادث السيبرانية الموثقة بصيغتي Markdown و JSON.
"""

import os  # استيراد مكتبة نظام التشغيل
import json  # استيراد مكتبة التعامل مع JSON
from datetime import datetime  # استيراد أدوات التوقيت والتاريخ
from typing import Dict, Any  # استيراد أدوات التوثيق
import configs.settings as cfg  # استيراد الإعدادات ومسارات التقارير
from src.genai_reporting.mitre_mapping import MitreAttackMapper  # استيراد كاشف MITRE
from src.genai_reporting.llm_client import CyberLLMClient  # استيراد عميل الـ LLM
from src.common.logger import get_logger  # استيراد المسجل
from src.common.decorators import time_execution  # استيراد مصمم قياس الزمن

logger = get_logger(__name__)  # تهيئة المسجل


class IncidentReportGenerator:
    """
    Consolidates model predictions, XAI payload saliency, and MITRE TTPs
    into finalized security incident documentation stored in the reports directory.
    """

    def __init__(self):
        self.mitre_mapper = MitreAttackMapper()  # تهيئة موديول MITRE
        self.llm_client = CyberLLMClient()  # تهيئة عميل الذكاء التوليدي

    @time_execution
    def create_incident_report(
        self,
        packet_record: Dict[str, Any],
        xai_explanation: Dict[str, Any],
        prediction_score: float = 0.98
    ) -> Dict[str, Any]:
        """
        بناء وحفظ تقرير الحادث الأمني الكامل لدفعة الحزم المكتشفة.
        """
        logger.info("=" * 75)
        logger.info("📝 بدء توليد تقرير الحادث الأمني بالذكاء الاصطناعي (GenAI Reporting)...")
        logger.info("=" * 75)

        # 1. تحديد فئة التهديد من نواتج التفسير (XAI)
        detected_patterns = xai_explanation.get("payload_threat_attribution", {}).get("detected_threat_patterns", [])
        threat_cat = detected_patterns[0] if detected_patterns else "PORT_SCAN_RECON"

        # 2. استخراج مطابقة MITRE ATT&CK المقابلة
        mitre_info = self.mitre_mapper.map_threat(threat_cat)

        # 3. تجميع سياق الحادث الشامل
        top_feats = [f["feature"] for f in xai_explanation.get("model_decision_factors", {}).get("top_contributing_features", [])[:3]]
        tokens = [t["token"] for t in xai_explanation.get("payload_threat_attribution", {}).get("highlighted_tokens", [])]

        incident_context = {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "src_ip": packet_record.get("src_ip", "192.168.1.105"),
            "dst_ip": packet_record.get("dst_ip", "10.0.0.1"),
            "dst_port": packet_record.get("dst_port", 80),
            "protocol": packet_record.get("protocol", 6),
            "confidence_score": prediction_score,
            "threat_category": threat_cat,
            "top_features": top_feats,
            "malicious_tokens": tokens if tokens else ["ANOMALOUS_HIGH_RATE_BURST"],
            "payload_sample": packet_record.get("payload", "EMPTY_PAYLOAD"),
            "mitre_tactic": mitre_info["tactic"],
            "mitre_technique_id": mitre_info["technique_id"],
            "mitre_technique_name": mitre_info["technique_name"]
        }

        # 4. توليد نص التقرير الكامل باستخدام LLM Client
        report_markdown = self.llm_client.generate_report(incident_context)

        # 5. حفظ التقرير في مسار reports بصيغتي Markdown و JSON
        timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        md_file_path = os.path.join(cfg.REPORTS_DIR, f"incident_report_{timestamp_str}.md")
        json_file_path = os.path.join(cfg.REPORTS_DIR, f"incident_report_{timestamp_str}.json")

        # كتابة ملف Markdown
        with open(md_file_path, "w", encoding="utf-8") as f_md:
            f_md.write(report_markdown)

        # كتابة ملف JSON المكمل
        final_payload = {
            "metadata": incident_context,
            "mitre_attack_details": mitre_info,
            "generated_report_path": md_file_path
        }
        with open(json_file_path, "w", encoding="utf-8") as f_json:
            json.dump(final_payload, f_json, indent=4, ensure_ascii=False)

        logger.info(f"✅ تم حفظ تقرير الحادث الأمني (Markdown) بنجاح في: {md_file_path}")
        logger.info(f"✅ تم حفظ البيانات المهيكلة (JSON) بنجاح في: {json_file_path}")
        logger.info("=" * 75)

        return final_payload