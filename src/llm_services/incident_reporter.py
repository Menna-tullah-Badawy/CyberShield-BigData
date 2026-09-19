"""
Advanced Incident Reporter with RAG Context Augmentation.
المنسق المتقدم لتوليد تقارير الحوادث المدعومة ببيانات الـ RAG وتصدير خطط الـ Playbooks لفرق الـ SOC.
"""

from typing import Dict, Any
from src.llm_services.rag_retriever import ThreatIntelligenceRAG
from src.genai_reporting.llm_client import CyberLLMClient
from src.common.logger import get_logger

logger = get_logger(__name__)


class AdvancedIncidentReporter:
    """
    Combines RAG threat intel with LLM reasoning to produce end-to-end security incident briefings.
    """

    def __init__(self):
        self.rag_engine = ThreatIntelligenceRAG()
        self.llm_client = CyberLLMClient()

    def generate_augmented_report(self, incident_telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """إنشاء التقرير المعزز بسياق التهديدات المسترجعة."""
        threat_cat = incident_telemetry.get("threat_category", "SQL_INJECTION")
        
        # 1. استرجاع السياق عبر الـ RAG
        rag_context = self.rag_engine.retrieve_threat_context(threat_cat)
        incident_telemetry["rag_knowledge"] = rag_context

        # 2. استدعاء النموذج التوليدي
        report_text = self.llm_client.generate_report(incident_telemetry)

        return {
            "report_content": report_text,
            "retrieved_knowledge": rag_context,
            "incident_metadata": incident_telemetry
        }