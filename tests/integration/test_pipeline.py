"""
End-to-End Pipeline Integration Test.
اختبار سريان البيانات من طبقة سحب البيانات وحتى التقييم النهائي والتفسير الجنائي.
"""

import pytest
from src.orchestrator.pipeline_orchestrator import EndToEndPipelineOrchestrator
from src.explainability.xai_engine import ExplainabilityEngine
from src.genai_reporting.report_generator import IncidentReportGenerator


def test_end_to_end_pipeline_flow(sample_network_df):
    """تشغيل واختبار خط الأنابيب المتكامل والتأكد من إنجاز كافة المراحل بنجاح."""
    orchestrator = EndToEndPipelineOrchestrator()
    
    # مضاعفة حجم البيانات قليلاً لضمان عمل الـ Cross-Validation
    large_df = sample_network_df.union(sample_network_df).union(sample_network_df).union(sample_network_df)
    
    results = orchestrator.run_training_pipeline(large_df)
    
    assert results["workflow_summary"]["overall_status"] == "COMPLETED"
    assert "validation_report" in results
    assert results["validation_report"]["model_metadata"]["status"] in ["APPROVED_FOR_PRODUCTION", "NEEDS_IMPROVEMENT"]


def test_xai_and_genai_reporting_flow():
    """اختبار توليد ملف التفسير والتقرير الاستخباراتي لحزمة هجوم."""
    xai_engine = ExplainabilityEngine()
    incident_gen = IncidentReportGenerator()

    mock_packet = {
        "src_ip": "192.168.1.105",
        "dst_ip": "10.0.0.1",
        "dst_port": 80,
        "protocol": 6,
        "payload": "SELECT * FROM admin WHERE id=1 OR 1=1--"
    }

    # اختبار التفسير النصي للحمولة
    explanation = xai_engine.token_attribution_engine.explain_payload(mock_packet["payload"])
    assert explanation["is_suspicious"] is True
    assert "SQL_INJECTION" in explanation["detected_threat_patterns"]

    # اختبار توليد تقرير الحادث
    mock_xai = {
        "model_decision_factors": {"top_contributing_features": [{"feature": "packet_length"}]},
        "payload_threat_attribution": explanation
    }
    report_res = incident_gen.create_incident_report(mock_packet, mock_xai, prediction_score=0.98)
    assert "generated_report_path" in report_res