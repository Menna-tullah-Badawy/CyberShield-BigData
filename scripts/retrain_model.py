"""
Automated Model Retraining Trigger Script.
سكريبت إعادة تدريب النموذج تلقائياً عند استلام إشارة حدوث انحراف للبيانات (Data Drift Trigger).
"""

import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestrator.pipeline_orchestrator import EndToEndPipelineOrchestrator
from src.data_pipeline.ingestion import DataIngestionEngine
import configs.settings as cfg
from src.common.logger import get_logger

logger = get_logger("CyberShield-AutoRetrain")


def check_and_retrain():
    drift_report_path = os.path.join(cfg.REPORTS_DIR, "monitoring_drift_report.json")
    
    if not os.path.exists(drift_report_path):
        logger.info("ℹ️ لم يتم العثور على تقرير انحراف حديث. سيتم تخطي إعادة التدريب.")
        return

    with open(drift_report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    retrain_needed = report.get("retraining_signal", {}).get("triggered", False)

    if retrain_needed:
        logger.warning("🚨 [Retrain Trigger] تم رصد انحراف جوهري في البيانات! بدء دورة إعادة تدريب النموذج الفائز...")
        from main import CyberShieldOrchestrator
        orchestrator = CyberShieldOrchestrator()
        orchestrator.run_full_pipeline()
        logger.info("✅ تم تحديث أوزان النموذج الفائز ونشرها في مسار الإنتاج models/champion_model.")
    else:
        logger.info("🟢 حالة التوزيعات مستقرة (Healthy Drift State)، لا توجد حاجة لإعادة التدريب حالياً.")


if __name__ == "__main__":
    check_and_retrain()