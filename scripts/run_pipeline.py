"""
Batch Pipeline Execution Runner Script.
سكريبت تشغيل خط أنابيب معالجة الدفعات وتدريب النماذج وجدولة المهام الدورية (Cron / Airflow).
"""

import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestrator.pipeline_orchestrator import EndToEndPipelineOrchestrator
from src.data_pipeline.ingestion import DataIngestionEngine
from src.common.logger import get_logger
from src.common.spark_manager import SparkManager

logger = get_logger("CyberShield-BatchRunner")


def run_batch_job(data_source_path: str = None):
    logger.info("🚀 بدء تشغيل وظيفة معالجة الدفعات المجدولة (Batch Training & Evaluation Job)...")
    spark = SparkManager.get_spark_session()
    
    if data_source_path and os.path.exists(data_source_path):
        ingestion_engine = DataIngestionEngine()
        raw_df = ingestion_engine.ingest_csv_dataset(data_source_path)
    else:
        logger.info("⚡ استخدام عينة البيانات النموذجية الموزعة لاختبار المسار.")
        from main import CyberShieldOrchestrator
        orchestrator = CyberShieldOrchestrator()
        orchestrator.run_full_pipeline()
        return

    orchestrator = EndToEndPipelineOrchestrator()
    result = orchestrator.run_training_pipeline(raw_df)
    logger.info(f"✅ اكتملت المهمة الدورية بحالة: {result['workflow_summary']['overall_status']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CyberShield Batch Training Runner")
    parser.add_argument("--data_path", type=str, default=None, help="مسار ملف البيانات المدخل")
    args = parser.parse_args()
    run_batch_job(args.data_path)