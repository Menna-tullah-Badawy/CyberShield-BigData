"""
Master End-to-End Pipeline Orchestration Engine.
المنسق العام المسؤول عن الربط البرمجي الداخلي لكافة طبقات المنصة.
"""

from typing import Dict, Any
from src.orchestrator.workflow_manager import WorkflowManager
from src.quality.engine import DataQualityEngine
from src.cleaning.cleaning_pipeline import CleaningPipeline
from src.feature_engineering.pipeline_builder import FeaturePipelineBuilder
from src.feature_store.data_splitter import SparkDataSplitter
from src.models.model_selector import SparkModelSelector
from src.evaluation.model_validator import ModelValidator
from src.common.logger import get_logger

logger = get_logger(__name__)


class EndToEndPipelineOrchestrator:
    """
    Coordinates data flow from ingestion through cleaning, feature generation,
    model training, and validation using robust workflow state tracking.
    """

    def __init__(self):
        self.workflow = WorkflowManager("CyberShield_E2E_Pipeline")

    def run_training_pipeline(self, raw_df) -> Dict[str, Any]:
        """تشغيل خط أنابيب التدريب والتقييم المتكامل."""
        # 1. تدقيق الجودة
        quality_engine = DataQualityEngine()
        quality_report = self.workflow.execute_stage(
            "Data_Quality_Audit",
            quality_engine.run_all_checks,
            raw_df
        )

        # 2. التنظيف والتطهير
        cleaner = CleaningPipeline()
        cleaned_df = self.workflow.execute_stage(
            "Data_Cleaning",
            cleaner.run_cleaning_workflow,
            raw_df
        )

        # 3. هندسة الخصائص
        fe_builder = FeaturePipelineBuilder()
        engineered_df, _ = self.workflow.execute_stage(
            "Feature_Engineering",
            fe_builder.build_features,
            cleaned_df
        )

        # 4. التقسيم
        splitter = SparkDataSplitter()
        train_df, val_df, test_df = self.workflow.execute_stage(
            "Data_Splitting",
            splitter.stratified_split_by_label,
            engineered_df
        )

        # 5. اختيار النموذج الفائز
        selector = SparkModelSelector()
        champion_model, champ_name, _, _ = self.workflow.execute_stage(
            "Model_Selection",
            selector.benchmark_and_select_champion,
            train_df,
            val_df
        )

        # 6. التقييم والاعتماد
        validator = ModelValidator()
        eval_report = self.workflow.execute_stage(
            "Model_Validation",
            validator.validate_champion_model,
            champion_model,
            val_df,
            test_df,
            champ_name
        )

        return {
            "workflow_summary": self.workflow.get_summary(),
            "validation_report": eval_report
        }