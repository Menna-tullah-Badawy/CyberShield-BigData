"""
==============================================================================
CyberShield-BigData: Enterprise Hybrid NIDS & MLOps Orchestrator
==============================================================================
"""

import sys
import os
import json
import argparse
from datetime import datetime
from pathlib import Path

# ضبط ترميز الإخراج ليتوافق مع مختلف البيئات
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# إضافة مسار المشروع الأساسي لضمان استيراد الحزم
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import configs.settings as cfg
from src.common.spark_manager import SparkManager
from src.common.logger import get_logger
from src.data_pipeline.ingestion import DataIngestionEngine
from src.data_pipeline.spark_pcap_loader import SparkPCAPLoader
from src.quality.engine import DataQualityEngine
from src.cleaning.cleaning_pipeline import CleaningPipeline
from src.analysis.analyzer import SparkEDAOrchestrator
from src.feature_engineering.pipeline_builder import FeaturePipelineBuilder
from src.feature_store.transformations import FeatureStoreManager
from src.feature_store.data_splitter import SparkDataSplitter
from src.models.model_selector import SparkModelSelector
from src.evaluation.model_validator import ModelValidator
from src.explainability.xai_engine import ExplainabilityEngine
from src.monitoring.monitoring_engine import MLOpsMonitoringEngine
from src.genai_reporting.report_generator import IncidentReportGenerator

logger = get_logger("CyberShield-Master")


class CheckpointManager:
    """إدارة حفظ واستعادة حالة تنفيذ المراحل لتجنب إعادة التنفيذ."""
    
    def __init__(self, checkpoint_dir: str = "data/checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "pipeline_state.json"
        self.state = self._load_state()
    
    def _load_state(self) -> dict:
        """تحميل حالة المراحل السابقة."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warn(f"⚠️ فشل تحميل الـ checkpoint: {e}")
                return {}
        return {}
    
    def _save_state(self):
        """حفظ الحالة الحالية."""
        try:
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"❌ فشل حفظ الـ checkpoint: {e}")
    
    def is_stage_completed(self, stage_name: str) -> bool:
        """فحص ما إذا كانت المرحلة مكتملة."""
        return self.state.get(stage_name, {}).get('completed', False)
    
    def mark_stage_completed(self, stage_name: str, metadata: dict = None):
        """تعليم المرحلة كمكتملة."""
        self.state[stage_name] = {
            'completed': True,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self._save_state()
        logger.info(f"✅ Checkpoint saved: {stage_name}")
    
    def get_stage_metadata(self, stage_name: str) -> dict:
        """استرجاع معلومات المرحلة المحفوظة."""
        return self.state.get(stage_name, {}).get('metadata', {})
    
    def reset_from_stage(self, stage_name: str):
        """إعادة تعيين المراحل بدءاً من مرحلة معينة."""
        stages = ['ingestion', 'quality', 'cleaning', 'eda', 'feature_engineering', 
                  'feature_store', 'model_selection', 'validation', 'xai', 'monitoring']
        
        if stage_name in stages:
            idx = stages.index(stage_name)
            for stage in stages[idx:]:
                if stage in self.state:
                    del self.state[stage]
            self._save_state()
            logger.info(f"🔄 Reset pipeline from stage: {stage_name}")
    
    def clear_all(self):
        """مسح جميع الـ checkpoints."""
        self.state = {}
        self._save_state()
        logger.info("🗑️ All checkpoints cleared")


def print_banner():
    """طباعة شعار النظام في واجهة الأوامر."""
    banner = """
    =========================================================================
      ______      __                _____ _     _      _     _ 
     / ____/_  __/ /_  ___  _____  / ___/| |   (_)__  | |   / /
    / /   / / / / __ \/ _ \/ ___/  \__ \ | |  / / _ \ | |  / / 
   / /___/ /_/ / /_/ /  __/ /      ___/ / | | / /  __/ | | / /  
   \____/\__, /_.___/\___/_/      /____/  |_|/_/\___/  |___/   
        /____/  Enterprise Distributed Hybrid NIDS (FAANG-MLOps Ready)
    =========================================================================
    """
    print(banner)


class CyberShieldOrchestrator:
    def __init__(self, use_checkpoints: bool = True):
        # بدء جلسة Spark المركزية
        self.spark = SparkManager.get_spark_session()
        self.checkpoint_mgr = CheckpointManager() if use_checkpoints else None
        logger.info("🚀 Apache Spark distributed session initialized successfully.")
        
        if self.checkpoint_mgr:
            logger.info("💾 Checkpoint system enabled - will skip completed stages")

    def run_full_pipeline(self, raw_data_path: str = None):
        """تنفيذ جميع طبقات خط الأنابيب الـ 10 بالتتابع مع دعم Checkpointing."""
        start_time = datetime.now()
        logger.info("⚡ [Pipeline] Starting full end-to-end execution pipeline...")

        # متغيرات لتخزين نتائج المراحل
        raw_df = None
        cleaned_df = None
        engineered_df = None
        scaler_model = None
        train_df = None
        val_df = None
        test_df = None
        champion_model = None
        champion_name = None
        best_score = None
        malicious_sample = None
        xai_explanation = None

        # ========== [1/10] Ingestion Layer ==========
        stage_name = 'ingestion'
        if self.checkpoint_mgr and self.checkpoint_mgr.is_stage_completed(stage_name):
            logger.info(f"\n>>> [1/10] ⏭️  Ingestion Layer - SKIPPED (already completed)")
            # تحميل البيانات من الكاش
            ingestion_engine = DataIngestionEngine()
            raw_df = ingestion_engine.load_or_fetch_dataset(sample_fraction=0.01)
        else:
            logger.info("\n>>> [1/10] Ingestion Layer...")
            ingestion_engine = DataIngestionEngine()

            if raw_data_path and (raw_data_path.lower() == "s3" or "s3a://" in raw_data_path):
                raw_df = ingestion_engine.load_or_fetch_dataset(sample_fraction=0.01)
            elif raw_data_path and os.path.exists(raw_data_path):
                raw_df = ingestion_engine.ingest_csv_dataset(raw_data_path)
            else:
                raw_df = ingestion_engine.generate_synthetic_stream()
            
            if self.checkpoint_mgr:
                self.checkpoint_mgr.mark_stage_completed(stage_name, {
                    'row_count': raw_df.count(),
                    'columns': len(raw_df.columns)
                })

        # ========== [2/10] Data Quality Gate ==========
        stage_name = 'quality'
        if self.checkpoint_mgr and self.checkpoint_mgr.is_stage_completed(stage_name):
            logger.info(f"\n>>> [2/10] ⏭️  Data Quality Gate - SKIPPED (already completed)")
        else:
            logger.info("\n>>> [2/10] Data Quality Gate...")
            quality_engine = DataQualityEngine()
            quality_engine.run_all_checks(raw_df, dataset_name="Network_Raw_Stream")
            
            if self.checkpoint_mgr:
                self.checkpoint_mgr.mark_stage_completed(stage_name)

        # ========== [3/10] Data Sanitization Pipeline ==========
        stage_name = 'cleaning'
        if self.checkpoint_mgr and self.checkpoint_mgr.is_stage_completed(stage_name):
            logger.info(f"\n>>> [3/10] ⏭️  Data Sanitization - SKIPPED (already completed)")
            # إعادة تشغيل التنظيف للحصول على cleaned_df
            cleaning_pipeline = CleaningPipeline()
            cleaned_df = cleaning_pipeline.run_cleaning_workflow(raw_df)
        else:
            logger.info("\n>>> [3/10] Data Sanitization Pipeline...")
            cleaning_pipeline = CleaningPipeline()
            cleaned_df = cleaning_pipeline.run_cleaning_workflow(raw_df)
            
            if self.checkpoint_mgr:
                self.checkpoint_mgr.mark_stage_completed(stage_name, {
                    'row_count': cleaned_df.count()
                })

        # ========== [4/10] Exploratory Data Analysis ==========
        stage_name = 'eda'
        if self.checkpoint_mgr and self.checkpoint_mgr.is_stage_completed(stage_name):
            logger.info(f"\n>>> [4/10] ⏭️  EDA - SKIPPED (already completed)")
        else:
            logger.info("\n>>> [4/10] Exploratory Data Analysis (EDA)...")
            eda_orchestrator = SparkEDAOrchestrator(cleaned_df)
            eda_orchestrator.run_full_analysis()
            
            if self.checkpoint_mgr:
                self.checkpoint_mgr.mark_stage_completed(stage_name)

        # ========== [5/10] Distributed Feature Engineering ==========
        stage_name = 'feature_engineering'
        if self.checkpoint_mgr and self.checkpoint_mgr.is_stage_completed(stage_name):
            logger.info(f"\n>>> [5/10] ⏭️  Feature Engineering - SKIPPED (already completed)")
            # تحميل من Feature Store
            fs_manager = FeatureStoreManager()
            try:
                engineered_df = fs_manager.load_features_from_store("nids_features_latest")
                logger.info(f"✅ Loaded engineered features from store")
            except:
                # إذا لم يكن موجود، إعادة التنفيذ
                logger.warn("⚠️ Could not load from store, re-running feature engineering...")
                fe_builder = FeaturePipelineBuilder()
                engineered_df, scaler_model = fe_builder.build_features(cleaned_df)
        else:
            logger.info("\n>>> [5/10] Distributed Feature Engineering...")
            fe_builder = FeaturePipelineBuilder()
            engineered_df, scaler_model = fe_builder.build_features(cleaned_df)
            
            if self.checkpoint_mgr:
                self.checkpoint_mgr.mark_stage_completed(stage_name, {
                    'feature_count': len(engineered_df.columns)
                })

        # ========== [6/10] Feature Store & Stratified Data Splitting ==========
        stage_name = 'feature_store'
        if self.checkpoint_mgr and self.checkpoint_mgr.is_stage_completed(stage_name):
            logger.info(f"\n>>> [6/10] ⏭️  Feature Store & Splitting - SKIPPED (already completed)")
            # تحميل التقسيمات
            fs_manager = FeatureStoreManager()
            try:
                train_df = fs_manager.load_features_from_store("train_split")
                val_df = fs_manager.load_features_from_store("val_split")
                test_df = fs_manager.load_features_from_store("test_split")
                logger.info(f"✅ Loaded train/val/test splits from store")
            except:
                logger.warn("⚠️ Could not load splits, re-running...")
                fs_manager.save_features_to_store(engineered_df, table_name="nids_features_latest")
                splitter = SparkDataSplitter()
                train_df, val_df, test_df = splitter.stratified_split_by_label(engineered_df, label_col="label")
                # حفظ التقسيمات
                fs_manager.save_features_to_store(train_df, table_name="train_split")
                fs_manager.save_features_to_store(val_df, table_name="val_split")
                fs_manager.save_features_to_store(test_df, table_name="test_split")
        else:
            logger.info("\n>>> [6/10] Feature Store & Stratified Data Splitting...")
            fs_manager = FeatureStoreManager()
            fs_manager.save_features_to_store(engineered_df, table_name="nids_features_latest")
            
            splitter = SparkDataSplitter()
            train_df, val_df, test_df = splitter.stratified_split_by_label(engineered_df, label_col="label")
            
            # حفظ التقسيمات
            fs_manager.save_features_to_store(train_df, table_name="train_split")
            fs_manager.save_features_to_store(val_df, table_name="val_split")
            fs_manager.save_features_to_store(test_df, table_name="test_split")
            
            if self.checkpoint_mgr:
                self.checkpoint_mgr.mark_stage_completed(stage_name, {
                    'train_count': train_df.count(),
                    'val_count': val_df.count(),
                    'test_count': test_df.count()
                })

        # ========== [7/10] Distributed Model Selection ==========
        stage_name = 'model_selection'
        if self.checkpoint_mgr and self.checkpoint_mgr.is_stage_completed(stage_name):
            logger.info(f"\n>>> [7/10] ⏭️  Model Selection - SKIPPED (already completed)")
            # تحميل النموذج المحفوظ
            metadata = self.checkpoint_mgr.get_stage_metadata(stage_name)
            champion_name = metadata.get('champion_name', 'LogisticRegression')
            best_score = metadata.get('best_score', 0.0)
            logger.info(f"✅ Using saved champion: {champion_name} (score: {best_score:.4f})")
            # سنحتاج لإعادة التدريب أو التحميل من export/models
        else:
            logger.info("\n>>> [7/10] Distributed Model Selection & Hyperparameter Tuning...")
            selector = SparkModelSelector(k_folds=3, metric_name="areaUnderROC")
            champion_model, champion_name, best_score, _ = selector.benchmark_and_select_champion(
                train_df=train_df,
                val_df=val_df,
                features_col="final_features",
                label_col="label"
            )
            
            if self.checkpoint_mgr:
                self.checkpoint_mgr.mark_stage_completed(stage_name, {
                    'champion_name': champion_name,
                    'best_score': float(best_score) if best_score else 0.0
                })

        # ========== [8/10] Model Validation ==========
        stage_name = 'validation'
        if self.checkpoint_mgr and self.checkpoint_mgr.is_stage_completed(stage_name):
            logger.info(f"\n>>> [8/10] ⏭️  Model Validation - SKIPPED (already completed)")
        else:
            logger.info("\n>>> [8/10] Model Validation & Optimal Threshold Tuning...")
            if champion_model:  # فقط إذا كان النموذج موجود
                validator = ModelValidator()
                validator.validate_champion_model(
                    champion_model=champion_model,
                    val_df=val_df,
                    test_df=test_df,
                    model_name=champion_name
                )
                
                if self.checkpoint_mgr:
                    self.checkpoint_mgr.mark_stage_completed(stage_name)

        # ========== [9/10] Explainable AI (XAI) ==========
        stage_name = 'xai'
        if self.checkpoint_mgr and self.checkpoint_mgr.is_stage_completed(stage_name):
            logger.info(f"\n>>> [9/10] ⏭️  XAI - SKIPPED (already completed)")
        else:
            logger.info("\n>>> [9/10] Explainable AI (XAI) Forensic Engine...")
            try:
                xai_engine = ExplainabilityEngine()
                malicious_sample = cleaned_df.filter(cleaned_df.label == 1).first()
                if malicious_sample:
                    malicious_sample = malicious_sample.asDict()
                    if champion_model:
                        xai_explanation = xai_engine.generate_incident_explanation(champion_model, malicious_sample)
                
                if self.checkpoint_mgr:
                    self.checkpoint_mgr.mark_stage_completed(stage_name)
            except Exception as e:
                logger.warn(f"⚠️ XAI stage failed: {e}")

        # ========== [10/10] MLOps Drift Monitoring & GenAI ==========
        stage_name = 'monitoring'
        if self.checkpoint_mgr and self.checkpoint_mgr.is_stage_completed(stage_name):
            logger.info(f"\n>>> [10/10] ⏭️  Monitoring & Reporting - SKIPPED (already completed)")
        else:
            logger.info("\n>>> [10/10] MLOps Drift Monitoring & GenAI Incident Reporting...")
            try:
                monitoring_engine = MLOpsMonitoringEngine()
                monitoring_engine.run_production_audit(baseline_df=train_df, current_batch_df=test_df)

                if malicious_sample and xai_explanation:
                    report_generator = IncidentReportGenerator()
                    report_generator.create_incident_report(
                        packet_record=malicious_sample,
                        xai_explanation=xai_explanation,
                        prediction_score=0.99
                    )
                
                if self.checkpoint_mgr:
                    self.checkpoint_mgr.mark_stage_completed(stage_name)
            except Exception as e:
                logger.warn(f"⚠️ Monitoring stage failed: {e}")

        total_duration = datetime.now() - start_time
        logger.info("=" * 75)
        logger.info(f"🎉 CyberShield-BigData pipeline completed successfully in: {total_duration}")
        logger.info("=" * 75)


def main():
    print_banner()
    parser = argparse.ArgumentParser(description="CyberShield-BigData Pipeline Orchestrator")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "quality", "clean", "eda", "train", "monitor", "report"])
    parser.add_argument("--data_path", type=str, default=None, help="Path to raw network dataset (.pcap or .csv)")
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable checkpoint system (re-run all stages)")
    parser.add_argument("--reset-from", type=str, default=None, 
                       help="Reset pipeline from this stage (ingestion, quality, cleaning, eda, etc.)")
    parser.add_argument("--clear-checkpoints", action="store_true", help="Clear all checkpoints and exit")
    args = parser.parse_args()

    # معالجة أوامر الـ checkpoint
    if args.clear_checkpoints:
        checkpoint_mgr = CheckpointManager()
        checkpoint_mgr.clear_all()
        logger.info("✅ All checkpoints cleared successfully")
        return
    
    if args.reset_from:
        checkpoint_mgr = CheckpointManager()
        checkpoint_mgr.reset_from_stage(args.reset_from)
        logger.info(f"✅ Pipeline reset from stage: {args.reset_from}")
        return

    # تشغيل الـ Pipeline
    use_checkpoints = not args.no_checkpoint
    orchestrator = CyberShieldOrchestrator(use_checkpoints=use_checkpoints)
    orchestrator.run_full_pipeline(args.data_path)


if __name__ == "__main__":
    main()