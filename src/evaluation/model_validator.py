"""
Enterprise Model Validation and Verification Orchestrator.
المنسق الشامل لتقييم النموذج الفائز على مجموعة الاختبار المعزولة وتصدير تقرير الاعتماد النهائي.
"""

import os  # استيراد مكتبة التعامل مع نظام التشغيل
import json  # استيراد مكتبة JSON لحفظ التقارير
from typing import Dict, Any  # استيراد أدوات التوثيق
from pyspark.sql import DataFrame  # استيراد جدول بيانات سبارك
import configs.settings as cfg  # استيراد الإعدادات المركزية
from src.evaluation.metrics import CyberEvaluationMetrics  # استيراد محرك المقاييس
from src.evaluation.threshold_optimizer import ThresholdOptimizer  # استيراد محرك تحسين العتبة
from src.common.logger import get_logger  # استيراد المسجل
from src.common.decorators import time_execution  # استيراد مصمم قياس الزمن

logger = get_logger(__name__)  # تهيئة المسجل


class ModelValidator:
    """
    Executes end-to-end model validation, threshold tuning verification,
    and exports the formal audit report: reports/model_evaluation_report.json.
    """

    def __init__(self):
        self.metrics_engine = CyberEvaluationMetrics()  # تهيئة محرك المقاييس
        self.threshold_optimizer = ThresholdOptimizer()  # تهيئة محرك تحسين العتبة

    @time_execution  # قياس زمن تقرير الاعتماد بالكامل
    def validate_champion_model(
        self,
        champion_model: Any,
        val_df: DataFrame,
        test_df: DataFrame,
        model_name: str = "Champion_GBT_Classifier"
    ) -> Dict[str, Any]:
        """
        تقييم النموذج الفائز، تحسين عتبة القرار على مجموعة التحقق، ثم تقييم الأداء النهائي على مجموعة الاختبار المعزولة تماماً.
        """
        logger.info("=" * 75)
        logger.info(f"🛡️ بدء مرحلة التحقق والاعتماد النهائي للنموذج الفائز: [{model_name}] 🛡️")
        logger.info("=" * 75)

        # 1. توليد التنبؤات على مجموعة التحقق (Validation Set) لتحسين العتبة
        logger.info("🔍 [1/3] تقييم مجموعة التحقق وضبط عتبة اتخاذ القرار...")
        val_predictions = champion_model.transform(val_df)
        optimal_threshold_info = self.threshold_optimizer.find_optimal_threshold(val_predictions, metric_target="f1")
        optimal_th = optimal_threshold_info["optimal_threshold"]

        # 2. توليد التنبؤات على مجموعة الاختبار المعزولة تماماً (Test Set)
        logger.info("🔍 [2/3] توليد التنبؤات على مجموعة الاختبار المعزولة (Test Set)...")
        raw_test_predictions = champion_model.transform(test_df)

        # تطبيق العتبة الافتراضية (0.50) لحساب الأداء القياسي للمقارنة
        default_metrics = self.metrics_engine.compute_all_metrics(raw_test_predictions)

        # 3. تطبيق العتبة المحسنة (Optimal Threshold) على مجموعة الاختبار
        logger.info(f"🔍 [3/3] تطبيق العتبة المحسنة ({optimal_th}) على مجموعة الاختبار...")
        optimized_test_predictions = self.threshold_optimizer.apply_custom_threshold(raw_test_predictions, threshold=optimal_th)
        optimized_metrics = self.metrics_engine.compute_all_metrics(optimized_test_predictions)

        # 4. بناء تقرير الاعتماد والتقييم المؤسسي المتكامل
        evaluation_report = {
            "model_metadata": {
                "model_name": model_name,
                "test_dataset_size": test_df.count(),
                "validation_dataset_size": val_df.count(),
                "status": "APPROVED_FOR_PRODUCTION" if optimized_metrics["f1_score"] >= 0.85 else "NEEDS_IMPROVEMENT"
            },
            "threshold_optimization": optimal_threshold_info,
            "baseline_metrics_default_threshold_0_5": default_metrics,
            "final_metrics_optimized_threshold": optimized_metrics
        }

        # 5. حفظ التقرير النهائي في مسار reports/model_evaluation_report.json
        report_path = os.path.join(cfg.REPORTS_DIR, "model_evaluation_report.json")
        with open(report_path, "w", encoding="utf-8") as file:
            json.dump(evaluation_report, file, indent=4, ensure_ascii=False)

        logger.info("=" * 75)
        logger.info(f"✅ تم حفظ تقرير اعتماد وتقييم النموذج بنجاح في: {report_path}")
        logger.info(f"🎯 حالة الاعتماد: [{evaluation_report['model_metadata']['status']}] | Final F1-Score: {optimized_metrics['f1_score']}")
        logger.info("=" * 75)

        return evaluation_report  # إرجاع التقرير النهائي