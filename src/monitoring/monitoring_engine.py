"""
Enterprise MLOps Continuous Monitoring Orchestrator.
المنسق العام لطبقة المراقبة الذي يدمج كشف الانحراف مع القياسات التشغيلية ويصدر monitoring_drift_report.json.
"""

import os  # استيراد مكتبة التعامل مع نظام التشغيل
import json  # استيراد مكتبة JSON لحفظ التقارير
from typing import Dict, Any, List  # استيراد أدوات التوثيق
from pyspark.sql import DataFrame  # استيراد جدول بيانات سبارك
import configs.settings as cfg  # استيراد الإعدادات المركزية
from src.monitoring.drift_detector import DataDriftDetector  # استيراد كاشف الانحراف
from src.monitoring.performance_monitor import PerformanceMonitor  # استيراد مراقب الأداء
from src.monitoring.alert_manager import AlertManager  # استيراد مدير التنبيهات
from src.common.logger import get_logger  # استيراد نظام التسجيل
from src.common.decorators import time_execution  # استيراد مصمم قياس زمن التنفيذ

logger = get_logger(__name__)  # تهيئة المسجل


class MLOpsMonitoringEngine:
    """
    Coordinates end-to-end MLOps surveillance:
    - Statistical Data Drift Auditing (KS-Test & PSI)
    - Inference Latency and SLA Tracking
    - Alert Notification & Automated Retraining Triggering
    """

    def __init__(self):
        self.drift_detector = DataDriftDetector()  # تهيئة كاشف الانحراف
        self.performance_monitor = PerformanceMonitor()  # تهيئة مراقب الأداء
        self.alert_manager = AlertManager()  # تهيئة مدير التنبيهات

    @time_execution
    def run_production_audit(
        self,
        baseline_df: DataFrame,
        current_batch_df: DataFrame,
        features_to_monitor: List[str] = None,
        simulated_latencies_ms: List[float] = None
    ) -> Dict[str, Any]:
        """
        تشغيل دورة التدقيق والمراقبة الكاملة لدفعة البيانات الحالية وتصدير التقرير الرقابي.
        """
        logger.info("=" * 75)
        logger.info("🛰️ بدء تشغيل دورة المراقبة المستمرة وكشف انحراف البيانات (MLOps Monitoring)...")
        logger.info("=" * 75)

        # تحديد الخصائص المطلوب مراقبتها افتراضياً في حال لم تُمرر قائمة
        monitored_features = features_to_monitor or [
            "packet_length", "time_delta", "header_length", "window_size",
            "byte_rate_proxy", "header_ratio", "window_to_packet_ratio"
        ]

        # 1. تدقيق انحراف البيانات الإحصائي (Statistical Data Drift Audit)
        drift_results = self.drift_detector.audit_features_drift(
            baseline_df=baseline_df,
            current_df=current_batch_df,
            features_to_monitor=monitored_features
        )

        # 2. تقييم زمن الاستجابة ومستوى الخدمة (SLA Latency Tracking)
        latencies = simulated_latencies_ms or [12.5, 15.2, 14.1, 18.3, 45.0, 22.1, 16.4, 19.8, 25.0]
        latency_results = self.performance_monitor.compute_latency_metrics(latencies)

        # 3. فحص شروط إطلاق إعادة التدريب التلقائي
        trigger_retrain = self.alert_manager.should_trigger_retraining(drift_results)

        # 4. بناء التقرير الموحد
        monitoring_report = {
            "timestamp": "PRODUCTION_LIVE_CYCLE",
            "data_drift_audit": drift_results,
            "inference_performance_telemetry": latency_results,
            "retraining_signal": {
                "triggered": trigger_retrain,
                "action_required": "RE-RUN_PIPELINE" if trigger_retrain else "SYSTEM_HEALTHY_CONTINUE"
            }
        }

        # 5. حفظ التقرير في مسار reports/monitoring_drift_report.json
        report_path = os.path.join(cfg.REPORTS_DIR, "monitoring_drift_report.json")
        with open(report_path, "w", encoding="utf-8") as file:
            json.dump(monitoring_report, file, indent=4, ensure_ascii=False)

        logger.info("=" * 75)
        logger.info(f"✅ تم حفظ تقرير المراقبة بنجاح في: {report_path}")
        logger.info(f"🎯 حالة النظام: [{drift_results['overall_drift_status']}] | إشارة إعادة التدريب: [{trigger_retrain}]")
        logger.info("=" * 75)

        return monitoring_report