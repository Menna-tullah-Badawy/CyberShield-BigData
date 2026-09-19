"""
Enterprise MLOps Alerting and Automated Retraining Dispatcher.
مدير التنبيهات الموزعة وإطلاق إشارات إعادة التدريب التلقائي (Automated Retraining Trigger).
"""

from typing import Dict, Any  # استيراد أدوات التوثيق النوعي
from src.common.logger import get_logger  # استيراد المسجل

logger = get_logger(__name__)  # تهيئة المسجل


class AlertManager:
    """
    Handles alert dispatching (Console / Logs / Security Webhooks)
    and evaluates conditions to trigger automated model retraining pipelines.
    """

    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url  # رابط الويب هوك لإرسال التنبيهات (مثل Slack أو Discord أو SIEM)

    def dispatch_alert(self, level: str, title: str, details: Dict[str, Any]):
        """
        إرسال تنبيه أمني وفق درجة الخطورة (INFO, WARNING, CRITICAL).
        """
        log_message = f"📢 [ALERT - {level.upper()}] {title} | Details: {details}"

        if level.upper() == "CRITICAL":
            logger.critical(log_message)
        elif level.upper() == "WARNING":
            logger.warning(log_message)
        else:
            logger.info(log_message)

    def should_trigger_retraining(self, drift_summary: Dict[str, Any]) -> bool:
        """
        فحص تقرير الانحراف وتحديد ما إذا كان النظام بحاجة لإعادة التدريب التلقائي (Automated Retraining).
        """
        status = drift_summary.get("overall_drift_status", "HEALTHY")
        drifted_pct = drift_summary.get("drifted_features_percentage", 0.0)

        # يتم إطلاق إعادة التدريب إذا كانت الحالة حرجة أو تجاوزت نسبة الخصائص المنحرفة 25%
        if status == "CRITICAL_DRIFT" or drifted_pct >= 25.0:
            self.dispatch_alert(
                level="CRITICAL",
                title="إطلاق إعادة التدريب التلقائي للنموذج (Auto-Retraining Triggered)",
                details={
                    "reason": "تجاوز انحراف الخصائص الحد الأقصى للأمان",
                    "drifted_features_pct": drifted_pct
                }
            )
            return True

        return False