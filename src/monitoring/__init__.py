"""
Continuous Monitoring and MLOps Package.
تصدير كلاسات كشف انحراف التوزيعات، مراقبة أداء الاستدلال، ونظام التنبيهات الآلية.
"""

from src.monitoring.drift_detector import DataDriftDetector  # استيراد محرك كشف انحراف البيانات
from src.monitoring.performance_monitor import PerformanceMonitor  # استيراد محرك مراقبة الأداء
from src.monitoring.alert_manager import AlertManager  # استيراد مدير التنبيهات
from src.monitoring.monitoring_engine import MLOpsMonitoringEngine  # استيراد المنسق العام للمراقبة

# إتاحة الكلاسات للاستدعاء المباشر
__all__ = [
    "DataDriftDetector",
    "PerformanceMonitor",
    "AlertManager",
    "MLOpsMonitoringEngine"
]