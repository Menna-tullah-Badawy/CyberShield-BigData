"""
Real-time Inference Telemetry and Performance Latency Monitor.
محرك قياس أداء الاستدلال اللحظي وتتبع زمن الاستجابة (P95/P99 Latency) ومعدل التدفق (Throughput).
"""

from typing import Dict, Any, List  # استيراد أدوات التوثيق
import numpy as np  # استيراد مكتبة العمليات الرياضية
from src.common.logger import get_logger  # استيراد نظام التسجيل

logger = get_logger(__name__)  # تهيئة المسجل


class PerformanceMonitor:
    """
    Monitors operational inference Service Level Agreements (SLAs),
    including P50, P95, and P99 latency percentiles and packet throughput.
    """

    def __init__(self, p95_latency_threshold_ms: float = 50.0):
        self.p95_latency_threshold_ms = p95_latency_threshold_ms  # الحد الأقصى المسموح به لزمن استجابة P95 بالمللي ثانية

    def compute_latency_metrics(self, latencies_ms: List[float]) -> Dict[str, Any]:
        """
        حساب المئين 50 و 95 و 99 لزمن استجابة التنبؤات والتحقق من الالتزام باتفاقية مستوى الخدمة (SLA).
        """
        if not latencies_ms:
            return {"status": "NO_DATA", "p95_breached": False}

        lat_array = np.array(latencies_ms)
        p50 = float(np.percentile(lat_array, 50))  # وسيط زمن الاستجابة (Median)
        p95 = float(np.percentile(lat_array, 95))  # 95% من الطلبات تنتهي قبل هذا الزمن
        p99 = float(np.percentile(lat_array, 99))  # 99% من الطلبات (Worst-case proxy)
        avg_lat = float(np.mean(lat_array))  # المتوسط

        p95_breached = bool(p95 > self.p95_latency_threshold_ms)

        if p95_breached:
            logger.warning(f"🚨 [Latency SLA Breach] P95 Latency ({p95:.2f}ms) تجاوز الحد المسموح به ({self.p95_latency_threshold_ms}ms)!")

        return {
            "mean_latency_ms": round(avg_lat, 2),
            "p50_latency_ms": round(p50, 2),
            "p95_latency_ms": round(p95, 2),
            "p99_latency_ms": round(p99, 2),
            "p95_sla_breached": p95_breached,
            "total_requests_profiled": len(latencies_ms)
        }

    def compute_throughput_metrics(self, total_packets: int, total_duration_seconds: float) -> Dict[str, Any]:
        """
        حساب معدل معالجة الحزم في الثانية (Inference Throughput).
        """
        duration = max(total_duration_seconds, 0.001)  # تجنب القسمة على صفر
        packets_per_sec = total_packets / duration  # حساب الحزم في الثانية

        return {
            "total_packets_served": total_packets,
            "duration_seconds": round(duration, 3),
            "throughput_pps": round(packets_per_sec, 2)
        }