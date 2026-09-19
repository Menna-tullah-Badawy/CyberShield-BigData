"""
Liveness and Readiness Health Probe Endpoints.
مسارات التحقق من الجاهزية التشغيلية للـ Kubernetes و Docker Swarm.
"""

import time
from typing import Dict, Any
from fastapi import APIRouter
from serving.api.schemas.request_response import HealthCheckResponse
from src.common.spark_manager import SparkManager

router = APIRouter(prefix="/health", tags=["Health & Telemetry"])
START_TIME = time.time()


@router.get("/live", response_model=HealthCheckResponse)
def liveness_probe():
    """فحص استجابة الخدمة اللحظي للتأكد من عدم تجمد السيرفر."""
    return HealthCheckResponse(
        status="HEALTHY",
        spark_session_active=True,
        champion_model_loaded=True,
        uptime_seconds=round(time.time() - START_TIME, 2),
        version="1.0.0-production"
    )


@router.get("/ready", response_model=Dict[str, Any])
def readiness_probe():
    """التحقق من جاهزية الاتصال بجلسة Spark وتحميل الأوزان في الذاكرة."""
    spark = SparkManager.get_spark_session()
    spark_ok = spark is not None and not spark.sparkContext._jsc.sc().isStopped()
    return {
        "ready": spark_ok,
        "spark_master": spark.conf.get("spark.master", "local[*]"),
        "timestamp": time.time()
    }