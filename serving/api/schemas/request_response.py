"""
Pydantic Schemas for FastAPI Request and Response Validation.
نماذج التحقق الصارم من صحة هياكل البيانات المدخلة والمخرجة لواجهات الاستدلال السريع.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class PacketFeaturesInput(BaseModel):
    """نموذج بيانات حزمة الشبكة المفردة للتحليل اللحظي."""
    timestamp: float = Field(default=1718000000.0, description="الطابع الزمني بالثواني")
    src_ip: str = Field(default="192.168.1.50", description="عنوان IP المصدر")
    dst_ip: str = Field(default="10.0.0.1", description="عنوان IP الوجهة")
    src_port: int = Field(default=49152, ge=0, le=65535, description="منفذ المصدر")
    dst_port: int = Field(default=80, ge=0, le=65535, description="منفذ الوجهة")
    protocol: int = Field(default=6, description="رقم البروتوكول (TCP=6, UDP=17)")
    packet_length: float = Field(default=512.0, ge=0.0, description="طول الحزمة بالبايت")
    time_delta: float = Field(default=0.001, ge=0.0, description="الفارق الزمني عن الحزمة السابقة")
    header_length: float = Field(default=32.0, ge=0.0, description="طول ترويسة الحزمة")
    window_size: float = Field(default=64240.0, ge=0.0, description="حجم نافذة الاستقبال TCP")
    flags: str = Field(default="SYN", description="رايات التحكم TCP Flags")
    payload: str = Field(default="EMPTY_PAYLOAD", description="الحمولة النصية أو الأوامر المستخرجة")


class BatchPacketInput(BaseModel):
    """نموذج استقبال دفعات متعددة من حزم الشبكة."""
    packets: List[PacketFeaturesInput] = Field(..., description="قائمة حزم الشبكة المراد فحصها")


class PredictionResponse(BaseModel):
    """نموذج الرد لتصنيف الحزمة المفردة."""
    is_malicious: bool = Field(..., description="هل الحزمة تمثل تهديداً أو هجوماً")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="درجة ثقة النموذج في التنبؤ")
    threat_label: str = Field(..., description="تصنيف الحالة (BENIGN أو ATTACK)")
    risk_level: str = Field(..., description="مستوى الخطر (LOW, MEDIUM, HIGH, CRITICAL)")
    inference_latency_ms: float = Field(..., description="زمن الاستدلال بالمللي ثانية")


class BatchPredictionResponse(BaseModel):
    """نموذج الرد لتصنيف دفعة الحزم المجمعة."""
    total_inspected: int
    malicious_detected: int
    threat_density_pct: float
    results: List[PredictionResponse]


class XAIExplanationResponse(BaseModel):
    """نموذج إرجاع تفسيرات الذكاء الاصطناعي للأدلة الجنائية."""
    prediction: PredictionResponse
    top_contributing_features: List[Dict[str, Any]]
    payload_attribution: Dict[str, Any]
    analyst_verdict: str


class HealthCheckResponse(BaseModel):
    """نموذج فحص صحة وجاهزية الخدمة."""
    status: str
    spark_session_active: bool
    champion_model_loaded: bool
    uptime_seconds: float
    version: str