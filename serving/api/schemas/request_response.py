"""
Pydantic Schemas for the NIDS API.
نفس نموذج الطلب من Cell 4 في cybershield.ipynb.
"""

from typing import List, Optional

from pydantic import BaseModel


class DetectRequest(BaseModel):
    """طلب الكشف: سلسلة زمنية (SEQ_LEN × N_FEATURES) + بيانات المصدر."""
    features: List[List[float]]
    src_ip: Optional[str] = "unknown"
    dst_port: Optional[int] = 0
    protocol: Optional[str] = "TCP"
