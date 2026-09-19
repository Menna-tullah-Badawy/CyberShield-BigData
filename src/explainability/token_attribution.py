"""
Payload Token-Level Attribution and Cyber Threat Saliency Module.
محرك رصد وتحليل المقاطع النصية والأوامر الخبيثة داخل الحمولة (Payload Attribution).
"""

import re  # استيراد مكتبة التعبيرات النمطية (Regex) لرصد الأنماط
from typing import List, Dict, Any  # استيراد أدوات التوثيق للقوائم والقواميس
from src.common.logger import get_logger  # استيراد المسجل

logger = get_logger(__name__)  # تهيئة المسجل لهذا الموديول


class PayloadTokenAttribution:
    """
    Inspects raw packet payloads and maps contextual threat tokens (SQLi, XSS, Command Injection, Shellcodes)
    to explain why a payload was flagged as malicious by deep transformer models.
    """

    def __init__(self):
        # قاموس الأنماط والكلمات المفتاحية عالية الخطورة مع درجة الخطر المقابلة (0.0 إلى 1.0)
        self.signature_weights = {
            # أنماط حقن قواعد البيانات (SQL Injection)
            r"(?i)\b(union\s+select|select\s+.*\s+from|or\s+1=1|--|;\s*drop\s+table)\b": ("SQL_INJECTION", 0.95),
            # أنماط حقن النصوص البرمجية عبر المواقع (Cross-Site Scripting - XSS)
            r"(?i)(<script.*?>|javascript:|onerror\s*=|onload\s*=|alert\(.*?\))": ("XSS_ATTACK", 0.90),
            # أنماط حقن أوامر النظام (OS Command Injection / Path Traversal)
            r"(?i)(\.\./\.\./|/etc/passwd|/bin/sh|/bin/bash|cmd\.exe|powershell)": ("COMMAND_INJECTION", 0.98),
            # أنماط استطلاع وتجاوز البروتوكولات (Protocol Exploitation)
            r"(?i)\b(admin|root|passwd|shadow|eval\(|base64_decode)\b": ("PRIVILEGE_RECON", 0.75)
        }

    def explain_payload(self, raw_payload: str) -> Dict[str, Any]:
        """
        فحص نص الحمولة وتحديد الكلمات والرموز المشبوهة وتوليد خريطة الإسناد الدلالي (Saliency Map).
        """
        if not raw_payload or raw_payload == "EMPTY_PAYLOAD":  # التحقق من أن الحمولة غير فارغة
            return {
                "payload_inspected": "EMPTY_PAYLOAD",
                "is_suspicious": False,
                "threat_score": 0.0,
                "detected_threat_patterns": [],
                "highlighted_tokens": []
            }

        detected_patterns = []  # قائمة لتجميع التهديدات المرصودة
        highlighted_tokens = []  # قائمة لتجميع الرموز الخطرة
        max_threat_score = 0.0  # أعلى درجة خطورة مرصودة

        # فحص النص بالبحث عن الأنماط والتعبيرات النمطية المعروفة
        for pattern_regex, (threat_type, score) in self.signature_weights.items():
            matches = re.finditer(pattern_regex, raw_payload)  # البحث عن كافة التطابقات
            for match in matches:  # المرور على كل تطابق
                token_text = match.group(0)  # النص المطابق للنمط الخبيث
                max_threat_score = max(max_threat_score, score)  # تحديث أعلى درجة خطر
                
                detected_patterns.append(threat_type)  # إضافة نوع الهجوم
                highlighted_tokens.append({
                    "token": token_text,  # المقطع الخبيث
                    "threat_category": threat_type,  # تصنيف التهديد
                    "attribution_score": score,  # وزن التأثير في القرار
                    "start_pos": match.start(),  # موضع بداية الكلمة في النص
                    "end_pos": match.end()  # موضع نهاية الكلمة
                })

        # إزالة التكرارات في أنواع الأنماط المرصودة
        unique_patterns = list(set(detected_patterns))  # تصفية الفئات الفريدة
        is_suspicious = len(highlighted_tokens) > 0  # تحديد ما إذا كانت الحمولة مشبوهة قطيعاً

        return {
            "payload_inspected": raw_payload[:200],  # اقتطاع أول 200 حرف للعرض السريع
            "is_suspicious": is_suspicious,  # مؤشر الاشتباه
            "threat_score": round(float(max_threat_score), 4),  # درجة الخطر الكلية
            "detected_threat_patterns": unique_patterns,  # قائمة التهديدات المرصودة
            "highlighted_tokens": highlighted_tokens  # قائمة الرموز المشبوهة المفسرة
        }