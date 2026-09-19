"""
MITRE ATT&CK Knowledge Base and TTP Mapping Module.
محرك مطابقة التهديدات السيبرانية المرصودة مع مصفوفة MITRE ATT&CK العالمية (Tactics & Techniques).
"""

from typing import Dict, Any, List  # استيراد أدوات التوثيق النوعي
from src.common.logger import get_logger  # استيراد المسجل

logger = get_logger(__name__)  # تهيئة المسجل لهذا الموديول


class MitreAttackMapper:
    """
    Maps detected anomalies, threat patterns, and protocol telemetry
    to formal MITRE ATT&CK Tactics, Techniques, and Mitigation references.
    """

    def __init__(self):
        # قاعدة المعرفة الخاصة بربط التهديدات مع معرفات MITRE ATT&CK
        self.attack_matrix_kb = {
            "SQL_INJECTION": {
                "tactic": "Initial Access / Defense Evasion",
                "technique_id": "T1190",
                "technique_name": "Exploit Public-Facing Application",
                "mitigation": "Use Parameterized Queries, Prepared Statements, and Web Application Firewall (WAF) rule sets."
            },
            "XSS_ATTACK": {
                "tactic": "Initial Access / Execution",
                "technique_id": "T1189",
                "technique_name": "Drive-by Compromise",
                "mitigation": "Implement Strict Content Security Policy (CSP), Context-Aware Output Encoding, and Input Sanitization."
            },
            "COMMAND_INJECTION": {
                "tactic": "Execution",
                "technique_id": "T1059",
                "technique_name": "Command and Scripting Interpreter",
                "mitigation": "Disable unnecessary shell execution APIs, run application under Least Privilege principle, isolate via containers."
            },
            "PORT_SCAN_RECON": {
                "tactic": "Discovery",
                "technique_id": "T1046",
                "technique_name": "Network Service Discovery",
                "mitigation": "Deploy Network Intrusion Detection (NIDS) rate-limiting, drop unsolicited SYN packets, hide open banners."
            },
            "DDOS_FLOOD": {
                "tactic": "Impact",
                "technique_id": "T1498",
                "technique_name": "Network Denial of Service",
                "mitigation": "Activate Anycast DNS routing, upstream BGP blackholing, and adaptive SYN-cookies thresholding."
            }
        }

    def map_threat(self, threat_category: str) -> Dict[str, str]:
        """
        استرجاع تفاصيل تقنية MITRE ATT&CK بناءً على فئة التهديد المرصودة.
        """
        category_key = threat_category.upper()
        if category_key in self.attack_matrix_kb:
            return self.attack_matrix_kb[category_key]

        # تصنيف افتراضي في حال كان الهجوم شذوذاً سلوكياً غير مخصص
        return {
            "tactic": "Lateral Movement / Anomalous Traffic",
            "technique_id": "T1071",
            "technique_name": "Application Layer Protocol",
            "mitigation": "Enforce Strict Network Segmentation, Zero-Trust network policies, and Deep Packet Inspection (DPI)."
        }