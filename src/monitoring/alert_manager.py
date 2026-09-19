"""
Alert Manager — Telegram SOC alerts + incident registry.
منطق Cell 7 (المعاد بناؤه): dispatch_telegram_alert + send_telegram_document + active_incidents.

الإعداد عبر: CYBERSHIELD_BOT_TOKEN / CYBERSHIELD_ADMIN_CHAT_ID
"""

import os
from typing import Dict

import requests

BOT_TOKEN = os.environ.get("CYBERSHIELD_BOT_TOKEN", "")
ADMIN_CHAT_ID = os.environ.get("CYBERSHIELD_ADMIN_CHAT_ID", "")

# سجل الحوادث النشطة (نفس اسم النوت بوك)
active_incidents: Dict[str, dict] = {}


class AlertManager:
    """إرسال تنبيهات الحوادث عبر تليجرام (وضع محاكاة بدون توكن)."""

    def __init__(self, bot_token: str = BOT_TOKEN, chat_id: str = ADMIN_CHAT_ID):
        self.bot_token = bot_token or BOT_TOKEN
        self.chat_id = chat_id or ADMIN_CHAT_ID

    def dispatch_alert(self, incident: dict) -> bool:
        """إرسال تنبيه الحادثة مع أزرار (حظر/تقرير/تجاهل) — نفس callback_data في Cell 7."""
        active_incidents[incident["src_ip"]] = incident
        text = (
            f"🚨 <b>تنبيه اختراق جديد!</b>\n"
            f"━━━━━━━━━━━━\n"
            f"🛡️ <b>الهجوم:</b> {incident['threat_name']}\n"
            f"🏷️ <b>التصنيف:</b> {incident['threat_id']}\n"
            f"🎯 <b>نسبة التأكد:</b> {incident['confidence']:.1f}%\n"
            f"🌐 <b>المصدر:</b> <code>{incident['src_ip']}</code>\n"
            f"🔒 <b>الحالة:</b> {incident['status']}"
        )
        keyboard = {"inline_keyboard": [[
            {"text": "🛑 حظر الـ IP فوراً", "callback_data": f"block_{incident['src_ip']}"},
            {"text": "📄 التقرير", "callback_data": f"report_{incident['src_ip']}"},
            {"text": "❌ تجاهل", "callback_data": f"ignore_{incident['src_ip']}"},
        ]]}
        if not self.bot_token or not self.chat_id:
            print(f"[SIMULATE] Telegram alert → chat {self.chat_id or '?'}: "
                  f"{incident['threat_name']} ({incident['src_ip']})")
            return False
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
                json={"chat_id": self.chat_id, "text": text,
                      "parse_mode": "HTML", "reply_markup": keyboard},
                timeout=20)
            return bool(r.ok)
        except Exception as e:
            print(f"⚠️ Telegram send failed: {e}")
            return False

    def send_document(self, pdf_path: str, caption: str) -> bool:
        """إرسال ملف PDF عبر sendDocument (وضع محاكاة بدون توكن)."""
        if not self.bot_token or not self.chat_id:
            print(f"[SIMULATE] Telegram document → chat {self.chat_id or '?'}: {pdf_path}")
            return False
        try:
            with open(pdf_path, "rb") as f:
                r = requests.post(
                    f"https://api.telegram.org/bot{self.bot_token}/sendDocument",
                    data={"chat_id": self.chat_id, "caption": caption, "parse_mode": "HTML"},
                    files={"document": f}, timeout=30)
            return bool(r.ok)
        except Exception as e:
            print(f"⚠️ Telegram document failed: {e}")
            return False
