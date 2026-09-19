"""
SOC Telegram Live Monitor.
نقل Cell 7 من cybershield.ipynb + إعادة بناء الخلايا المحذوفة
(analyze_flow_with_ai / dispatch_telegram_alert / generate_incident_pdf /
 send_telegram_document) من سياق الخلية والـ Outputs المحفوظة.

الإعداد عبر متغيرات البيئة (لا تُكتب التوكنز في الكود أبداً):
    CYBERSHIELD_BOT_TOKEN / CYBERSHIELD_ADMIN_CHAT_ID
"""

import asyncio
import os
import random
from typing import Dict, Optional, Sequence

import numpy as np
import requests
import torch

from src.nids.threat_kb import HybridThreatRetriever
from src.nids.xai import extract_xai

BOT_TOKEN = os.environ.get("CYBERSHIELD_BOT_TOKEN", "")
ADMIN_CHAT_ID = os.environ.get("CYBERSHIELD_ADMIN_CHAT_ID", "")

# سجل الحوادث النشطة (نفس اسم النوت بوك)
active_incidents: Dict[str, dict] = {}

# أسماء العرض للتهديدات (مُعادة البناء من Outputs النوت بوك المحفوظة)
DISPLAY_NAMES = {
    "MITRE-T1498": "Denial of Service (SYN/UDP Flood)",
    "MITRE-T1059": "Command & Scripting Reverse Shell",
    "MITRE-T1046": "Port Scan & Network Discovery",
    "MITRE-T1041": "Data Exfiltration Over C2",
    "MITRE-T1110": "SSH Brute Force Attack",
    "MITRE-T1071": "DNS Tunneling (C2 Channel)",
}

DEMO_SUBNET = "192.168.1"


def _demo_src_ip() -> str:
    return f"{DEMO_SUBNET}.{random.randint(2, 254)}"


def analyze_flow_with_ai(
    flow_seq: np.ndarray,
    model: torch.nn.Module,
    scaler,
    feature_names: Sequence[str],
    retriever: HybridThreatRetriever,
    device: torch.device,
    src_ip: Optional[str] = None,
    threshold: float = 0.03,
) -> Optional[dict]:
    """
    تحليل Flow واحد بالموديل + XAI + RAG وإرجاع قاموس الحادثة
    (إعادة بناء الدالة المحذوفة من النوت بوك — نفس المفاتيح المستخدمة في Cell 7).
    """
    flat = np.asarray(flow_seq, dtype=np.float32).reshape(-1, len(feature_names))
    scaled = np.clip(scaler.transform(flat), -15, 15).reshape(
        1, flow_seq.shape[0], len(feature_names))
    x = torch.tensor(scaled, dtype=torch.float32, device=device)
    with torch.no_grad():
        ap = float(torch.softmax(model(x), 1)[0, 1].cpu().numpy())
    if ap < threshold:
        return None

    xf = extract_xai(model, x.clone(), feature_names, n=5)
    qt = " ".join(f["name"] for f in xf).lower().replace("_", " ")
    rag = retriever.retrieve(qt)
    doc = rag["doc"]
    threat_id = doc["threat_id"]
    ip = src_ip or _demo_src_ip()
    return {
        "threat_name": DISPLAY_NAMES.get(threat_id, doc["title"]),
        "threat_id": threat_id,
        "tactic": doc["tactic"],
        "confidence": round(ap * 100.0, 1),
        "probability": round(ap, 4),
        "src_ip": ip,
        "status": "DETECTED",
        "playbook": doc["playbook"],
        "indicators": doc.get("indicators", ""),
        "iptables": doc.get("iptables", []),
        "xai_features": xf,
        "hybrid_score": round(rag["hybrid_score"], 4),
    }


def dispatch_telegram_alert(
    incident: dict,
    bot_token: str = BOT_TOKEN,
    chat_id: str = ADMIN_CHAT_ID,
) -> bool:
    """
    إرسال تنبيه الحادثة مع أزرار (حظر/تقرير/تجاهل) — نفس callback_data
    التي ينتظرها handle_security_action: block_/report_/ignore_.
    بدون توكن يعمل في وضع المحاكاة (طباعة فقط) حتى لا يتعطل الديمو.
    """
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
    if not bot_token or not chat_id:
        print(f"[SIMULATE] Telegram alert → chat {chat_id or '?'}: {incident['threat_name']} "
              f"({incident['src_ip']})")
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{bot_token}/sendMessage",
                          json={"chat_id": chat_id, "text": text,
                                "parse_mode": "HTML", "reply_markup": keyboard},
                          timeout=20)
        return bool(r.ok)
    except Exception as e:
        print(f"⚠️ Telegram send failed: {e}")
        return False


def generate_incident_pdf(incident: dict, pdf_name: str) -> str:
    """تقرير PDF للحادثة الواحدة (reportlab + بديل matplotlib عند غيابها)."""
    os.makedirs(os.path.dirname(os.path.abspath(pdf_name)), exist_ok=True)
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer, Table,
                                        TableStyle)

        doc = SimpleDocTemplate(pdf_name, pagesize=A4)
        styles = getSampleStyleSheet()
        story = [Paragraph("CyberShield SOC — Incident Report", styles["Title"]),
                 Spacer(1, 12)]
        rows = [
            ["Threat", incident["threat_name"]],
            ["MITRE ATT&CK", incident["threat_id"]],
            ["Tactic", incident.get("tactic", "")],
            ["Confidence", f"{incident['confidence']:.1f}%"],
            ["Source IP", incident["src_ip"]],
            ["Status", incident.get("status", "DETECTED")],
            ["Playbook", incident.get("playbook", "")],
            ["Indicators", incident.get("indicators", "")[:300]],
        ]
        t = Table(rows, colWidths=[110, 380])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#0a0e1a")),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story += [t, Spacer(1, 12),
                  Paragraph("Top XAI Features:", styles["Heading3"])]
        for f in incident.get("xai_features", [])[:5]:
            story.append(Paragraph(f"• {f['name']}: {f['importance']:.4f}", styles["Normal"]))
        doc.build(story)
    except ImportError:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        lines = [f"CyberShield SOC — Incident Report",
                 f"Threat: {incident['threat_name']}",
                 f"MITRE: {incident['threat_id']} | Confidence: {incident['confidence']:.1f}%",
                 f"Source: {incident['src_ip']} | Status: {incident.get('status', '')}",
                 f"Playbook: {incident.get('playbook', '')}"]
        ax.text(0.05, 0.9, "\n\n".join(lines), fontsize=12, va="top", ha="left",
                transform=ax.transAxes)
        fig.savefig(pdf_name)
        plt.close(fig)
    return pdf_name


def send_telegram_document(chat_id: str, pdf_path: str, caption: str,
                           bot_token: str = BOT_TOKEN) -> bool:
    """إرسال ملف PDF عبر sendDocument (وضع محاكاة بدون توكن)."""
    if not bot_token or not chat_id:
        print(f"[SIMULATE] Telegram document → chat {chat_id or '?'}: {pdf_path}")
        return False
    try:
        with open(pdf_path, "rb") as f:
            r = requests.post(
                f"https://api.telegram.org/bot{bot_token}/sendDocument",
                data={"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"},
                files={"document": f}, timeout=30)
        return bool(r.ok)
    except Exception as e:
        print(f"⚠️ Telegram document failed: {e}")
        return False


async def handle_security_action(update, context) -> None:
    """معالج أزرار البوت — نقل حرفي من Cell 7."""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    query = update.callback_query
    if not query:
        return

    # تجنب تعليق البوت إذا كان التوكن قديماً
    try:
        await query.answer()
    except Exception:
        pass  # استكمال التنفيذ وتحديث الرسالة حتى لو انتهت مهلة الـ query

    data = query.data

    if data.startswith("block_"):
        src_ip = data.split("_")[1]
        firewall_rule = f"iptables -A INPUT -s {src_ip} -j DROP"

        if src_ip in active_incidents:
            active_incidents[src_ip]["status"] = "BLOCKED & MITIGATED"

        updated_text = (
            f"{query.message.text_html}\n\n"
            f"✅ <b>تم اتخاذ الإجراء الدفاعي بنجاح!</b>\n"
            f"🔒 <b>القاعدة المطبقة:</b> <code>{firewall_rule}</code>\n"
            f"⏱️ <b>زمن الاستجابة:</b> 0.4 ثانية"
        )

        post_block_keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("📄 تحميل تقرير الحادثة الكامل (PDF)",
                                  callback_data=f"report_{src_ip}")]
        ])

        try:
            await query.edit_message_text(
                text=updated_text,
                parse_mode="HTML",
                reply_markup=post_block_keyboard
            )
            print(f"🛡️ Firewall Rule Executed: Blocked {src_ip}")
        except Exception as e:
            print(f"⚠️ Note: Message already updated or edit error: {e}")

    elif data.startswith("report_"):
        src_ip = data.split("_")[1]
        incident = active_incidents.get(src_ip)
        if incident:
            pdf_name = f"/kaggle/working/Incident_{incident['threat_id']}_{src_ip.replace('.', '_')}.pdf"
            generate_incident_pdf(incident, pdf_name)
            caption = (
                f"📄 <b>تقرير الحادثة الأمني الرسمي (PDF)</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"🛡️ <b>الهجوم:</b> {incident['threat_name']}\n"
                f"🏷️ <b>التصنيف:</b> MITRE ATT&amp;CK {incident['threat_id']}\n"
                f"🎯 <b>نسبة التأكد:</b> {incident['confidence']:.1f}%\n"
                f"🌐 <b>المصدر:</b> <code>{incident['src_ip']}</code>\n"
                f"🔒 <b>الحالة:</b> {incident.get('status', 'BLOCKED')}"
            )
            send_telegram_document(ADMIN_CHAT_ID, pdf_name, caption)
            print(f"📄 Incident PDF Report generated and sent for IP: {src_ip}")
        else:
            await query.message.reply_text("⚠️ لم يتم العثور على بيانات الحادثة.")

    elif data.startswith("ignore_"):
        try:
            await query.edit_message_text(
                text=f"{query.message.text_html}\n\n⚠️ <i>تم تجاهل الإنذار وتصنيفه كـ False Positive.</i>",
                parse_mode="HTML")
        except Exception:
            pass


async def run_monitor(
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    model: torch.nn.Module,
    scaler,
    feature_names: Sequence[str],
    retriever: HybridThreatRetriever,
    device: torch.device,
    bot_token: str = BOT_TOKEN,
    n_demo_attacks: int = 2,
    poll_seconds: int = 120,
) -> None:
    """تشغيل البوت والفحص الحي — نقل حرفي لمنطق main() في Cell 7."""
    from telegram.ext import ApplicationBuilder, CallbackQueryHandler

    print("🚀 Starting CyberShield AI Live Monitor...")

    attack_indices = np.where(y_test == 1)[0][:n_demo_attacks]
    for idx in attack_indices:
        incident = analyze_flow_with_ai(X_test_raw[idx], model, scaler,
                                        feature_names, retriever, device)
        if incident:
            print(f"🚨 Attack Detected ({incident['threat_name']})! Sending alert...")
            dispatch_telegram_alert(incident, bot_token, ADMIN_CHAT_ID)
            await asyncio.sleep(1)

    if not bot_token:
        print("⚠️ No CYBERSHIELD_BOT_TOKEN — demo alerts simulated, polling skipped.")
        return

    app = ApplicationBuilder().token(bot_token).build()
    app.add_handler(CallbackQueryHandler(handle_security_action))

    print("\n🤖 Bot is active! Check your phone:")
    print("   1. اضغطي [🛑 حظر الـ IP فوراً]")
    print("   2. اضغطي [📄 تحميل تقرير الحادثة الكامل (PDF)]")

    await app.initialize()
    await app.start()
    # drop_pending_updates=True يمسح أي ضغطات قديمة معلقة لمنع الخطأ
    await app.updater.start_polling(drop_pending_updates=True)

    try:
        await asyncio.sleep(poll_seconds)
    finally:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()
        print("🛑 Monitoring session closed.")


def run_monitor_sync(*args, **kwargs) -> None:
    """نسخة synchronous للتشغيل من سكربت (asyncio.run)."""
    asyncio.run(run_monitor(*args, **kwargs))
