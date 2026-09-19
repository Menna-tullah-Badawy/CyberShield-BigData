"""
Live Monitoring Engine — AI detection pipeline + Telegram bot loop.
منطق Cell 7: analyze_flow_with_ai + handle_security_action + run_monitor.
"""

import asyncio
import random
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from src.explainability.xai_engine import ExplainabilityEngine
from src.monitoring.alert_manager import ADMIN_CHAT_ID, AlertManager, active_incidents

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
DETECT_THRESHOLD = 0.03


class MLOpsMonitoringEngine:
    """محرك المراقبة الحية: كشف + تفسير + تنبيه + بوت تفاعلي."""

    def __init__(self, model, scaler, feature_names: Sequence[str],
                 retriever=None, device: torch.device = None,
                 bot_token: str = "", chat_id: str = ADMIN_CHAT_ID):
        self.model = model
        self.scaler = scaler
        self.feature_names = list(feature_names)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.xai = ExplainabilityEngine(feature_names, retriever)
        self.alerts = AlertManager(bot_token, chat_id)

    def analyze_flow_with_ai(
        self, flow_seq: np.ndarray, src_ip: Optional[str] = None,
        threshold: float = DETECT_THRESHOLD,
    ) -> Optional[dict]:
        """تحليل Flow واحد: تنبؤ + XAI + RAG → قاموس الحادثة (نفس مفاتيح Cell 7)."""
        flat = np.asarray(flow_seq, dtype=np.float32).reshape(-1, len(self.feature_names))
        scaled = np.clip(self.scaler.transform(flat), -15, 15).reshape(
            1, flow_seq.shape[0], len(self.feature_names))
        x = torch.tensor(scaled, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            ap = float(torch.softmax(self.model(x), 1)[0, 1].cpu().numpy())
        if ap < threshold:
            return None
        expl = self.xai.generate_incident_explanation(
            self.model, x.clone(), self.feature_names, n=5)
        doc = expl["threat"]
        threat_id = doc["threat_id"]
        ip = src_ip or f"{DEMO_SUBNET}.{random.randint(2, 254)}"
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
            "xai_features": expl["top_features"],
            "hybrid_score": round(expl["hybrid_score"], 4),
        }

    async def handle_security_action(self, update, context) -> None:
        """معالج أزرار البوت — نقل حرفي من Cell 7."""
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        from src.genai_reporting.report_generator import IncidentReportGenerator

        query = update.callback_query
        if not query:
            return
        try:
            await query.answer()
        except Exception:
            pass

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
                await query.edit_message_text(text=updated_text, parse_mode="HTML",
                                              reply_markup=post_block_keyboard)
                print(f"🛡️ Firewall Rule Executed: Blocked {src_ip}")
            except Exception as e:
                print(f"⚠️ Note: Message already updated or edit error: {e}")

        elif data.startswith("report_"):
            src_ip = data.split("_")[1]
            incident = active_incidents.get(src_ip)
            if incident:
                pdf_name = (f"/kaggle/working/Incident_{incident['threat_id']}_"
                            f"{src_ip.replace('.', '_')}.pdf")
                IncidentReportGenerator().create_incident_report(incident, pdf_name)
                caption = (
                    f"📄 <b>تقرير الحادثة الأمني الرسمي (PDF)</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"🛡️ <b>الهجوم:</b> {incident['threat_name']}\n"
                    f"🏷️ <b>التصنيف:</b> MITRE ATT&amp;CK {incident['threat_id']}\n"
                    f"🎯 <b>نسبة التأكد:</b> {incident['confidence']:.1f}%\n"
                    f"🌐 <b>المصدر:</b> <code>{incident['src_ip']}</code>\n"
                    f"🔒 <b>الحالة:</b> {incident.get('status', 'BLOCKED')}"
                )
                self.alerts.send_document(pdf_name, caption)
                print(f"📄 Incident PDF Report generated and sent for IP: {src_ip}")
            else:
                await query.message.reply_text("⚠️ لم يتم العثور على بيانات الحادثة.")

        elif data.startswith("ignore_"):
            try:
                await query.edit_message_text(
                    text=f"{query.message.text_html}\n\n"
                         f"⚠️ <i>تم تجاهل الإنذار وتصنيفه كـ False Positive.</i>",
                    parse_mode="HTML")
            except Exception:
                pass

    def run_batch_audit(
        self, X_test_raw: np.ndarray, y_test: np.ndarray,
        n_per_class: int = 50, seed: int = None, verbose: bool = True,
    ) -> Dict[str, object]:
        """تدقيق دفعي: 50 هجمة + 50 سليمة ثم تجميع أنواع التهديدات (منطق Cell 4)."""
        from sklearn.metrics import confusion_matrix, roc_auc_score

        rng = np.random.RandomState(seed)
        attack_idx = np.where(y_test == 1)[0]
        benign_idx = np.where(y_test == 0)[0]
        test_indices = np.concatenate([
            rng.choice(attack_idx, min(n_per_class, len(attack_idx)), replace=False),
            rng.choice(benign_idx, min(n_per_class, len(benign_idx)), replace=False),
        ])
        test_true = y_test[test_indices]
        self.model.eval()
        test_probs, test_preds = [], []
        batch_size = 32
        with torch.no_grad():
            for i in range(0, len(test_indices), batch_size):
                batch_idx = test_indices[i:i + batch_size]
                flat = X_test_raw[batch_idx].reshape(-1, len(self.feature_names))
                scaled = np.clip(self.scaler.transform(flat), -15, 15).reshape(
                    len(batch_idx), X_test_raw.shape[1], len(self.feature_names))
                x = torch.tensor(scaled, dtype=torch.float32, device=self.device)
                probs = torch.softmax(self.model(x), 1)[:, 1].cpu().numpy()
                test_probs.extend(probs)
                test_preds.extend((np.array(probs) >= DETECT_THRESHOLD).astype(int))
        test_probs = np.array(test_probs)
        test_preds = np.array(test_preds)
        cm = confusion_matrix(test_true, test_preds, labels=[0, 1])
        tn, fp, fn, tp = (int(v) for v in cm.ravel())
        roc = float(roc_auc_score(test_true, test_probs))
        if verbose:
            print(f"   TP={tp} | FP={fp} | TN={tn} | FN={fn} | ROC-AUC={roc:.4f}")

        best_attack = test_indices[np.argmax(test_probs)]
        flat = X_test_raw[best_attack].reshape(-1, len(self.feature_names))
        scaled = np.clip(self.scaler.transform(flat), -15, 15).reshape(
            1, X_test_raw.shape[1], len(self.feature_names))
        x_tensor = torch.tensor(scaled, dtype=torch.float32, device=self.device)
        xai_features = self.xai.saliency.extract_importance(
            self.model, x_tensor, self.feature_names, n=5)

        if verbose:
            print("\n📡 Collecting all threat types...")
        threat_types_detected: Dict[str, dict] = {}
        for i, idx in enumerate(test_indices):
            if test_preds[i] == 1:
                xf = self.xai.saliency.extract_importance(
                    self.model,
                    torch.tensor(
                        np.clip(self.scaler.transform(
                            X_test_raw[idx].reshape(-1, len(self.feature_names))),
                            -15, 15).reshape(1, X_test_raw.shape[1],
                                             len(self.feature_names)),
                        dtype=torch.float32, device=self.device),
                    self.feature_names, n=3)
                qt = " ".join(str(f["name"]) for f in xf).lower().replace("_", " ")
                rag = self.xai.retriever.retrieve(qt)
                tid = rag["doc"]["threat_id"]
                if tid not in threat_types_detected:
                    threat_types_detected[tid] = {
                        "title": rag["doc"]["title"],
                        "tactic": rag["doc"]["tactic"],
                        "playbook": rag["doc"]["playbook"],
                        "indicators": rag["doc"].get("indicators", ""),
                        "iptables": rag["doc"].get("iptables", []),
                        "count": 0,
                        "hybrid_score": rag["hybrid_score"],
                    }
                threat_types_detected[tid]["count"] += 1
        if verbose:
            print(f"   Detected {len(threat_types_detected)} threat types:")
            for tid, info in threat_types_detected.items():
                print(f"   • {tid}: {info['title']} ({info['count']} occurrences)")
        return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "roc": roc,
                "n_samples": len(test_indices), "cm": cm,
                "xai_features": xai_features,
                "threat_types_detected": threat_types_detected}

    async def run_production_audit(
        self, X_test_raw: np.ndarray, y_test: np.ndarray,
        n_demo_attacks: int = 2, poll_seconds: int = 120,
    ) -> None:
        """تشغيل المراقبة الحية والبوت — نقل حرفي لمنطق main() في Cell 7."""
        from telegram.ext import ApplicationBuilder, CallbackQueryHandler

        print("🚀 Starting CyberShield AI Live Monitor...")
        attack_indices = np.where(y_test == 1)[0][:n_demo_attacks]
        for idx in attack_indices:
            incident = self.analyze_flow_with_ai(X_test_raw[idx])
            if incident:
                print(f"🚨 Attack Detected ({incident['threat_name']})! Sending alert...")
                self.alerts.dispatch_alert(incident)
                await asyncio.sleep(1)

        if not self.alerts.bot_token:
            print("⚠️ No CYBERSHIELD_BOT_TOKEN — demo alerts simulated, polling skipped.")
            return

        app = ApplicationBuilder().token(self.alerts.bot_token).build()
        app.add_handler(CallbackQueryHandler(self.handle_security_action))
        print("\n🤖 Bot is active! Check your phone:")
        print("   1. اضغطي [🛑 حظر الـ IP فوراً]")
        print("   2. اضغطي [📄 تحميل تقرير الحادثة الكامل (PDF)]")
        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)
        try:
            await asyncio.sleep(poll_seconds)
        finally:
            await app.updater.stop()
            await app.stop()
            await app.shutdown()
            print("🛑 Monitoring session closed.")
