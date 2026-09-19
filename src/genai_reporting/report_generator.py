"""
Incident Report Generator: 7-page SOC PDF + single-incident PDF.
نقل حرفي من Cell 4 + Cell 7 في cybershield.ipynb.

ملحوظة: صفحة مقارنة الموديلات كانت بأرقام ثابتة في النوت بوك،
هنا تُبنى من نتائج البنش مارك الحقيقية إن توفرت.
"""

import os
import time
from typing import Dict, Mapping, Optional

import numpy as np

C_DARK = '#0a0e1a'
C_CYAN = '#00d4ff'
C_GREEN = '#00ff88'
C_RED = '#ff3366'
C_YELLOW = '#ffcc00'
C_PURPLE = '#a855f7'
C_WHITE = '#ffffff'
C_GRAY = '#9ca3af'

_FALLBACK_MODELS = ['LightGBM', 'XGBoost', 'BiLSTM', 'BiGRU', 'Mamba SSM']
_FALLBACK_RECALL = [0.011, 0.078, 0.391, 0.301, 0.606]
_FALLBACK_F1 = [0.021, 0.123, 0.469, 0.381, 0.640]
_FALLBACK_ROC = [0.558, 0.567, 0.727, 0.712, 0.809]


class IncidentReportGenerator:
    """مولّد تقارير الحوادث (SOC الشامل + الحادثة الواحدة)."""

    @staticmethod
    def _comparison_arrays(results: Optional[Mapping[str, Mapping[str, float]]]):
        if results:
            order = [m for m in ["LightGBM", "XGBoost", "BiLSTM", "BiGRU", "MAMBA_SSM"]
                     if m in results]
            if len(order) >= 3:
                labels = ["Mamba SSM" if m == "MAMBA_SSM" else m for m in order]
                return (labels,
                        [results[m]["Recall"] for m in order],
                        [results[m]["F1-Score"] for m in order],
                        [results[m]["ROC-AUC"] for m in order])
        return _FALLBACK_MODELS, _FALLBACK_RECALL, _FALLBACK_F1, _FALLBACK_ROC

    def create_soc_report(
        self, pdf_path: str, tp: int, fp: int, tn: int, fn: int, roc: float,
        n_samples: int, cm: np.ndarray, xai_features: list,
        threat_types_detected: Mapping[str, dict], threat_kb: list, port: int,
        results: Optional[Mapping[str, Mapping[str, float]]] = None,
        report_id: Optional[str] = None, verbose: bool = True,
    ) -> str:
        """تقرير SOC السبع صفحات (نفس كود النوت بوك)."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.patches as mpatches

        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['font.size'] = 10
        os.makedirs(os.path.dirname(os.path.abspath(pdf_path)), exist_ok=True)
        if report_id is None:
            report_id = f"INC-2026-NIDS-{np.random.randint(10000, 99999)}"

        with PdfPages(pdf_path) as pdf:
            # ── PAGE 1: Cover ──
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.patch.set_facecolor(C_DARK)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_facecolor(C_DARK)
            ax.add_patch(mpatches.Rectangle((0, 0.92), 1, 0.08, color=C_CYAN))
            ax.text(0.5, 0.78, '🛡️', fontsize=80, ha='center', va='center', color=C_CYAN)
            ax.text(0.5, 0.65, 'CYBERSHIELD', fontsize=36, ha='center', color=C_CYAN,
                    fontweight='bold', fontfamily='monospace')
            ax.text(0.5, 0.58, 'Network Intrusion Detection System', fontsize=14,
                    ha='center', color=C_GRAY)
            ax.add_patch(mpatches.FancyBboxPatch((0.15, 0.38), 0.7, 0.12,
                         boxstyle="round,pad=0.02", facecolor='#111827',
                         edgecolor=C_CYAN, linewidth=2))
            ax.text(0.5, 0.44, 'SOC INCIDENT REPORT', fontsize=22, ha='center', va='center',
                    color=C_WHITE, fontweight='bold')
            ax.text(0.5, 0.30, f'Report ID: {report_id}', fontsize=12, ha='center', color=C_GRAY)
            ax.text(0.5, 0.26, f'Generated: {time.strftime("%Y-%m-%d %H:%M:%S UTC")}',
                    fontsize=11, ha='center', color=C_GRAY)
            ax.text(0.5, 0.22, 'Detection Model: Mamba SSM (State Space Model)',
                    fontsize=11, ha='center', color=C_GRAY)
            ax.text(0.5, 0.18, 'Classification: CONFIDENTIAL', fontsize=11,
                    ha='center', color=C_RED, fontweight='bold')
            ax.add_patch(mpatches.Rectangle((0, 0), 1, 0.06, color=C_CYAN))
            ax.text(0.5, 0.03, 'CyberShield BigData — Autonomous SOC Division', fontsize=9,
                    ha='center', va='center', color=C_DARK)
            pdf.savefig(fig, facecolor=C_DARK)
            plt.close()

            # ── PAGE 2: Executive Summary ──
            fig, axes = plt.subplots(2, 2, figsize=(8.27, 11.69))
            fig.patch.set_facecolor('white')
            fig.suptitle('Executive Summary — Detection Metrics', fontsize=16,
                         fontweight='bold', color=C_DARK, y=0.98)
            for a in axes.flat:
                a.axis('off')
            cards = [
                (axes[0, 0], C_CYAN, C_RED, f'{tp + fp}', 'Threats Detected',
                 f'out of {n_samples} samples'),
                (axes[0, 1], C_GREEN, C_GREEN, f'{roc:.4f}', 'ROC-AUC Score',
                 'Discrimination Ability'),
                (axes[1, 0], C_PURPLE, C_PURPLE, f'{len(threat_types_detected)}',
                 'Threat Types Identified', f'from {len(threat_kb)} MITRE categories'),
            ]
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            cards.append((axes[1, 1], C_YELLOW, C_YELLOW, f'{fnr * 100:.1f}%',
                          'False Negative Rate', f'{fn} attacks missed'))
            for a, edge, num_color, num, title, sub in cards:
                a.add_patch(mpatches.FancyBboxPatch((0.1, 0.2), 0.8, 0.6,
                            boxstyle="round,pad=0.05", facecolor=C_DARK,
                            edgecolor=edge, linewidth=2))
                a.text(0.5, 0.65, num, fontsize=48, ha='center', va='center',
                       color=num_color, fontweight='bold')
                a.text(0.5, 0.30, title, fontsize=14, ha='center', va='center', color=C_WHITE)
                a.text(0.5, 0.15, sub, fontsize=10, ha='center', color=C_GRAY)
            plt.subplots_adjust(top=0.93, hspace=0.3, wspace=0.2)
            pdf.savefig(fig, facecolor='white')
            plt.close()

            # ── PAGE 3: XAI + Confusion Matrix ──
            fig, axes = plt.subplots(1, 2, figsize=(8.27, 11.69))
            fig.suptitle('Root Cause Analysis & Detection Performance', fontsize=16,
                         fontweight='bold', color=C_DARK, y=0.98)
            ax1 = axes[0]
            names = [f["name"][:20] for f in xai_features]
            values = [f["importance"] for f in xai_features]
            colors = [C_RED, C_YELLOW, C_CYAN, C_GREEN, C_PURPLE][:len(names)]
            y_pos = np.arange(len(names))
            ax1.barh(y_pos, values, color=colors, edgecolor='white', linewidth=1.5)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(names, fontsize=9)
            ax1.invert_yaxis()
            ax1.set_xlabel('Impact Weight', fontsize=10)
            ax1.set_title('Top 5 Anomalous Features\n(XAI Saliency)', fontsize=12,
                          fontweight='bold', color=C_DARK)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            for i, v in enumerate(values):
                ax1.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=9, fontweight='bold')
            ax1.set_xlim(0, max(values) * 1.3)
            ax2 = axes[1]
            ax2.imshow(cm, cmap='Blues', aspect='auto')
            ax2.set_xticks([0, 1])
            ax2.set_yticks([0, 1])
            ax2.set_xticklabels(['Benign', 'Attack'], fontsize=10)
            ax2.set_yticklabels(['Benign', 'Attack'], fontsize=10)
            ax2.set_xlabel('Predicted', fontsize=10)
            ax2.set_ylabel('Actual', fontsize=10)
            ax2.set_title('Confusion Matrix\n(Batch Test)', fontsize=12,
                          fontweight='bold', color=C_DARK)
            for i in range(2):
                for j in range(2):
                    color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                    ax2.text(j, i, str(cm[i, j]), ha='center', va='center',
                             fontsize=20, fontweight='bold', color=color)
            plt.subplots_adjust(top=0.88, wspace=0.4)
            pdf.savefig(fig, facecolor='white')
            plt.close()

            # ── PAGE 4: Model Comparison ──
            fig, ax = plt.subplots(figsize=(8.27, 5))
            fig.suptitle('Model Performance Comparison', fontsize=14,
                         fontweight='bold', color=C_DARK)
            models, recall, f1, roc_auc = self._comparison_arrays(results)
            x = np.arange(len(models))
            width = 0.25
            ax.bar(x - width, recall, width, label='Recall', color=C_GREEN, edgecolor='white')
            ax.bar(x, f1, width, label='F1-Score', color=C_CYAN, edgecolor='white')
            ax.bar(x + width, roc_auc, width, label='ROC-AUC', color=C_PURPLE, edgecolor='white')
            ax.set_ylabel('Score', fontsize=10)
            ax.set_title(f'{len(models)} Models × 3 Metrics', fontsize=11, color=C_GRAY)
            ax.set_xticks(x)
            ax.set_xticklabels(models, fontsize=9)
            ax.legend(fontsize=9, loc='upper left')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylim(0, 1.0)
            best_i = int(np.argmax(roc_auc))
            ax.axvspan(best_i - 0.5, best_i + 0.5, alpha=0.1, color=C_GREEN)
            ax.annotate('BEST', xy=(best_i, roc_auc[best_i]), xytext=(best_i, 0.9),
                        fontsize=10, fontweight='bold', color=C_GREEN, ha='center',
                        arrowprops=dict(arrowstyle='->', color=C_GREEN))
            plt.tight_layout()
            pdf.savefig(fig, facecolor='white')
            plt.close()

            # ── PAGE 5: Threat Catalog ──
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis('off')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            fig.suptitle('Threat Intelligence Catalog — All Detected Threats',
                         fontsize=14, fontweight='bold', color=C_DARK, y=0.98)
            y_pos = 0.92
            for tid, info in threat_types_detected.items():
                card_height = 0.12
                ax.add_patch(mpatches.FancyBboxPatch((0.05, y_pos - card_height), 0.9, card_height,
                             boxstyle="round,pad=0.01", facecolor='#f8f9fa',
                             edgecolor=C_CYAN, linewidth=1.5))
                ax.text(0.08, y_pos - 0.02, f"{tid}", fontsize=10, fontweight='bold', color=C_RED)
                ax.text(0.08, y_pos - 0.045, f"Title: {info['title']}", fontsize=9, color=C_DARK)
                ax.text(0.08, y_pos - 0.065, f"Tactic: {info['tactic']}", fontsize=8, color=C_GRAY)
                ax.text(0.08, y_pos - 0.085,
                        f"Occurrences: {info['count']} | Hybrid: {info['hybrid_score']:.4f}",
                        fontsize=8, color=C_PURPLE)
                ax.text(0.08, y_pos - 0.105, f"Indicators: {info['indicators'][:80]}...",
                        fontsize=7, color=C_GRAY, style='italic')
                y_pos -= card_height + 0.02
            ax.text(0.05, y_pos - 0.02, "Reference: All MITRE ATT&CK Threats in Knowledge Base:",
                    fontsize=10, fontweight='bold', color=C_DARK)
            y_pos -= 0.05
            for threat in threat_kb:
                detected = "DETECTED" if threat["threat_id"] in threat_types_detected else "—"
                color = C_GREEN if threat["threat_id"] in threat_types_detected else C_GRAY
                ax.text(0.08, y_pos, f"• {threat['threat_id']}: {threat['title']}  [{detected}]",
                        fontsize=8, color=color)
                y_pos -= 0.025
            pdf.savefig(fig, facecolor='white')
            plt.close()

            # ── PAGE 6: Playbooks ──
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis('off')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            fig.suptitle('Autonomous Mitigation Playbooks & Response Scripts',
                         fontsize=14, fontweight='bold', color=C_DARK, y=0.98)
            y_pos = 0.92
            for tid, info in threat_types_detected.items():
                card_h = 0.15
                ax.add_patch(mpatches.FancyBboxPatch((0.03, y_pos - card_h), 0.94, card_h,
                             boxstyle="round,pad=0.01", facecolor=C_DARK,
                             edgecolor=C_GREEN, linewidth=2))
                ax.text(0.06, y_pos - 0.02, f"🛡️ {tid} — {info['title']}", fontsize=9,
                        fontweight='bold', color=C_GREEN)
                ax.text(0.06, y_pos - 0.04, f"Playbook: {info['playbook'][:90]}...",
                        fontsize=7, color=C_WHITE)
                ipt_text = "\n".join(r.replace("{src_ip}", "0.0.0.0") for r in info['iptables'][:3])
                ax.text(0.06, y_pos - 0.075, "iptables:", fontsize=7,
                        color=C_YELLOW, fontweight='bold')
                ax.text(0.10, y_pos - 0.095, ipt_text[:120], fontsize=6, color=C_GREEN,
                        fontfamily='monospace')
                y_pos -= card_h + 0.02
            ax.text(0.5, 0.02, f'CyberShield SOC Report — {report_id} — {time.strftime("%Y-%m-%d")}',
                    fontsize=8, ha='center', color=C_GRAY, style='italic')
            pdf.savefig(fig, facecolor='white')
            plt.close()

            # ── PAGE 7: API Summary ──
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis('off')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            fig.suptitle('API Endpoints — Threat Detection & Response',
                         fontsize=14, fontweight='bold', color=C_DARK, y=0.98)
            endpoints = [
                ("GET", "/health", "Health check + model status"),
                ("GET", "/api/metrics", "Detection metrics (TP/FP/TN/FN/ROC)"),
                ("GET", "/api/threats", "ALL threat types + indicators + mitigations"),
                ("POST", "/api/detect", "Real-time threat detection + XAI + RAG + iptables"),
            ]
            y = 0.88
            for method, path, desc in endpoints:
                color = C_CYAN if method == "GET" else C_GREEN
                ax.add_patch(mpatches.FancyBboxPatch((0.05, y - 0.06), 0.9, 0.05,
                             boxstyle="round,pad=0.01", facecolor='#f8f9fa',
                             edgecolor=color, linewidth=2))
                ax.text(0.08, y - 0.03, method, fontsize=9, fontweight='bold', color=color)
                ax.text(0.18, y - 0.03, path, fontsize=10, fontfamily='monospace', color=C_DARK)
                ax.text(0.18, y - 0.05, desc, fontsize=8, color=C_GRAY)
                y -= 0.08
            ax.text(0.05, y - 0.02, "Live API Test Results:", fontsize=11,
                    fontweight='bold', color=C_DARK)
            y -= 0.05
            ax.text(0.08, y, f"• Batch Test: {n_samples} samples", fontsize=9, color=C_DARK)
            y -= 0.03
            ax.text(0.08, y, f"• Threats Detected: {tp + fp} / {n_samples}", fontsize=9, color=C_RED)
            y -= 0.03
            ax.text(0.08, y, f"• True Positives: {tp} | False Negatives: {fn}",
                    fontsize=9, color=C_DARK)
            y -= 0.03
            ax.text(0.08, y, f"• ROC-AUC: {roc:.4f}", fontsize=9, color=C_GREEN)
            y -= 0.03
            ax.text(0.08, y, f"• Threat Types: {len(threat_types_detected)}",
                    fontsize=9, color=C_PURPLE)
            y -= 0.05
            ax.text(0.05, y, f"API Base URL: http://localhost:{port}", fontsize=9,
                    fontfamily='monospace', color=C_CYAN)
            pdf.savefig(fig, facecolor='white')
            plt.close()

        if verbose:
            size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
            print(f"\n📄 PDF Report saved: {pdf_path}")
            print(f"   Size: {size_mb:.2f} MB")
            print("   Pages: 7 (Cover + Metrics + XAI + Models + Threats + Mitigations + API)")
            print(f"   Threat Types: {len(threat_types_detected)}")
        return pdf_path

    def create_incident_report(self, incident: Dict, pdf_name: str) -> str:
        """تقرير PDF للحادثة الواحدة (reportlab + بديل matplotlib)."""
        os.makedirs(os.path.dirname(os.path.abspath(pdf_name)), exist_ok=True)
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

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
            story += [t, Spacer(1, 12), Paragraph("Top XAI Features:", styles["Heading3"])]
            for f in incident.get("xai_features", [])[:5]:
                story.append(Paragraph(f"• {f['name']}: {f['importance']:.4f}", styles["Normal"]))
            doc.build(story)
        except ImportError:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis("off")
            lines = ["CyberShield SOC — Incident Report",
                     f"Threat: {incident['threat_name']}",
                     f"MITRE: {incident['threat_id']} | Confidence: {incident['confidence']:.1f}%",
                     f"Source: {incident['src_ip']} | Status: {incident.get('status', '')}",
                     f"Playbook: {incident.get('playbook', '')}"]
            ax.text(0.05, 0.9, "\n\n".join(lines), fontsize=12, va="top", ha="left",
                    transform=ax.transAxes)
            fig.savefig(pdf_name)
            plt.close(fig)
        return pdf_name
