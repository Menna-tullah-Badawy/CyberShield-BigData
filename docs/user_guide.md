

#### 1️⃣1️⃣ `docs/user_guide.md`
📍 **المسار الكامل:** `CyberShield-BigData/docs/user_guide.md`

```markdown
# 🛡️ CyberShield SOC Analyst Operational Guide

## 1. Accessing the Monitoring Dashboard
Navigate to `http://localhost:8501` to access the Streamlit SOC UI:
- **Real-time Surveillance**: View throughput (PPS), bandwidth (Mbps), and live attack detection alerts.
- **Forensic Threat Analysis (XAI)**: Inspect flagged packet payloads, examine highlighted exploit strings, and view feature saliency.
- **GenAI Incident Reports**: Review AI-generated triage briefs mapped to the **MITRE ATT&CK** framework.

## 2. Triggering Manual Retraining
If drift exceeds acceptable thresholds, run:
```bash
python scripts/retrain_model.py