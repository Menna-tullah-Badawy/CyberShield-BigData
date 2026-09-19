
#### 1️⃣2️⃣ `README.md`
📍 **المسار الكامل:** `CyberShield-BigData/README.md`

```markdown
# 🛡️ CyberShield-BigData: Enterprise Distributed Hybrid NIDS

[![CI/CD Pipeline](https://github.com/cybershield/nids-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/cybershield/nids-engine/actions)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.4.1-orange.svg)](https://spark.apache.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Ready-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

An enterprise-grade, distributed **Hybrid Network Intrusion Detection System (NIDS)** and MLOps framework. Built for high-throughput environments processing millions of packets using **Apache Spark**, **SecBERT NLP Transformers**, **Explainable AI (XAI)**, and **GenAI Incident Reporting** aligned with **MITRE ATT&CK**.

---

## 🌟 Key Architecture Pillars
1. **Scalable Ingestion**: Raw PCAP disassembly and Spark Structured Streaming.
2. **Independent Data Quality Gate**: Automated auditing of missing values, duplicates, outliers, and schema drift.
3. **7-Component Feature Pipeline**: Behavioral aggregations, datetime extraction, scaling, variance selection, and domain transforms.
4. **Explainable AI (XAI)**: Gini feature importance + token saliency for payload threat attribution.
5. **Continuous MLOps Monitoring**: Statistical drift detection via **Kolmogorov-Smirnov (KS-Test)** and automated model retraining.
6. **GenAI Threat Intelligence**: Automated MITRE ATT&CK mapping and executive incident triage generation.

---

## 🚀 Quickstart

### Local Setup
```bash
# 1. Clone repository
git clone [https://github.com/your-org/CyberShield-BigData.git](https://github.com/your-org/CyberShield-BigData.git)
cd CyberShield-BigData

# 2. Install dependencies
make install

# 3. Run full End-to-End Pipeline
python main.py --mode all