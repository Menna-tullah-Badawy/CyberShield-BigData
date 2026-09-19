# 🛡️ CyberShield-BigData: Hybrid NIDS

Network Intrusion Detection on CICIDS-2018 with **temporal deep learning**
(BiLSTM · BiGRU · CNN-BiLSTM · **Mamba SSM**), gradient-boosted baselines
(XGBoost · CatBoost · LightGBM), a calibrated **Hybrid Ensemble**,
gradient **XAI**, **MITRE ATT&CK RAG** (BM25 + LSA), a 7-page SOC PDF report,
a **FastAPI** serving layer, and a **Telegram** live SOC monitor.

Reference implementation: `cybershield.ipynb` (Kaggle) — every cell maps 1:1
to a module in this repo (see `docs/architecture.md`).

## 🚀 Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Full pipeline: extract → benchmark → diagnostics → figures
python main.py --mode all --data-path /path/to/csvs   # or omit for kagglehub download

# 3. Serve the API / Build the PDF / Run the Telegram bot
python main.py --mode serve     # http://localhost:8000/docs
python main.py --mode report    # benchmark_results/CyberShield_SOC_Report.pdf
python main.py --mode bot       # needs CYBERSHIELD_BOT_TOKEN
```

```bash
make test        # pytest suite
make run-api     # uvicorn serving.api.main:app
```

## 🧭 Layout

```
src/
├── data_pipeline/   CSV ingestion (kagglehub + day files)
├── quality/         lightweight validation gate
├── cleaning/        to_numeric + fillna(0) + nan_to_num
├── feature_engineering/  timestamp sort · sequences · scaling · selection
├── feature_store/   .npy store + metadata + 1:4 balancing
├── models/          tabular factory · DL suite · trainer · champion selector
├── evaluation/      threshold optimizer · diagnostics · publication figures
├── explainability/  gradient saliency XAI engine
├── monitoring/      batch audit + Telegram live monitor
├── genai_reporting/ MITRE RAG retriever + SOC PDF reports
├── orchestrator/    benchmark end-to-end runner
└── common/          logger · exceptions · decorators
serving/api/         FastAPI (/health, /api/metrics, /api/threats, POST /api/detect)
scripts/             run_benchmark.py · deploy.sh
configs/             YAML constants mirroring the notebook
```

## 📊 Sweet-spot targets

Thresholds are calibrated per model for **Precision ≥ 92% & Recall ≥ 90%**
with FPR ≤ 1.5% (`configs/model_config.yaml`).
