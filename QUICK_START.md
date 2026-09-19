# ⚡ CyberShield Quick Start

## Kaggle (reference track)

1. Upload this repo as the Kaggle dataset `cybershield-bigdata-project`.
2. Open `cybershield.ipynb` and **Run All** (GPU recommended):
   - Cell 1 → builds `data/feature_store/nids_features_latest/*.npy`
   - Cell 2 → trains 7 models + Hybrid Ensemble → `benchmark_results/*.csv`
   - Cell 3 → IEEE/ACM diagnostics · Cell 4 → publication figures
   - Cell 5 → FastAPI + 7-page SOC PDF · Cell 7 → Telegram live monitor
3. Put secrets in Kaggle Secrets (never in code):
   - `CYBERSHIELD_BOT_TOKEN`, `CYBERSHIELD_ADMIN_CHAT_ID`

## Local / Server

```bash
pip install -r requirements.txt
python main.py --mode all --data-path ./data/raw
python main.py --mode serve   # API on :8000
```

## Docker

```bash
docker compose -f deployment/docker-compose.yml up --build -d
curl http://localhost:8000/health
```
