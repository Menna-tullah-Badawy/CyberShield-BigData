# 🏛️ CyberShield-BigData: Architecture (torch track)

Source of truth: `cybershield.ipynb`. Every notebook cell maps to repo modules:

| Notebook | Modules |
|---|---|
| Cell 1 — Temporal extraction & split | `data_pipeline/ingestion` → `quality/engine` → `cleaning/cleaning_pipeline` → `feature_engineering/*` → `feature_store/*` (via `FeaturePipelineBuilder`) |
| Cell 2 — Sweet-spot benchmark | `orchestrator/pipeline_orchestrator` + `models/*` + `evaluation/threshold_optimizer` + `evaluation/metrics` |
| Cell 3 — IEEE/ACM diagnostics | `evaluation/model_validator` |
| Cell 4 — Publication figures | `evaluation/figures` |
| Cell 5 — RAG + XAI + API + PDF | `genai_reporting/mitre_mapping` + `explainability/*` + `serving/api/*` + `genai_reporting/report_generator` (+ `monitoring` batch audit) |
| Cell 7 — Telegram monitor | `monitoring/alert_manager` + `monitoring/monitoring_engine` |

## Data flow

```
CICIDS-2018 CSVs ─▶ ingestion ─▶ quality gate ─▶ cleaning ─▶ feature_engineering
      │                                                        (sort · sample · target
      │                                                         exclude · split · sequences
      │                                                         zero-var filter)
      ▼
feature_store (.npy + metadata) ─▶ balancing 1:4 ─▶ RobustScaler ─▶ tabular feats
      │                                                             │
      ▼                                                             ▼
  DL tensors ─▶ CNN-BiLSTM / BiLSTM / BiGRU / Mamba ─┐
  tabular    ─▶ XGBoost / CatBoost / LightGBM ──────┤
                                                    ▼
                              threshold optimizer (P≥92 / R≥90 / FPR≤1.5%)
                                                    ▼
                              Hybrid Ensemble (.40/.35/.15/.10) ─▶ benchmark CSV
                                                    ▼
                              diagnostics ─▶ figures ─▶ API ─▶ SOC PDF ─▶ Telegram bot
```

## Key design decisions

- **Leakage-proof**: per-day temporal splits, scaler/variance-filter fit on train only,
  strict identity/port exclusion (`STRICT_EXCLUDE`).
- **Champion serving**: the API serves the trained Mamba SSM (+XAI + MITRE RAG + iptables).
- **No Spark**: the legacy distributed track was removed; everything is pandas/numpy/torch.
