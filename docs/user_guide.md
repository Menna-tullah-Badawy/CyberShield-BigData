# 📖 User Guide

## CLI modes (`python main.py --mode <mode>`)

| Mode | What it does |
|---|---|
| `all` | extract → benchmark → diagnostics → figures (full run) |
| `extract` | build the `.npy` Feature Store from CSVs (skips if built; `--force` rebuilds) |
| `benchmark` | train 7 models + Hybrid Ensemble → `benchmark_results/*.csv` |
| `serve` | start the FastAPI server on `--port` (default 8000) |
| `report` | run the 50/50 batch audit + write the 7-page SOC PDF |
| `bot` | run the Telegram live monitor (needs `CYBERSHIELD_BOT_TOKEN`) |

Common flags: `--data-path`, `--store-dir`, `--out-dir`, `--port`, `--force`.

## Outputs

- `data/feature_store/nids_features_latest/` — `X/y_{train,val,test}.npy` + metadata
- `benchmark_results/` — benchmark CSV, diagnostics CSV, PNG figures, SOC PDF

## Telegram bot

1. Create a bot via `@BotFather`, get the token + your chat id.
2. Export `CYBERSHIELD_BOT_TOKEN` and `CYBERSHIELD_ADMIN_CHAT_ID` (or Kaggle Secrets).
3. `python main.py --mode bot` — press **Block IP** / **Report** / **Ignore** inline.

Without a token the monitor runs in simulation mode (alerts printed, no polling).

## Troubleshooting

- `ModuleNotFoundError: src...` — run from the repo root (or `pip install -e .` equivalent `sys.path` is handled by `main.py`).
- Kaggle `ModuleNotFoundError` — re-upload the repo dataset; Cell 1 copies it to `/kaggle/working`.
- Out of memory on scaling — the scaler fits on train only by design; reduce `--data-path` days.
