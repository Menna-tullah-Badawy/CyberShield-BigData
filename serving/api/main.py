"""
CyberShield NIDS FastAPI application.
نفس endpoints في Cell 4: /health, /api/threats, /api/metrics, POST /api/detect.

- `create_app(ctx)`: بناء التطبيق من سياق جاهز (النوت بوك / الاختبارات).
- `app`: نسخة عامة تُبنى عند الاستيراد (uvicorn serving.api.main:app) —
  تُحمّل الموديل من الـ Store إن وُجد، وإلا تعمل بموديل جديد مع تحذير.
"""

import os
import socket
import threading
import time
from contextlib import asynccontextmanager
from typing import Dict

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from serving.api.routes import explain, health, predict
from src.common.logger import get_logger
from src.genai_reporting.mitre_mapping import THREAT_KB, HybridThreatRetriever
from src.models.deep_learning_models import MambaNIDS

logger = get_logger("CyberShield-API")

API_TITLE = "CyberShield NIDS API"
API_VERSION = "3.0.0"


def build_default_ctx(store_dir: str = None) -> Dict:
    """بناء سياق التشغيل الافتراضي من الـ Feature Store (مع بدائل آمنة)."""
    from sklearn.preprocessing import RobustScaler

    from src.feature_store.transformations import METADATA_FILE, FeatureStoreManager

    store_dir = store_dir or os.environ.get(
        "CYBERSHIELD_STORE", "data/feature_store/nids_features_latest")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = FeatureStoreManager(store_dir)

    if store.exists():
        X_chk = np.load(os.path.join(store_dir, "X_train.npy"), mmap_mode='r')
        api_nf, api_sl = X_chk.shape[2], X_chk.shape[1]
        feature_names = store.load_metadata().get("feature_names") or []
        scaler = RobustScaler(quantile_range=(5.0, 95.0))
        scaler.fit(np.load(os.path.join(store_dir, "X_train.npy")).reshape(-1, api_nf))
        logger.info(f"⚡ Store loaded: input=({api_sl}, {api_nf})")
    else:
        api_nf, api_sl, feature_names = 66, 10, []
        scaler = RobustScaler(quantile_range=(5.0, 95.0))
        logger.warn("⚠️ Feature Store not found — API running with untrained model.")

    while len(feature_names) < api_nf:
        feature_names.append(f"Feature_{len(feature_names)}")

    mamba = MambaNIDS(input_dim=api_nf, d_model=64, d_state=16,
                      num_layers=2, num_classes=2, dropout=0.2).to(device)
    ckpt_path = os.path.join(os.environ.get("CYBERSHIELD_OUT", "benchmark_results"),
                             "mamba_ssm_best.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        try:
            mamba.load_state_dict(ckpt['model_state_dict'])
            logger.info("✅ Champion model loaded")
        except Exception:
            logger.warn("⚠️ Fresh model (checkpoint mismatch)")
    mamba.eval()

    return {
        "mamba": mamba, "scaler": scaler, "feature_names": feature_names,
        "api_sl": api_sl, "api_nf": api_nf, "device": device,
        "retriever": HybridThreatRetriever(), "threat_kb": THREAT_KB,
        "threat_types_detected": {},
        "batch_stats": {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "roc": 0.0},
        "inc_cnt": [1],
    }


def create_app(ctx: Dict = None) -> FastAPI:
    """مصنع التطبيق (نفس endpoints النوت بوك)."""
    ctx = ctx or build_default_ctx()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("=" * 70)
        logger.info("🚀 CyberShield NIDS API up — input=%s", [ctx["api_sl"], ctx["api_nf"]])
        logger.info("📖 Docs: http://localhost:8000/docs")
        logger.info("=" * 70)
        yield

    app = FastAPI(title=API_TITLE, version=API_VERSION, docs_url="/docs",
                  redoc_url="/redoc", lifespan=lifespan)
    app.state.ctx = ctx
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])
    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(explain.router)

    @app.get("/", tags=["Root"])
    def root_index():
        return {"service": "CyberShield NIDS Serving Layer", "status": "OPERATIONAL",
                "endpoints": {"health": "/health", "metrics": "/api/metrics",
                              "threats": "/api/threats", "detect": "/api/detect",
                              "docs": "/docs"}}
    return app


def find_free_port(start: int = 8000, end: int = 8020) -> int:
    """نفس دالة find_port من النوت بوك."""
    for p in range(start, end):
        try:
            sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sk.bind(('0.0.0.0', p))
            sk.close()
            return p
        except OSError:
            continue
    return end + 1


def serve_in_background(app: FastAPI, port: int = None, wait_sec: float = 3.0) -> int:
    """تشغيل السيرفر في Thread خلفي (نفس سلوك النوت بوك)."""
    import uvicorn

    port = port or find_free_port()
    threading.Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning"),
        daemon=True).start()
    time.sleep(wait_sec)
    print(f"\n🚀 API: http://localhost:{port}")
    print(f"📚 Docs: http://localhost:{port}/docs")
    print("📡 /api/threats — All threat types + indicators + mitigations")
    print("📡 /api/metrics — Detection metrics")
    return port


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serving.api.main:app", host="127.0.0.1", port=8000, reload=False)
