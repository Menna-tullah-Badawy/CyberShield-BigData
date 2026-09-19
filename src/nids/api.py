"""
CyberShield NIDS FastAPI (Mamba + XAI + RAG + iptables).
نقل حرفي من Cell 4 في cybershield.ipynb:
    GET  /health, /api/threats, /api/metrics  +  POST /api/detect
"""

import socket
import threading
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from src.nids.threat_kb import THREAT_KB, HybridThreatRetriever
from src.nids.xai import extract_xai

API_TITLE = "CyberShield NIDS API"
API_VERSION = "3.0.0"
DETECT_THRESHOLD = 0.03   # نفس عتبة النوت بوك
SCALE_CLIP = 15.0         # نفس الـ Clip في Cell 4 (±15)


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


def create_app(
    mamba: torch.nn.Module,
    scaler,
    feature_names: Sequence[str],
    api_sl: int,
    api_nf: int,
    device: torch.device,
    batch_stats: Dict[str, float],
    threat_types_detected: Dict[str, dict],
    retriever: Optional[HybridThreatRetriever] = None,
):
    """بناء تطبيق FastAPI بالنص كما في النوت بوك."""
    if not HAS_FASTAPI:
        raise ImportError("fastapi/pydantic مطلوبة: pip install fastapi uvicorn pydantic")

    threat_kb = THREAT_KB
    rag = retriever or HybridThreatRetriever()

    app = FastAPI(title=API_TITLE, version=API_VERSION)
    inc_cnt = [1]

    class Req(BaseModel):
        features: List[List[float]]
        src_ip: Optional[str] = "unknown"
        dst_port: Optional[int] = 0
        protocol: Optional[str] = "TCP"

    @app.get("/health")
    async def health():
        return {"status": "healthy", "model": "Mamba SSM",
                "threats_in_kb": len(threat_kb),
                "threats_detected": len(threat_types_detected),
                "input": [api_sl, api_nf]}

    @app.get("/api/threats")
    async def get_all_threats():
        """Returns ALL threat types, indicators, and mitigations."""
        return {"threats": threat_kb, "detected": threat_types_detected}

    @app.get("/api/metrics")
    async def metrics():
        return {"model": "Mamba SSM",
                "batch_test": {"TP": int(batch_stats["tp"]), "FP": int(batch_stats["fp"]),
                              "TN": int(batch_stats["tn"]), "FN": int(batch_stats["fn"]),
                              "ROC_AUC": round(float(batch_stats["roc"]), 4)},
                "threat_types_detected": len(threat_types_detected)}

    @app.post("/api/detect")
    async def detect(req: Req):
        try:
            feat = np.array(req.features, dtype=np.float32)
            scaled = np.clip(scaler.transform(feat.reshape(-1, api_nf)),
                             -SCALE_CLIP, SCALE_CLIP).reshape(1, api_sl, api_nf)
            x = torch.tensor(scaled, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits = mamba(x)
                probs = torch.softmax(logits, 1)
                ap = float(probs[0, 1].cpu().numpy())
            if ap < DETECT_THRESHOLD:
                return {"detection": {"is_threat": False, "probability": round(ap, 4)},
                        "verdict": "BENIGN"}
            xf = extract_xai(mamba, x.clone(), feature_names, n=5)
            qt = " ".join(f["name"] for f in xf).lower().replace("_", " ")
            res = rag.retrieve(qt)
            ipt_lines = [r.replace("{src_ip}", req.src_ip)
                         for r in res["doc"].get("iptables", [])]
            iid = f"INC-2026-NIDS-{inc_cnt[0]:05d}"
            inc_cnt[0] += 1
            return {"incident_id": iid,
                    "detection": {"is_threat": True, "probability": round(ap, 4)},
                    "threat": {"id": res["doc"]["threat_id"], "title": res["doc"]["title"],
                              "tactic": res["doc"]["tactic"],
                              "indicators": res["doc"].get("indicators", ""),
                              "playbook": res["doc"]["playbook"]},
                    "xai": {"top_features": xf},
                    "rag": {"hybrid_score": round(res["hybrid_score"], 4)},
                    "response": {"iptables": ipt_lines}}
        except Exception as e:
            raise HTTPException(400, detail=str(e))

    return app


def serve_in_background(app, port: Optional[int] = None, wait_sec: float = 3.0) -> int:
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
