"""
Detection route — POST /api/detect (نفس Cell 4: Mamba + XAI + RAG + iptables).
"""

import numpy as np
import torch
from fastapi import APIRouter, HTTPException, Request

from serving.api.schemas.request_response import DetectRequest
from src.explainability.xai_engine import ExplainabilityEngine

router = APIRouter(tags=["Detection"])

DETECT_THRESHOLD = 0.03
SCALE_CLIP = 15.0


@router.post("/api/detect")
async def detect(req: DetectRequest, request: Request):
    ctx = request.app.state.ctx
    mamba, scaler = ctx["mamba"], ctx["scaler"]
    feature_names, api_sl, api_nf, device = (ctx["feature_names"], ctx["api_sl"],
                                             ctx["api_nf"], ctx["device"])
    retriever = ctx["retriever"]
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
        xai = ExplainabilityEngine(feature_names, retriever)
        expl = xai.generate_incident_explanation(mamba, x.clone(), feature_names, n=5)
        rag = expl["rag"]
        ipt_lines = [r.replace("{src_ip}", req.src_ip)
                     for r in rag["doc"].get("iptables", [])]
        iid = f"INC-2026-NIDS-{ctx['inc_cnt'][0]:05d}"
        ctx["inc_cnt"][0] += 1
        return {"incident_id": iid,
                "detection": {"is_threat": True, "probability": round(ap, 4)},
                "threat": {"id": rag["doc"]["threat_id"], "title": rag["doc"]["title"],
                          "tactic": rag["doc"]["tactic"],
                          "indicators": rag["doc"].get("indicators", ""),
                          "playbook": rag["doc"]["playbook"]},
                "xai": {"top_features": expl["top_features"]},
                "rag": {"hybrid_score": round(rag["hybrid_score"], 4)},
                "response": {"iptables": ipt_lines}}
    except Exception as e:
        raise HTTPException(400, detail=str(e))
