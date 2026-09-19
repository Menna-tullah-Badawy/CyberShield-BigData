"""
Health & Metrics routes — GET /health + GET /api/metrics (نفس Cell 4).
"""

from fastapi import APIRouter, Request

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health(request: Request):
    ctx = request.app.state.ctx
    return {"status": "healthy", "model": "Mamba SSM",
            "threats_in_kb": len(ctx["threat_kb"]),
            "threats_detected": len(ctx["threat_types_detected"]),
            "input": [ctx["api_sl"], ctx["api_nf"]]}


@router.get("/api/metrics")
async def metrics(request: Request):
    ctx = request.app.state.ctx
    bs = ctx["batch_stats"]
    return {"model": "Mamba SSM",
            "batch_test": {"TP": int(bs["tp"]), "FP": int(bs["fp"]),
                          "TN": int(bs["tn"]), "FN": int(bs["fn"]),
                          "ROC_AUC": round(float(bs["roc"]), 4)},
            "threat_types_detected": len(ctx["threat_types_detected"])}
