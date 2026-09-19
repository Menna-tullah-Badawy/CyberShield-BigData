"""
Threat-intel route — GET /api/threats (نفس Cell 4).
"""

from fastapi import APIRouter, Request

router = APIRouter(tags=["ThreatIntel"])


@router.get("/api/threats")
async def get_all_threats(request: Request):
    """Returns ALL threat types, indicators, and mitigations."""
    ctx = request.app.state.ctx
    return {"threats": ctx["threat_kb"], "detected": ctx["threat_types_detected"]}
