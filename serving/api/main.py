"""
Enterprise FastAPI Serving Application Entrypoint.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from serving.api.routes import health, predict, explain
from src.common.logger import get_logger

logger = get_logger("CyberShield-API")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 70)
    logger.info("🚀 تم تشغيل CyberShield REST API Serving Engine بنجاح على المنفذ 8000.")
    logger.info("📖 وثائق الـ API متاحة عبر الرابط: http://localhost:8000/docs")
    logger.info("=" * 70)
    yield


app = FastAPI(
    title="🛡️ CyberShield-BigData: Hybrid NIDS & MLOps Serving API",
    description="Production-grade real-time inference microservice with Explainable AI & GenAI Incident Reporting.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(explain.router)


@app.get("/", tags=["Root"])
def root_index():
    return {
        "service": "CyberShield-BigData NIDS Serving Layer",
        "status": "OPERATIONAL",
        "endpoints": {
            "health": "/health/live",
            "predict_single": "/predict/packet",
            "predict_batch": "/predict/batch",
            "explain_xai": "/explain/packet",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serving.api.main:app", host="127.0.0.1", port=8000, reload=False)