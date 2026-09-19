"""
CyberShield NIDS Pipeline — Modular port of `cybershield.ipynb` (Kaggle reference).

كل موديول هنا هو نقل حرفي (Logic Faithful) لخلية من خلايا النوت بوك المرجعية:
    Cell 0 → temporal_dataset   (Leakage-Proof Temporal Feature Extraction & Split)
    Cell 1 → benchmark          (Sweet-Spot Benchmark: Tabular + DL + Hybrid Ensemble)
    Cell 2 → diagnostics        (IEEE/ACM Generalization Diagnostic Suite)
    Cell 3 → figures            (Publication-Quality Visualizations)
    Cell 4 → threat_kb, xai, api, soc_report (RAG + XAI + FastAPI + 7-page PDF)
    Cell 7 → telegram_bot       (SOC Telegram Live Monitor + reconstructed cells)

الثوابت والـ Hyperparameters منقولة بالنص من النوت بوك لضمان نفس النتائج.
"""

__version__ = "1.0.0"
__source_notebook__ = "cybershield.ipynb"

from src.nids import temporal_dataset  # noqa: F401
from src.nids import threshold  # noqa: F401
from src.nids import ensemble  # noqa: F401
from src.nids import threat_kb  # noqa: F401

__all__ = [
    "temporal_dataset",
    "threshold",
    "ensemble",
    "threat_kb",
    "tabular_models",
    "dl_trainer",
    "benchmark",
    "diagnostics",
    "figures",
    "xai",
    "soc_report",
    "api",
    "telegram_bot",
]
