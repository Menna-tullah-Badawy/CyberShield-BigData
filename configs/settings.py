"""
CyberShield-BigData global paths (torch track).
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
STORE_DIR = os.path.join(DATA_DIR, "feature_store", "nids_features_latest")
OUT_DIR = os.path.join(BASE_DIR, "benchmark_results")

REPORTS_DIR = os.path.join(BASE_DIR, "reports")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MODELS_DIR = os.path.join(BASE_DIR, "models")

KAGGLE_PROJECT_DATASET = "mennatullahbadawy/cybershield-bigdata-project"
KAGGLE_IDS_DATASET = "solarmainframe/ids-intrusion-csv"

for directory_path in [RAW_DATA_DIR, STORE_DIR, OUT_DIR, REPORTS_DIR, LOGS_DIR, MODELS_DIR]:
    os.makedirs(directory_path, exist_ok=True)
