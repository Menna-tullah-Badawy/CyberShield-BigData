"""
Data Ingestion Engine (pandas/numpy track).
منطق Cell 0 من cybershield.ipynb: تنزيل CICIDS-2018 + اكتشاف ملفات الأيام + قراءتها.
"""

import glob
import os
from typing import List, Optional

import pandas as pd

KAGGLE_DATASET_ID = "solarmainframe/ids-intrusion-csv"


class DataIngestionEngine:
    """محرك استيراد البيانات اليومية (يوم = ملف CSV)."""

    def __init__(self, kaggle_dataset_id: str = KAGGLE_DATASET_ID):
        self.kaggle_dataset_id = kaggle_dataset_id

    def download_dataset(self, verbose: bool = True) -> str:
        """تنزيل الداتاسيت عبر kagglehub (نفس سلوك النوت بوك)."""
        try:
            import kagglehub
        except ImportError:
            os.system("pip install -q kagglehub")
            import kagglehub
        if verbose:
            print("\n>>> Downloading CICIDS 2018 dataset...")
        return kagglehub.dataset_download(self.kaggle_dataset_id)

    @staticmethod
    def discover_csv_files(csv_dir: str) -> List[str]:
        """اكتشاف ملفات CSV (مع بحث متكرر عند الحاجة) — نفس كود النوت بوك."""
        csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
        if not csv_files:
            csv_files = glob.glob(os.path.join(csv_dir, "**", "*.csv"), recursive=True)
        return sorted(csv_files)

    @staticmethod
    def read_day_csv(file_path: str) -> pd.DataFrame:
        """قراءة ملف يوم واحد وتنظيف أسماء الأعمدة."""
        df_day = pd.read_csv(file_path, low_memory=False)
        df_day.columns = df_day.columns.str.strip()
        return df_day

    def load_or_fetch_dataset(
        self, csv_dir: Optional[str] = None, verbose: bool = True
    ) -> List[str]:
        """إرجاع قائمة ملفات الأيام مرتبة (تنزيل تلقائي على Kaggle)."""
        if csv_dir is None:
            csv_dir = self.download_dataset(verbose=verbose)
        csv_files = self.discover_csv_files(csv_dir)
        if verbose:
            print(f"📅 Found {len(csv_files)} days of traffic.")
        return csv_files
