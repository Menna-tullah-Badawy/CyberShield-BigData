"""
Feature Pipeline Builder — Cell 0 orchestrator.
يربط: Ingestion → Quality → Datetime → Target → Exclude → Cleaning →
Temporal Split → Sequences → Zero-Var Filter → Feature Store.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.cleaning.cleaning_pipeline import CleaningPipeline
from src.data_pipeline.ingestion import DataIngestionEngine
from src.feature_engineering.aggregations import SEQ_LEN, FeatureAggregator
from src.feature_engineering.datetime_features import SAMPLE_STEP, DatetimeFeatureGenerator
from src.feature_engineering.selection import STRICT_EXCLUDE, FeatureSelector
from src.feature_store.transformations import NPY_FILES, FeatureStoreManager
from src.quality.engine import DataQualityEngine

TRAIN_RATIO = 0.70
VAL_RATIO = 0.85  # أي: train 70% / val 15% / test 15%


class FeaturePipelineBuilder:
    """باني خط استخراج الخصائص الزمنية (منطق Cell 0 بالنص)."""

    def __init__(self, store_dir: str = "data/feature_store/nids_features_latest"):
        self.store = FeatureStoreManager(base_path=store_dir)
        self.ingestion = DataIngestionEngine()
        self.quality = DataQualityEngine()
        self.cleaning = CleaningPipeline()

    def process_day(
        self, file_path: str, seq_len: int = SEQ_LEN, sample_step: int = SAMPLE_STEP,
    ) -> Optional[Tuple[Tuple[np.ndarray, np.ndarray], ...]]:
        """معالجة ملف يوم واحد كاملاً."""
        df_day = self.ingestion.read_day_csv(file_path)
        report = self.quality.run_all_checks(df_day, os.path.basename(file_path))
        ts_col, lbl_col = report["timestamp_column"], report["label_column"]

        if ts_col:
            df_day = DatetimeFeatureGenerator.parse_and_sort(df_day, ts_col)
        df_day = DatetimeFeatureGenerator.uniform_sample(df_day, sample_step)

        df_day['target'] = df_day[lbl_col].apply(
            lambda x: 0 if str(x).strip().lower() == 'benign' else 1)

        valid_cols = FeatureSelector.apply_strict_exclude(df_day)
        df_day = self.cleaning.run_cleaning_workflow(df_day, valid_cols)

        n = len(df_day)
        t_end, v_end = int(n * TRAIN_RATIO), int(n * VAL_RATIO)
        df_tr, df_va, df_te = df_day.iloc[:t_end], df_day.iloc[t_end:v_end], df_day.iloc[v_end:]

        x_tr, y_tr = FeatureAggregator.create_day_sequences(
            df_tr[valid_cols].to_numpy(dtype=np.float32),
            df_tr['target'].to_numpy(dtype=np.int64), seq_len)
        x_va, y_va = FeatureAggregator.create_day_sequences(
            df_va[valid_cols].to_numpy(dtype=np.float32),
            df_va['target'].to_numpy(dtype=np.int64), seq_len)
        x_te, y_te = FeatureAggregator.create_day_sequences(
            df_te[valid_cols].to_numpy(dtype=np.float32),
            df_te['target'].to_numpy(dtype=np.int64), seq_len)

        attacks = int((df_day['target'] == 1).sum())
        return (x_tr, y_tr), (x_va, y_va), (x_te, y_te), valid_cols, attacks

    def build_features(
        self, csv_dir: Optional[str] = None, seq_len: int = SEQ_LEN,
        sample_step: int = SAMPLE_STEP, verbose: bool = True,
    ) -> Dict[str, object]:
        """بناء الداتاسيت الزمنية كاملة وحفظها في الـ Feature Store."""
        csv_files = self.ingestion.load_or_fetch_dataset(csv_dir, verbose=verbose)

        train_x, train_y, val_x, val_y, test_x, test_y = [], [], [], [], [], []
        feature_names: List[str] = []

        for file in csv_files:
            day_name = os.path.basename(file)
            try:
                (x_tr, y_tr), (x_va, y_va), (x_te, y_te), valid_cols, attacks = self.process_day(
                    file, seq_len=seq_len, sample_step=sample_step)
                if not feature_names:
                    feature_names = list(valid_cols)
                if len(x_tr) > 0:
                    train_x.append(x_tr); train_y.append(y_tr)
                if len(x_va) > 0:
                    val_x.append(x_va); val_y.append(y_va)
                if len(x_te) > 0:
                    test_x.append(x_te); test_y.append(y_te)
                if verbose:
                    print(f"   ✅ Processed {day_name} | Attacks: {attacks}")
            except Exception as e:
                if verbose:
                    print(f"   ⚠️ Skipped {day_name}: {e}")

        X_train = np.concatenate(train_x, axis=0)
        y_train = np.concatenate(train_y, axis=0)
        X_val = np.concatenate(val_x, axis=0)
        y_val = np.concatenate(val_y, axis=0)
        X_test = np.concatenate(test_x, axis=0)
        y_test = np.concatenate(test_y, axis=0)

        X_train, X_val, X_test, kept = FeatureSelector.filter_zero_variance(
            X_train, X_val, X_test, feature_names)

        arrays = {"X_train": X_train, "y_train": y_train, "X_val": X_val,
                  "y_val": y_val, "X_test": X_test, "y_test": y_test}
        meta = self.store.save_sequences(arrays, kept, seq_len)

        if verbose:
            print("\n" + "=" * 50)
            print(f"🟢 Clean Train Shape: {X_train.shape} | Attacks: {(y_train == 1).sum()}")
            print(f"🔵 Clean Val Shape:   {X_val.shape}   | Attacks: {(y_val == 1).sum()}")
            print(f"🟠 Clean Test Shape:  {X_test.shape}  | Attacks: {(y_test == 1).sum()}")
            print(f"📊 Safe Feature Count: {X_train.shape[2]} (Zero Ports, Zero Metadata)")
            print("=" * 50)
        return meta
