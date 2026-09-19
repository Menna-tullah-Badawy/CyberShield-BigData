"""
End-to-End Pipeline Orchestrator — Sweet-Spot Benchmark runner.
نقل حرفي من Cell 1: موازنة 1:4 → RobustScaler → Tabular → DL → Ensemble → CSV.
"""

import os
from typing import Any, Dict

import numpy as np
import torch

from src.evaluation.metrics import CyberEvaluationMetrics
from src.evaluation.threshold_optimizer import optimize_security_threshold
from src.feature_engineering.aggregations import FeatureAggregator
from src.feature_engineering.scaling import FeatureScaler
from src.feature_store.data_splitter import DataSplitter
from src.feature_store.transformations import FeatureStoreManager
from src.models.candidate_models import CandidateModelFactory
from src.models.dl_trainer import DLTrainer
from src.models.model_selector import HYBRID_WEIGHTS, ModelSelector
from src.models.sequence_dataset import get_temporal_dataloaders, to_device_tensors


class EndToEndPipelineOrchestrator:
    """منسق البنش مارك الكامل (نفس تدفق Cell 1 بالنص)."""

    def __init__(self, store_dir: str = "data/feature_store/nids_features_latest",
                 out_dir: str = "benchmark_results",
                 device: torch.device = None, seed: int = 42):
        self.store = FeatureStoreManager(store_dir)
        self.out_dir = out_dir
        self.seed = seed
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run_training_pipeline(self, verbose: bool = True) -> Dict[str, Any]:
        """تشغيل البنش مارك الكامل وإرجاع كل المخرجات."""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        os.makedirs(self.out_dir, exist_ok=True)

        X_train_raw, y_train_raw, X_val_raw, y_val, X_test_raw, y_test = \
            self.store.load_sequences()
        if verbose:
            print(f"✅ Data verified. Train shape: {X_train_raw.shape}")
            print(f"⚡ Device: {self.device}")

        X_train, y_train = DataSplitter.balance_train_1_to_4(
            X_train_raw, y_train_raw, seed=self.seed)
        scaler = FeatureScaler()
        X_train_3d, X_val_3d, X_test_3d = scaler.scale_all(X_train, X_val_raw, X_test_raw)

        X_train_t, y_train_t = to_device_tensors(X_train_3d, y_train, self.device)
        X_val_t, y_val_t = to_device_tensors(X_val_3d, y_val, self.device)
        X_test_t, y_test_t = to_device_tensors(X_test_3d, y_test, self.device)
        train_loader, val_loader, test_loader = get_temporal_dataloaders(
            X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, y_test_t)

        if verbose:
            print("🧪 Extracting flow statistical features...")
        X_tr_tab = FeatureAggregator.extract_flow_stats(X_train_3d)
        X_val_tab = FeatureAggregator.extract_flow_stats(X_val_3d)
        X_te_tab = FeatureAggregator.extract_flow_stats(X_test_3d)

        results: Dict[str, Dict[str, float]] = {}
        val_predictions: Dict[str, np.ndarray] = {}
        test_predictions: Dict[str, np.ndarray] = {}
        fitted: Dict[str, Any] = {}

        if verbose:
            print("\n🌲 1. Training Tabular Models...")
        use_gpu = torch.cuda.is_available()
        for name, available in CandidateModelFactory.available().items():
            if not available:
                continue
            model, t_sec = CandidateModelFactory.train(
                name, X_tr_tab, y_train, X_val_tab, y_val, use_gpu=use_gpu)
            fitted[name] = model
            vp = model.predict_proba(X_val_tab)[:, 1]
            tp = model.predict_proba(X_te_tab)[:, 1]
            val_predictions[name] = vp
            test_predictions[name] = tp
            met = optimize_security_threshold(y_test, tp, y_val, vp)
            met["Train_Time_Sec"] = round(t_sec, 2)
            met["Latency_ms"] = float("nan")
            results[name] = met
            if verbose:
                print(f"   ✅ {name} | Recall: {met['Recall'] * 100:.1f}% | "
                      f"Precision: {met['Precision'] * 100:.1f}% | ROC: {met['ROC-AUC']}")

        if verbose:
            print("\n🚀 2. Training Deep Learning Models...")
        trainer = DLTrainer(self.device)
        for name, model in DLTrainer.build_dl_dict(X_train_3d.shape[2], self.device).items():
            vp, tp, t_sec, lat = trainer.fit_and_evaluate(
                model, train_loader, val_loader, test_loader, len(y_test))
            fitted[name] = model
            val_predictions[name] = vp
            test_predictions[name] = tp
            met = optimize_security_threshold(y_test, tp, y_val, vp)
            met["Train_Time_Sec"] = round(t_sec, 2)
            met["Latency_ms"] = round(lat, 4)
            results[name] = met
            if verbose:
                print(f"   ✅ {name} | Recall: {met['Recall'] * 100:.1f}% | "
                      f"Precision: {met['Precision'] * 100:.1f}% | ROC: {met['ROC-AUC']}")

        if verbose:
            print("\n🏆 3. Evaluating Balanced Hybrid Ensemble...")
        vp_ens = ModelSelector.hybrid_ensemble_predict(val_predictions, HYBRID_WEIGHTS)
        tp_ens = ModelSelector.hybrid_ensemble_predict(test_predictions, HYBRID_WEIGHTS)
        test_predictions["Hybrid_Ensemble"] = tp_ens
        met_ens = optimize_security_threshold(y_test, tp_ens, y_val, vp_ens)
        met_ens["Train_Time_Sec"] = round(
            sum(results[k]["Train_Time_Sec"] for k in HYBRID_WEIGHTS), 2)
        met_ens["Latency_ms"] = round(results["CNN_BiLSTM"]["Latency_ms"], 4)
        results["Hybrid_Ensemble"] = met_ens
        if verbose:
            print(f"   ✅ Hybrid Ensemble | Recall: {met_ens['Recall'] * 100:.1f}% | "
                  f"Precision: {met_ens['Precision'] * 100:.1f}% | ROC: {met_ens['ROC-AUC']}")

        df = CyberEvaluationMetrics.save_and_report(results, self.out_dir, verbose=verbose)
        return {
            "results": results, "df": df,
            "val_predictions": val_predictions, "test_predictions": test_predictions,
            "models": fitted, "feature_scaler": scaler,
            "X_train_t": X_train_t, "y_train_t": y_train_t,
            "X_tr_tab": X_tr_tab, "y_train": y_train,
            "y_val": y_val, "y_test": y_test,
            "device": self.device,
        }
