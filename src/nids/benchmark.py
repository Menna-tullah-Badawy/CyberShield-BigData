"""
Target Sweet-Spot Benchmark orchestrator (Precision>=92% & Recall>=90%).
نقل حرفي من Cell 1 في cybershield.ipynb.

الخطوات: موازنة 1:4 → RobustScaler(5,95)+Clip → Tabular → DL → Ensemble → CSV
"""

import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset

from src.nids import ensemble as ens_mod
from src.nids import tabular_models as tab_mod
from src.nids import temporal_dataset as td_mod
from src.nids import threshold as th_mod
from src.nids.dl_trainer import build_dl_dict, train_dl_model

NEG_POS_RATIO = 4          # موازنة فئات التدريب بنسبة 1:4
SCALER_QMIN, SCALER_QMAX = 5.0, 95.0
SCALER_CLIP = 6.0
BATCH_SIZE = 512
BENCHMARK_CSV = "production_balanced_benchmark.csv"


def balance_train_1_to_4(
    X_train_raw: np.ndarray, y_train_raw: np.ndarray, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Undersampling للـ Negatives بنسبة 1:4 (نفس كود النوت بوك)."""
    rng = np.random.RandomState(seed)
    pos_idx = np.where(y_train_raw == 1)[0]
    neg_idx = np.where(y_train_raw == 0)[0]
    sampled_neg = rng.choice(neg_idx, size=min(len(neg_idx), len(pos_idx) * NEG_POS_RATIO),
                             replace=False)
    train_idx = np.concatenate([pos_idx, sampled_neg])
    rng.shuffle(train_idx)
    return X_train_raw[train_idx], y_train_raw[train_idx]


def scale_sequences(
    X_train: np.ndarray, X_val_raw: np.ndarray, X_test_raw: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, RobustScaler]:
    """RobustScaler على الـ Train فقط + Clip (نفس كود النوت بوك)."""
    num_features = X_train.shape[2]
    scaler = RobustScaler(quantile_range=(SCALER_QMIN, SCALER_QMAX))
    X_train_flat = np.clip(scaler.fit_transform(X_train.reshape(-1, num_features)),
                           -SCALER_CLIP, SCALER_CLIP)
    X_val_flat = np.clip(scaler.transform(X_val_raw.reshape(-1, num_features)),
                         -SCALER_CLIP, SCALER_CLIP)
    X_test_flat = np.clip(scaler.transform(X_test_raw.reshape(-1, num_features)),
                          -SCALER_CLIP, SCALER_CLIP)
    seq_len = X_train.shape[1]
    X_train_3d = X_train_flat.reshape(X_train.shape[0], seq_len, num_features)
    X_val_3d = X_val_flat.reshape(X_val_raw.shape[0], seq_len, num_features)
    X_test_3d = X_test_flat.reshape(X_test_raw.shape[0], seq_len, num_features)
    return X_train_3d, X_val_3d, X_test_3d, scaler


def run_benchmark(
    store_dir: str,
    out_dir: str,
    device: torch.device = None,  # type: ignore[assignment]
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    تشغيل البنش مارك الكامل وإرجاع كل المخرجات:
    results, val_predictions, test_predictions, models, scaler, tensors...
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.makedirs(out_dir, exist_ok=True)

    # ── تحميل البيانات ──
    X_train_raw, y_train_raw, X_val_raw, y_val, X_test_raw, y_test = td_mod.load_sequences(store_dir)
    if verbose:
        print(f"✅ Data verified. Train shape: {X_train_raw.shape}")
        print(f"⚡ Device: {device}")

    # ── موازنة + تقييس ──
    X_train, y_train = balance_train_1_to_4(X_train_raw, y_train_raw, seed=seed)
    num_samples_tr, seq_len, num_features = X_train.shape
    X_train_3d, X_val_3d, X_test_3d, scaler = scale_sequences(X_train, X_val_raw, X_test_raw)

    X_train_t = torch.tensor(X_train_3d, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_val_t = torch.tensor(X_val_3d, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)
    X_test_t = torch.tensor(X_test_3d, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.long, device=device)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)

    if verbose:
        print("🧪 Extracting flow statistical features...")
    X_tr_tab = tab_mod.extract_flow_stats(X_train_3d)
    X_val_tab = tab_mod.extract_flow_stats(X_val_3d)
    X_te_tab = tab_mod.extract_flow_stats(X_test_3d)

    results: Dict[str, Dict[str, float]] = {}
    val_predictions: Dict[str, np.ndarray] = {}
    test_predictions: Dict[str, np.ndarray] = {}
    fitted: Dict[str, Any] = {}

    # ── Tabular ──
    if verbose:
        print("\n🌲 1. Training Tabular Models...")
    use_gpu = torch.cuda.is_available()
    for name, available in tab_mod.available_tabular_models().items():
        if not available:
            continue
        model, t_sec = tab_mod.train_tabular_model(
            name, X_tr_tab, y_train, X_val_tab, y_val, use_gpu=use_gpu)
        fitted[name] = model
        vp = model.predict_proba(X_val_tab)[:, 1]
        tp = model.predict_proba(X_te_tab)[:, 1]
        val_predictions[name] = vp
        test_predictions[name] = tp
        met = th_mod.optimize_security_threshold(y_test, tp, y_val, vp)
        met["Train_Time_Sec"] = round(t_sec, 2)
        met["Latency_ms"] = float("nan")
        results[name] = met
        if verbose:
            print(f"   ✅ {name} | Recall: {met['Recall'] * 100:.1f}% | "
                  f"Precision: {met['Precision'] * 100:.1f}% | ROC: {met['ROC-AUC']}")

    # ── Deep Learning ──
    if verbose:
        print("\n🚀 2. Training Deep Learning Models...")
    dl_dict = build_dl_dict(num_features, device)
    for name, model in dl_dict.items():
        vp, tp, t_sec, lat = train_dl_model(
            model, train_loader, val_loader, test_loader, device, len(y_test))
        fitted[name] = model
        val_predictions[name] = vp
        test_predictions[name] = tp
        met = th_mod.optimize_security_threshold(y_test, tp, y_val, vp)
        met["Train_Time_Sec"] = round(t_sec, 2)
        met["Latency_ms"] = round(lat, 4)
        results[name] = met
        if verbose:
            print(f"   ✅ {name} | Recall: {met['Recall'] * 100:.1f}% | "
                  f"Precision: {met['Precision'] * 100:.1f}% | ROC: {met['ROC-AUC']}")

    # ── Hybrid Ensemble ──
    if verbose:
        print("\n🏆 3. Evaluating Balanced Hybrid Ensemble...")
    w = ens_mod.HYBRID_WEIGHTS
    vp_ens = ens_mod.hybrid_ensemble_predict(val_predictions, w)
    tp_ens = ens_mod.hybrid_ensemble_predict(test_predictions, w)
    test_predictions["Hybrid_Ensemble"] = tp_ens
    met_ens = th_mod.optimize_security_threshold(y_test, tp_ens, y_val, vp_ens)
    met_ens["Train_Time_Sec"] = round(sum(results[k]["Train_Time_Sec"] for k in w), 2)
    met_ens["Latency_ms"] = round(results["CNN_BiLSTM"]["Latency_ms"], 4)
    results["Hybrid_Ensemble"] = met_ens
    if verbose:
        print(f"   ✅ Hybrid Ensemble | Recall: {met_ens['Recall'] * 100:.1f}% | "
              f"Precision: {met_ens['Precision'] * 100:.1f}% | ROC: {met_ens['ROC-AUC']}")

    # ── حفظ المصفوفة ──
    df = pd.DataFrame(results).T
    if verbose:
        print("\n" + "=" * 115)
        print("🏆 SOTA CYBERSHIELD BENCHMARK MATRIX (PRECISION >= 92% & RECALL >= 90%)")
        print("=" * 115)
        print(df.to_string())
    csv_path = os.path.join(out_dir, BENCHMARK_CSV)
    df.to_csv(csv_path)
    if verbose:
        print(f"\n📁 Saved: {csv_path}")
        best_model_name = df["F1-Score"].idxmax()
        print(f"\n🌟 Top Balanced Model: {best_model_name} "
              f"(Recall: {df.loc[best_model_name, 'Recall'] * 100:.2f}%, "
              f"Precision: {df.loc[best_model_name, 'Precision'] * 100:.2f}%)")

    return {
        "results": results,
        "df": df,
        "val_predictions": val_predictions,
        "test_predictions": test_predictions,
        "models": fitted,
        "scaler": scaler,
        "X_train_t": X_train_t, "y_train_t": y_train_t,
        "X_tr_tab": X_tr_tab, "y_train": y_train,
        "y_val": y_val, "y_test": y_test,
        "num_features": num_features, "seq_len": seq_len,
        "device": device, "csv_path": csv_path,
    }
