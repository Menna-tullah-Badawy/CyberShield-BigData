"""
Tabular Models: Flow Statistical Features + XGBoost/CatBoost/LightGBM.
نقل حرفي من Cell 1 في cybershield.ipynb (الـ Hyperparameters بالنص).
"""

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def extract_flow_stats(X_3d: np.ndarray) -> np.ndarray:
    """آخر قيمة + mean + max + min + std لكل سلسلة (دالة extract_features بالنص)."""
    f_last = X_3d[:, -1, :]
    f_mean = np.mean(X_3d, axis=1)
    f_max = np.max(X_3d, axis=1)
    f_min = np.min(X_3d, axis=1)
    f_std = np.std(X_3d, axis=1)
    return np.hstack([f_last, f_mean, f_max, f_min, f_std])


def build_xgb() -> Any:
    return xgb.XGBClassifier(
        n_estimators=350,
        learning_rate=0.035,
        max_depth=5,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_lambda=8.0,
        min_child_weight=4,
        scale_pos_weight=4.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="hist",
    )


def build_catboost(use_gpu: bool) -> Any:
    return cb.CatBoostClassifier(
        iterations=400,
        learning_rate=0.04,
        depth=5,
        l2_leaf_reg=8.0,
        bootstrap_type='Bernoulli',
        subsample=0.80,
        scale_pos_weight=4.0,
        verbose=0,
        random_seed=42,
        task_type='GPU' if use_gpu else 'CPU',
    )


def build_lgbm() -> Any:
    return lgb.LGBMClassifier(
        n_estimators=350,
        learning_rate=0.035,
        max_depth=5,
        num_leaves=31,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_lambda=8.0,
        min_child_samples=40,
        scale_pos_weight=4.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def train_tabular_model(
    name: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    use_gpu: bool = False,
) -> Tuple[Any, float]:
    """تدريب موديل tabular واحد وإرجاع (الموديل، زمن التدريب)."""
    t0 = time.time()
    if name == "XGBoost":
        model = build_xgb()
        model.fit(X_tr, y_tr)
    elif name == "CatBoost":
        model = build_catboost(use_gpu)
        if X_val is not None and y_val is not None:
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val),
                      early_stopping_rounds=30, verbose=False)
        else:
            model.fit(X_tr, y_tr, verbose=False)
    elif name == "LightGBM":
        model = build_lgbm()
        model.fit(X_tr, y_tr)
    else:
        raise ValueError(f"Unknown tabular model: {name}")
    return model, time.time() - t0


def available_tabular_models() -> Dict[str, bool]:
    return {"XGBoost": HAS_XGB, "CatBoost": HAS_CATBOOST, "LightGBM": HAS_LGBM}
