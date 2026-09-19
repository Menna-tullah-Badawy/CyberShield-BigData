"""
Candidate Tabular Models Factory: XGBoost / CatBoost / LightGBM.
منطق Cell 1 من cybershield.ipynb (الـ Hyperparameters بالنص).
"""

from typing import Any, Dict, Optional, Tuple

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

TABULAR_NAMES = ("XGBoost", "CatBoost", "LightGBM")


class CandidateModelFactory:
    """مصنع النماذج الجدولية المرشحة بنفس إعدادات النوت بوك."""

    @staticmethod
    def available() -> Dict[str, bool]:
        return {"XGBoost": HAS_XGB, "CatBoost": HAS_CATBOOST, "LightGBM": HAS_LGBM}

    @staticmethod
    def build(name: str, use_gpu: bool = False) -> Any:
        if name == "XGBoost":
            return xgb.XGBClassifier(
                n_estimators=350, learning_rate=0.035, max_depth=5,
                subsample=0.80, colsample_bytree=0.80, reg_lambda=8.0,
                min_child_weight=4, scale_pos_weight=4.0, random_state=42,
                n_jobs=-1, eval_metric="logloss", tree_method="hist")
        if name == "CatBoost":
            return cb.CatBoostClassifier(
                iterations=400, learning_rate=0.04, depth=5, l2_leaf_reg=8.0,
                bootstrap_type='Bernoulli', subsample=0.80, scale_pos_weight=4.0,
                verbose=0, random_seed=42, task_type='GPU' if use_gpu else 'CPU')
        if name == "LightGBM":
            return lgb.LGBMClassifier(
                n_estimators=350, learning_rate=0.035, max_depth=5, num_leaves=31,
                subsample=0.80, subsample_freq=1, colsample_bytree=0.80,
                reg_lambda=8.0, min_child_samples=40, scale_pos_weight=4.0,
                random_state=42, n_jobs=-1, verbose=-1)
        raise ValueError(f"Unknown tabular model: {name}")

    @classmethod
    def train(cls, name: str, X_tr, y_tr, X_val=None, y_val=None,
              use_gpu: bool = False) -> Tuple[Any, float]:
        """تدريب موديل واحد وإرجاع (الموديل، زمن التدريب بالثواني)."""
        import time

        t0 = time.time()
        model = cls.build(name, use_gpu=use_gpu)
        if name == "CatBoost" and X_val is not None and y_val is not None:
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val),
                      early_stopping_rounds=30, verbose=False)
        else:
            model.fit(X_tr, y_tr)
        return model, time.time() - t0

    @classmethod
    def get_candidate_models(cls, use_gpu: bool = False) -> Dict[str, Any]:
        """إرجاع كل النماذج المتاحة مبنية وغير مدربة."""
        return {n: cls.build(n, use_gpu) for n, ok in cls.available().items() if ok}
