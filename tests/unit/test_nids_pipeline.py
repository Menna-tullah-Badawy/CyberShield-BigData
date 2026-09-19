"""
Unit tests for the redistributed notebook track (existing architecture).
اختبارات سريعة ببيانات صناعية — أجزاء torch تُتخطى تلقائياً إن لم تتوفر.
"""

import numpy as np
import pandas as pd
import pytest


def test_create_day_sequences_shape_and_vote():
    from src.feature_engineering.aggregations import FeatureAggregator

    X = np.random.rand(25, 4).astype(np.float32)
    y = np.array([1] * 10 + [0] * 15)
    Xs, ys = FeatureAggregator.create_day_sequences(X, y, seq_len=10)
    assert Xs.shape == (2, 10, 4)
    assert list(ys) == [1, 0]


def test_strict_exclude_filters_identity(tmp_path):
    from src.feature_engineering.pipeline_builder import FeaturePipelineBuilder

    df = pd.DataFrame({
        "Timestamp": pd.date_range("2018-02-14", periods=200, freq="min").astype(str),
        "Flow Duration": np.random.rand(200) * 10,
        "Src IP": ["10.0.0.1"] * 200,
        "Dst Port": np.random.randint(1, 5000, 200),
        "Protocol": [6] * 200,
        "Label": ["Benign"] * 150 + ["DoS"] * 50,
    })
    p = tmp_path / "day.csv"
    df.to_csv(p, index=False)
    builder = FeaturePipelineBuilder(store_dir=str(tmp_path / "store"))
    (x_tr, y_tr), (x_va, y_va), (x_te, y_te), cols, atk = builder.process_day(str(p))
    low = [c.lower() for c in cols]
    assert "src ip" not in low and "dst port" not in low and "protocol" not in low
    assert x_tr.shape[1] == 10 and atk == 10


def test_threshold_optimizer_keys_and_bounds():
    from src.evaluation.threshold_optimizer import optimize_security_threshold

    rng = np.random.RandomState(0)
    y = (rng.rand(1000) > 0.7).astype(int)
    p = np.clip(y * 0.7 + rng.rand(1000) * 0.4, 0, 1)
    m = optimize_security_threshold(y, p, y, p)
    assert set(m) == {"Accuracy", "Precision", "Recall", "F1-Score", "F2-Score",
                      "MCC", "PR-AUC", "ROC-AUC", "FPR", "FNR", "Optimal_Tau"}
    assert 0.01 <= m["Optimal_Tau"] <= 0.95


def test_hybrid_ensemble_math():
    from src.models.model_selector import HYBRID_WEIGHTS, ModelSelector

    assert abs(sum(HYBRID_WEIGHTS.values()) - 1.0) < 1e-9
    preds = {k: np.full(10, 0.5) for k in HYBRID_WEIGHTS}
    out = ModelSelector.hybrid_ensemble_predict(preds)
    assert out.shape == (10,) and np.allclose(out, 0.5)


def test_bm25_lsa_retrieval_top_hit():
    from src.genai_reporting.mitre_mapping import THREAT_KB, HybridThreatRetriever

    assert len(THREAT_KB) == 6
    r = HybridThreatRetriever()
    hit = r.retrieve("syn flood ddos denial service packet")
    assert hit["doc"]["threat_id"] == "MITRE-T1498"
    assert 0.0 <= hit["hybrid_score"] <= 1.0


def test_generalization_status_rules():
    from src.evaluation.model_validator import ModelValidator

    assert "Ideal" in ModelValidator.classify_status(0.02, 0.03, 0.90, 0.92, 0.95)
    assert "SOTA" in ModelValidator.classify_status(0.10, 0.20, 0.87, 0.97, 0.91)
    assert "Drift" in ModelValidator.classify_status(0.14, 0.20, 0.83, 0.97, 0.80)
    assert "Overfitting" in ModelValidator.classify_status(0.30, 0.30, 0.60, 0.90, 0.50)


def test_balance_and_scale():
    from src.feature_engineering.scaling import FeatureScaler
    from src.feature_store.data_splitter import DataSplitter

    rng = np.random.RandomState(3)
    X = rng.rand(500, 10, 6).astype(np.float32)
    y = (rng.rand(500) > 0.85).astype(np.int64)
    Xb, yb = DataSplitter.balance_train_1_to_4(X, y)
    pos, neg = int(yb.sum()), int((yb == 0).sum())
    assert neg == min((y == 0).sum(), pos * 4)
    X3, _, _ = FeatureScaler().scale_all(Xb, X[:40], X[40:80])
    assert X3.shape[1:] == (10, 6) and float(np.abs(X3).max()) <= 6.0 + 1e-6


def test_feature_store_roundtrip(tmp_path):
    from src.feature_store.transformations import FeatureStoreManager

    store = FeatureStoreManager(str(tmp_path / "store"))
    arrays = {n: np.random.rand(8, 10, 4).astype(np.float32) if n.startswith("X_")
              else np.random.randint(0, 2, 8) for n in
              ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test")}
    meta = store.save_sequences(arrays, ["a", "b", "c", "d"], 10)
    assert meta["final_feature_dimension"] == 4
    assert store.exists()
    loaded = store.load_sequences()
    assert len(loaded) == 6 and loaded[0].shape == (8, 10, 4)


torch = pytest.importorskip("torch", reason="torch not installed")


def test_dl_models_forward_shapes():
    from src.models.deep_learning_models import (BiGRUModel, BiLSTMModel,
                                                 CNNBiLSTMModel, MambaNIDS,
                                                 SequenceClassifier)

    x = torch.randn(2, 10, 8)
    assert tuple(CNNBiLSTMModel(input_dim=8)(x).shape) == (2, 2)
    assert tuple(BiLSTMModel(input_dim=8)(x).shape) == (2, 2)
    assert tuple(BiGRUModel(input_dim=8)(x).shape) == (2, 2)
    assert tuple(MambaNIDS(input_dim=8, d_model=8, d_state=4, num_layers=1)(x).shape) == (2, 2)
    assert tuple(SequenceClassifier("cnn_bilstm", input_dim=8)(x).shape) == (2, 2)


def test_xai_importance_sums_to_one():
    from src.explainability.feature_importance import GlobalFeatureImportance
    from src.models.deep_learning_models import MambaNIDS

    m = MambaNIDS(input_dim=4, d_model=8, d_state=4, num_layers=1)
    x = torch.randn(1, 10, 4)
    feats = GlobalFeatureImportance(["a", "b", "c", "d"]).extract_importance(m, x, n=3)
    assert len(feats) == 3
    assert abs(sum(f["importance"] for f in feats) - 1.0) < 1e-6
