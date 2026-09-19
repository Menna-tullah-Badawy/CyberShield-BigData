"""
Temporal Sliding Window Dataset & PyTorch DataLoaders.
استخراج مصفوفات الخصائص من Spark وتحويلها إلى نوافذ زمنية منزلقة (Sliding Windows) بطول 10 حزم.
"""

from typing import Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pyspark.sql import DataFrame
from pyspark.ml.linalg import DenseVector

class NIDSTemporalDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.X = torch.tensor(sequences, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def spark_to_numpy_matrices(df: DataFrame, feature_col: str = "final_features", label_col: str = "label") -> Tuple[np.ndarray, np.ndarray]:
    """تحويل DataFrame من PySpark إلى مصفوفات NumPy مع التطهير من أي أخطاء."""
    rows = df.select(feature_col, label_col).collect()
    features = np.array([
        r[0].toArray() if isinstance(r[0], DenseVector) else np.array(r[0]) 
        for r in rows
    ], dtype=np.float32)
    
    # تنظيف مصفوفة الخصائص من NaN/Inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # تحويل التصنيفات لتمثيل ثنائي رقمي
    labels_raw = [r[1] for r in rows]
    labels = np.array([
        0 if str(lbl).lower() in ["benign", "normal", "0", "0.0"] else 1 
        for lbl in labels_raw
    ], dtype=np.int64)

    return features, labels


def create_sliding_windows(features: np.ndarray, labels: np.ndarray, seq_len: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    تحويل البيانات الجدولية المسطحة إلى نوافذ ثلاثية الأبعاد (N - L + 1, L, D).
    """
    n_samples, n_feats = features.shape
    if n_samples < seq_len:
        raise ValueError(f"عدد السجلات ({n_samples}) أصغر من طول النافذة ({seq_len})")

    num_windows = n_samples - seq_len + 1
    shape = (num_windows, seq_len, n_feats)
    strides = (features.strides[0], features.strides[0], features.strides[1])
    
    X_seq = np.lib.stride_tricks.as_strided(features, shape=shape, strides=strides)
    y_seq = labels[seq_len - 1:]
    return np.ascontiguousarray(X_seq), np.ascontiguousarray(y_seq)


def get_temporal_dataloaders(
    train_np: Tuple[np.ndarray, np.ndarray], 
    val_np: Tuple[np.ndarray, np.ndarray], 
    test_np: Tuple[np.ndarray, np.ndarray], 
    seq_len: int = 10, 
    batch_size: int = 64
) -> Dict[str, Any]:
    """توليد كائنات PyTorch DataLoader لكافة مراحل التدريب والتحقق والاختبار."""
    X_train, y_train = create_sliding_windows(train_np[0], train_np[1], seq_len)
    X_val, y_val = create_sliding_windows(val_np[0], val_np[1], seq_len)
    X_test, y_test = create_sliding_windows(test_np[0], test_np[1], seq_len)

    return {
        "train": DataLoader(NIDSTemporalDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True),
        "val": DataLoader(NIDSTemporalDataset(X_val, y_val), batch_size=batch_size, shuffle=False),
        "test": DataLoader(NIDSTemporalDataset(X_test, y_test), batch_size=batch_size, shuffle=False),
        "input_dim": int(X_train.shape[2]),
        "seq_len": seq_len
    }