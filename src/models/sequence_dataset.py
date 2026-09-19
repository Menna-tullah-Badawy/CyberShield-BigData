"""
Temporal Datasets & DataLoaders (torch track).
منطق Cell 1: TensorDataset على الـ GPU + BATCH_SIZE=512.
"""

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

BATCH_SIZE = 512


class NIDSTemporalDataset(Dataset):
    """داتاسيت السلاسل الزمنية (numpy → torch)."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray,
                 device: torch.device = None):
        self.X = torch.tensor(np.asarray(sequences), dtype=torch.float32)
        self.y = torch.tensor(np.asarray(labels), dtype=torch.long)
        if device is not None:
            self.X, self.y = self.X.to(device), self.y.to(device)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def to_device_tensors(
    X_3d: np.ndarray, y: np.ndarray, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """تحويل مصفوفات numpy لتنسورات على الجهاز (نفس سطور النوت بوك)."""
    return (torch.tensor(X_3d, dtype=torch.float32, device=device),
            torch.tensor(y, dtype=torch.long, device=device))


def get_temporal_dataloaders(
    X_train_t: torch.Tensor, y_train_t: torch.Tensor,
    X_val_t: torch.Tensor, y_val_t: torch.Tensor,
    X_test_t: torch.Tensor, y_test_t: torch.Tensor,
    batch_size: int = BATCH_SIZE,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """بناء الـ Loaders الثلاثة (نفس إعدادات النوت بوك بالنص)."""
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t),
                            batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t),
                             batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
