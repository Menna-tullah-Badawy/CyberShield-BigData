"""
Deep Learning training loop (8 epochs, AdamW, CosineAnnealing, class weights).
نقل حرفي من Cell 1 في cybershield.ipynb.
"""

import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.deep_learning_models import (BiGRUModel, BiLSTMModel,
                                             CNNBiLSTMModel, MambaNIDS)

EPOCHS = 8
LR = 1e-3
WEIGHT_DECAY = 2e-3
GRAD_CLIP = 1.0
POS_WEIGHT = 4.0
HIDDEN_DIM = 64
MAMBA_D_MODEL = 64
MAMBA_D_STATE = 16
DL_DROPOUT = 0.25
MAMBA_DROPOUT = 0.20


def build_dl_dict(num_features: int, device: torch.device) -> Dict[str, nn.Module]:
    """قاموس موديلات الـ DL الأربعة (نفس تعريفات النوت بوك بالنص)."""
    return {
        "CNN_BiLSTM": CNNBiLSTMModel(input_dim=num_features, dropout=DL_DROPOUT).to(device),
        "BiLSTM": BiLSTMModel(input_dim=num_features, hidden_dim=HIDDEN_DIM,
                              num_layers=2, num_classes=2, dropout=DL_DROPOUT).to(device),
        "BiGRU": BiGRUModel(input_dim=num_features, hidden_dim=HIDDEN_DIM,
                            num_layers=2, num_classes=2, dropout=DL_DROPOUT).to(device),
        "MAMBA_SSM": MambaNIDS(input_dim=num_features, d_model=MAMBA_D_MODEL,
                               d_state=MAMBA_D_STATE, num_layers=2,
                               num_classes=2, dropout=MAMBA_DROPOUT).to(device),
    }


def train_dl_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    y_test_len: int,
    epochs: int = EPOCHS,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    تدريب موديل DL واحد وإرجاع:
    (val_probs, test_probs, train_time_sec, latency_ms_per_sample)
    """
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, POS_WEIGHT], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    t0 = time.time()
    for _ in range(1, epochs + 1):
        model.train()
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        scheduler.step()
    t_train = time.time() - t0

    model.eval()
    with torch.inference_mode():
        vp = torch.cat([torch.softmax(model(X_b), dim=1)[:, 1]
                        for X_b, _ in val_loader]).cpu().numpy()
        t_start = time.perf_counter()
        tp = torch.cat([torch.softmax(model(X_b), dim=1)[:, 1]
                        for X_b, _ in test_loader]).cpu().numpy()
        lat = ((time.perf_counter() - t_start) * 1000.0) / y_test_len

    return vp, tp, t_train, lat
