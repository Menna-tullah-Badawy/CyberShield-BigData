"""
DL Trainer — exact Cell 1 loop: 8 epochs, AdamW(1e-3, wd 2e-3),
CosineAnnealing(T=8), class weights [1, 4], grad clip 1.0 + latency measure.
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


class DLTrainer:
    """مدرب نماذج التعلم العميق (حلقة النوت بوك بالنص)."""

    def __init__(self, device: torch.device, epochs: int = EPOCHS):
        self.device = device
        self.epochs = epochs

    @staticmethod
    def build_dl_dict(num_features: int, device: torch.device) -> Dict[str, nn.Module]:
        """قاموس موديلات الـ DL الأربعة (نفس تعريفات النوت بوك بالنص)."""
        return {
            "CNN_BiLSTM": CNNBiLSTMModel(input_dim=num_features, dropout=0.25).to(device),
            "BiLSTM": BiLSTMModel(input_dim=num_features, hidden_dim=64,
                                  num_layers=2, num_classes=2, dropout=0.25).to(device),
            "BiGRU": BiGRUModel(input_dim=num_features, hidden_dim=64,
                                num_layers=2, num_classes=2, dropout=0.25).to(device),
            "MAMBA_SSM": MambaNIDS(input_dim=num_features, d_model=64, d_state=16,
                                   num_layers=2, num_classes=2, dropout=0.20).to(device),
        }

    def fit(self, model: nn.Module, train_loader: DataLoader) -> float:
        """تدريب كامل وإرجاع زمن التدريب (نفس الحلقة بالنص)."""
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, POS_WEIGHT], dtype=torch.float32, device=self.device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        t0 = time.time()
        for _ in range(1, self.epochs + 1):
            model.train()
            for X_b, y_b in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(X_b), y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            scheduler.step()
        return time.time() - t0

    @torch.inference_mode()
    def predict_proba(self, model: nn.Module, loader: DataLoader) -> np.ndarray:
        """احتمالات الفئة Attack على Loader كامل."""
        model.eval()
        return torch.cat([torch.softmax(model(X_b), dim=1)[:, 1]
                          for X_b, _ in loader]).cpu().numpy()

    def fit_and_evaluate(
        self, model: nn.Module, train_loader: DataLoader,
        val_loader: DataLoader, test_loader: DataLoader, y_test_len: int,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """تدريب + احتمالات Val/Test + زمن التدريب + الـ Latency (نفس كود النوت بوك)."""
        t_train = self.fit(model, train_loader)
        model.eval()
        with torch.inference_mode():
            vp = self.predict_proba(model, val_loader)
            t_start = time.perf_counter()
            tp = self.predict_proba(model, test_loader)
            lat = ((time.perf_counter() - t_start) * 1000.0) / y_test_len
        return vp, tp, t_train, lat
