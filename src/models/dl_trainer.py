"""
Deep Learning Training Engine with Early Stopping and Validation Threshold Tuning.
"""

import os
import time
import copy
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import matthews_corrcoef, precision_recall_curve, auc, roc_auc_score, confusion_matrix

from src.common.logger import get_logger

logger = get_logger("DL-Trainer")


class DLTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: str = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        clip_norm: float = 1.0
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.clip_norm = clip_norm
        self.best_threshold: float = 0.50
        self.best_model_weights = None

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
            self.optimizer.step()
            total_loss += loss.item() * len(y_batch)
        return total_loss / len(train_loader.dataset)

    def predict_proba(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        y_true_list, y_prob_list = [], []
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                logits = self.model(X_batch)
                probs = torch.softmax(logits, dim=1)[:, 1]
                y_prob_list.extend(probs.cpu().numpy())
                y_true_list.extend(y_batch.numpy())
        return np.array(y_true_list), np.array(y_prob_list)

    def _find_best_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        best_f2 = -1.0
        best_tau = 0.50
        for tau in np.linspace(0.05, 0.95, 91):
            y_pred = (y_prob >= tau).astype(int)
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f2 = (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
            if f2 > best_f2:
                best_f2 = f2
                best_tau = float(tau)
        return round(best_tau, 4)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 15, patience: int = 4):
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    val_loss += self.criterion(self.model(X_batch), y_batch).item() * len(y_batch)
            val_loss /= len(val_loader.dataset)

            logger.info(f"Epoch [{epoch}/{epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_model_weights = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"🛑 تفعيل التوقف المبكر (Early Stopping) عند Epoch {epoch}")
                break

        if self.best_model_weights:
            self.model.load_state_dict(self.best_model_weights)

        # ضبط العتبة المثلى على Validation Set
        y_val_true, y_val_prob = self.predict_proba(val_loader)
        self.best_threshold = self._find_best_threshold(y_val_true, y_val_prob)
        logger.info(f"🎯 العتبة المثلى لـ Deep Learning: [{self.best_threshold}]")

    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        start_time = time.perf_counter()
        y_test_true, y_test_prob = self.predict_proba(test_loader)
        total_time_ms = (time.perf_counter() - start_time) * 1000.0
        latency_per_sample = total_time_ms / max(len(y_test_true), 1)

        y_test_pred = (y_test_prob >= self.best_threshold).astype(int)
        
        cm = confusion_matrix(y_test_true, y_test_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        acc = float((tp + tn) / len(y_test_true)) if len(y_test_true) > 0 else 0.0
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float((2 * prec * rec) / (prec + rec)) if (prec + rec) > 0 else 0.0
        f2 = float((5 * prec * rec) / (4 * prec + rec)) if (4 * prec + rec) > 0 else 0.0
        
        denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        mcc = float(((tp * tn) - (fp * fn)) / denom) if denom > 0 else 0.0

        auc_roc, auc_pr = 0.0, 0.0
        try:
            auc_roc = float(roc_auc_score(y_test_true, y_test_prob))
            p_vals, r_vals, _ = precision_recall_curve(y_test_true, y_test_prob)
            auc_pr = float(auc(r_vals, p_vals))
        except Exception:
            pass

        return {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "f2_score": round(f2, 4),
            "mcc": round(mcc, 4),
            "auc_roc": round(auc_roc, 4),
            "auc_pr": round(auc_pr, 4),
            "false_positive_rate": round(float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0, 4),
            "false_negative_rate": round(float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0, 4),
            "latency_ms_per_sample": round(latency_per_sample, 4),
            "optimal_threshold": self.best_threshold,
            "confusion_matrix": {"true_positives": int(tp), "false_positives": int(fp), "true_negatives": int(tn), "false_negatives": int(fn)}
        }

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "best_threshold": self.best_threshold
        }, path)