"""
Gradient Saliency Feature Importance.
نقل حرفي من Cell 4 في cybershield.ipynb (دالة extract_xai).
"""

from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn


class GlobalFeatureImportance:
    """أهمية الخصائص عبر Gradient Saliency (نفس كود النوت بوك)."""

    def __init__(self, feature_names: Sequence[str] = None):
        self.feature_names = list(feature_names) if feature_names else []

    def extract_importance(
        self, model: nn.Module, x: torch.Tensor,
        names: Sequence[str] = None, n: int = 5,
    ) -> List[Dict[str, object]]:
        names = list(names) if names else self.feature_names
        model.eval()
        x.requires_grad_(True)
        logits = model(x)
        p = torch.softmax(logits, 1)[:, 1]
        model.zero_grad()
        p.backward()
        g = x.grad[0].abs().mean(0).cpu().numpy()
        idx = np.argsort(g)[::-1][:n]
        f = [{"name": names[i] if i < len(names) else f"F{i}",
              "importance": float(g[i])} for i in idx]
        t = sum(d["importance"] for d in f)
        if t > 0:
            for d in f:
                d["importance"] = d["importance"] / t
        x.requires_grad_(False)
        return f
