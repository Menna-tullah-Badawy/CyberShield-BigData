"""
Gradient-based token/feature saliency (XAI).
نقل حرفي من Cell 4 في cybershield.ipynb (دالة extract_xai).
"""

from typing import Dict, List, Sequence

import torch
import torch.nn as nn


def extract_xai(
    model: nn.Module,
    x: torch.Tensor,
    names: Sequence[str],
    n: int = 5,
) -> List[Dict[str, object]]:
    """أهم n ميزة عبر Gradient Saliency + تطبيع (نفس كود النوت بوك)."""
    model.eval()
    x.requires_grad_(True)
    logits = model(x)
    p = torch.softmax(logits, 1)[:, 1]
    model.zero_grad()
    p.backward()
    g = x.grad[0].abs().mean(0).cpu().numpy()
    idx = __import__("numpy").argsort(g)[::-1][:n]
    f = [{"name": names[i] if i < len(names) else f"F{i}",
          "importance": float(g[i])} for i in idx]
    t = sum(d["importance"] for d in f)
    if t > 0:
        for d in f:
            d["importance"] = d["importance"] / t
    x.requires_grad_(False)
    return f
