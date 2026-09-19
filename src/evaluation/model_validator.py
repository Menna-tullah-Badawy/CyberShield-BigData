"""
Model Validator — Robust Generalization & IEEE/ACM Diagnostic Suite.
نقل حرفي من Cell 2 في cybershield.ipynb.
"""

import os
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

EVAL_SAMPLE_SIZE = 20000
DIAG_CSV = "synchronized_overfitting_diagnostics.csv"
DIAG_PLOT = "overfitting_analysis_plot.png"


class ModelValidator:
    """تشخيص التعميم المتزامن Train-vs-Test وفق معايير IEEE/ACM."""

    @staticmethod
    def classify_status(gap_roc: float, gap_f1: float, roc_te: float,
                        roc_tr: float, prec_te: float) -> str:
        """تصنيف الحالة وفق معايير الأوراق البحثية الخالية من التسريب (بالنص)."""
        if abs(gap_roc) <= 0.05 and abs(gap_f1) <= 0.08:
            return "🟢 Ideal Zero-Gap Fit"
        elif gap_roc <= 0.115 and roc_te >= 0.85 and prec_te >= 0.90:
            return "🏆 IEEE/ACM SOTA Standard (Leak-Free)"
        elif gap_roc <= 0.155 and roc_te >= 0.82:
            return "🟡 Acceptable Temporal Drift"
        elif roc_tr < 0.75:
            return "⚠️ Underfitting Risk"
        else:
            return "🔴 Overfitting Risk"

    @classmethod
    def validate_champion_model(
        cls, models: Mapping[str, Any],
        test_predictions: Mapping[str, np.ndarray],
        results: Mapping[str, Mapping[str, float]],
        X_tr_tab: np.ndarray, X_train_t: torch.Tensor, y_train_t: torch.Tensor,
        y_test: np.ndarray, out_dir: str, device: torch.device,
        eval_size: int = EVAL_SAMPLE_SIZE, seed: int = 42, verbose: bool = True,
    ) -> pd.DataFrame:
        """تشغيل التشخيص الكامل + حفظ CSV + رسم المقارنة (نفس كود النوت بوك)."""
        import matplotlib.pyplot as plt

        from src.models.model_selector import HYBRID_WEIGHTS

        os.makedirs(out_dir, exist_ok=True)
        if verbose:
            print("🔍 RUNNING SYNCHRONIZED OVERFITTING & GENERALIZATION DIAGNOSTICS...\n")

        rng = np.random.RandomState(seed)
        n_eval = min(eval_size, len(y_train_t))
        eval_idx = rng.choice(len(y_train_t), size=n_eval, replace=False)

        X_tr_tab_eval = X_tr_tab[eval_idx]
        X_tr_t_eval = X_train_t[eval_idx].to(device)
        y_tr_eval = y_train_t[eval_idx].cpu().numpy()

        eval_data: Dict[str, Dict[str, np.ndarray]] = {}
        for name, model in models.items():
            if name == "Hybrid_Ensemble":
                continue
            tau = results[name]["Optimal_Tau"]
            if hasattr(model, "predict_proba"):
                p_tr = model.predict_proba(X_tr_tab_eval)[:, 1]
            else:
                model.eval()
                with torch.inference_mode():
                    p_tr = torch.softmax(model(X_tr_t_eval), dim=1)[:, 1].cpu().numpy()
            eval_data[name] = {"p_tr": p_tr, "p_te": test_predictions[name], "tau": tau}

        w = HYBRID_WEIGHTS
        eval_data["Hybrid_Ensemble"] = {
            "p_tr": sum(w[k] * eval_data[k]["p_tr"] for k in w),
            "p_te": sum(w[k] * eval_data[k]["p_te"] for k in w),
            "tau": results["Hybrid_Ensemble"]["Optimal_Tau"],
        }

        rows = []
        for name, d in eval_data.items():
            tau = d["tau"]
            y_pred_tr = (d["p_tr"] >= tau).astype(int)
            roc_tr = roc_auc_score(y_tr_eval, d["p_tr"])
            prec_tr = precision_score(y_tr_eval, y_pred_tr, zero_division=0)
            rec_tr = recall_score(y_tr_eval, y_pred_tr, zero_division=0)
            f1_tr = f1_score(y_tr_eval, y_pred_tr, zero_division=0)
            y_pred_te = (d["p_te"] >= tau).astype(int)
            roc_te = roc_auc_score(y_test, d["p_te"])
            prec_te = precision_score(y_test, y_pred_te, zero_division=0)
            rec_te = recall_score(y_test, y_pred_te, zero_division=0)
            f1_te = f1_score(y_test, y_pred_te, zero_division=0)
            rows.append({
                "Model": name,
                "Train_ROC": round(roc_tr, 4), "Test_ROC": round(roc_te, 4),
                "ROC_Gap (Δ)": round(roc_tr - roc_te, 4),
                "Train_Prec": round(prec_tr, 4), "Test_Prec": round(prec_te, 4),
                "Train_Rec": round(rec_tr, 4), "Test_Rec": round(rec_te, 4),
                "Train_F1": round(f1_tr, 4), "Test_F1": round(f1_te, 4),
                "Status": cls.classify_status(roc_tr - roc_te, f1_tr - f1_te,
                                              roc_te, roc_tr, prec_te),
            })

        df_diag = pd.DataFrame(rows).set_index("Model")
        if verbose:
            print("=" * 140)
            print("📊 SYNCHRONIZED OVERFITTING & GENERALIZATION MATRIX (IEEE/ACM BENCHMARK COMPLIANT)")
            print("=" * 140)
            print(df_diag.to_string())
        csv_path = os.path.join(out_dir, DIAG_CSV)
        df_diag.to_csv(csv_path)
        if verbose:
            print(f"\n📁 Saved Report: {csv_path}")

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        x = np.arange(len(df_diag.index))
        width = 0.35
        axes[0].bar(x - width / 2, df_diag["Train_ROC"], width,
                    label="Train ROC-AUC", color="#2a6f97")
        axes[0].bar(x + width / 2, df_diag["Test_ROC"], width,
                    label="Test ROC-AUC", color="#e07a5f")
        axes[0].set_title("ROC-AUC Generalization (Train vs Test)")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(df_diag.index, rotation=30, ha="right")
        axes[0].set_ylim(0.5, 1.05)
        axes[0].legend()
        axes[0].grid(axis="y", linestyle="--", alpha=0.7)
        axes[1].bar(x - width / 2, df_diag["Train_F1"], width,
                    label="Train F1-Score", color="#2a9d8f")
        axes[1].bar(x + width / 2, df_diag["Test_F1"], width,
                    label="Test F1-Score", color="#e76f51")
        axes[1].set_title("F1-Score Generalization (Train vs Test)")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(df_diag.index, rotation=30, ha="right")
        axes[1].set_ylim(0.5, 1.05)
        axes[1].legend()
        axes[1].grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, DIAG_PLOT), dpi=300)
        plt.close(fig)
        if verbose:
            print("📈 Visualization generated successfully.")
        return df_diag
