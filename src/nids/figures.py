"""
Publication-Quality Visualization & Confusion Matrix Suite.
نقل حرفي من Cell 3 في cybershield.ipynb (3 أشكال بدقة 300 DPI).
"""

import os
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc

TOP_MODELS = ["MAMBA_SSM", "Hybrid_Ensemble", "XGBoost", "CNN_BiLSTM"]
CURVE_COLORS = {
    "MAMBA_SSM": "#1f77b4", "Hybrid_Ensemble": "#d62728",
    "XGBoost": "#2ca02c", "LightGBM": "#9467bd",
    "CatBoost": "#8c564b", "CNN_BiLSTM": "#ff7f0e",
    "BiLSTM": "#e377c2", "BiGRU": "#7f7f7f",
}
FIGURE_FILES = ("figure1_confusion_matrices.png",
                "figure2_roc_pr_curves.png",
                "figure3_benchmark_barchart.png")


def apply_pub_style() -> None:
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-v0_8-whitegrid'
                  if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 15,
        'figure.autolayout': True,
    })


def plot_confusion_matrices(
    test_predictions: Mapping[str, np.ndarray],
    results: Mapping[str, Mapping[str, float]],
    y_test: np.ndarray,
    out_dir: str,
) -> str:
    import matplotlib.pyplot as plt
    import seaborn as sns

    top_models = [m for m in TOP_MODELS if m in test_predictions]
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.ravel()

    for idx, name in enumerate(top_models):
        tau = results[name]["Optimal_Tau"]
        y_pred = (test_predictions[name] >= tau).astype(int)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        annot = np.array([
            [f"{cm[0, 0]:,}\n({cm_perc[0, 0]:.1f}%)", f"{cm[0, 1]:,}\n({cm_perc[0, 1]:.1f}%)"],
            [f"{cm[1, 0]:,}\n({cm_perc[1, 0]:.1f}%)", f"{cm[1, 1]:,}\n({cm_perc[1, 1]:.1f}%)"],
        ])
        sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False, ax=axes[idx],
                    annot_kws={"size": 11, "weight": "bold"},
                    xticklabels=["Pred Benign", "Pred Attack"],
                    yticklabels=["True Benign", "True Attack"])
        rec = results[name]['Recall'] * 100
        prec = results[name]['Precision'] * 100
        axes[idx].set_title(f"{name}\n(Recall: {rec:.1f}% | Precision: {prec:.1f}%)", weight="bold")
        axes[idx].set_ylabel("Actual Traffic")
        axes[idx].set_xlabel("Predicted Traffic")

    plt.suptitle("Multi-Model Confusion Matrices (True Counts & Percentages)",
                 fontsize=16, weight="bold")
    cm_path = os.path.join(out_dir, FIGURE_FILES[0])
    plt.savefig(cm_path, dpi=300)
    plt.close(fig)
    return cm_path


def plot_roc_pr_curves(
    test_predictions: Mapping[str, np.ndarray], y_test: np.ndarray, out_dir: str
) -> str:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for name, p_te in test_predictions.items():
        fpr, tpr, _ = roc_curve(y_test, p_te)
        roc_score = auc(fpr, tpr)
        lw = 2.5 if name in ["MAMBA_SSM", "Hybrid_Ensemble"] else 1.5
        axes[0].plot(fpr, tpr, label=f"{name} (AUC = {roc_score:.4f})",
                     color=CURVE_COLORS.get(name), linewidth=lw)
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Random Guess (0.50)")
    axes[0].set_title("Receiver Operating Characteristic (ROC) Curves", weight="bold")
    axes[0].set_xlabel("False Positive Rate (FPR)")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].set_xlim([-0.01, 1.0])
    axes[0].set_ylim([0.0, 1.02])
    axes[0].legend(loc="lower right")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    for name, p_te in test_predictions.items():
        prec, rec, _ = precision_recall_curve(y_test, p_te)
        pr_score = auc(rec, prec)
        lw = 2.5 if name in ["MAMBA_SSM", "Hybrid_Ensemble"] else 1.5
        axes[1].plot(rec, prec, label=f"{name} (PR-AUC = {pr_score:.4f})",
                     color=CURVE_COLORS.get(name), linewidth=lw)
    axes[1].set_title("Precision-Recall (PR) Curves (Imbalance Metric)", weight="bold")
    axes[1].set_xlabel("Recall (Detection Rate)")
    axes[1].set_ylabel("Precision (Detection Purity)")
    axes[1].set_xlim([0.0, 1.02])
    axes[1].set_ylim([0.65, 1.02])
    axes[1].legend(loc="lower left")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    curves_path = os.path.join(out_dir, FIGURE_FILES[1])
    plt.savefig(curves_path, dpi=300)
    plt.close(fig)
    return curves_path


def plot_benchmark_barchart(
    df: pd.DataFrame, test_predictions: Mapping[str, np.ndarray], out_dir: str
) -> str:
    import matplotlib.pyplot as plt

    metrics_to_plot = ["Precision", "Recall", "F1-Score", "ROC-AUC"]
    df_plot = df.loc[list(test_predictions.keys()), metrics_to_plot] * 100

    plt.figure(figsize=(14, 6))
    x = np.arange(len(df_plot.index))
    width = 0.20
    metric_colors = ["#2b5c8f", "#e26d5c", "#38b000", "#ffaa00"]

    for i, col in enumerate(metrics_to_plot):
        plt.bar(x + (i - 1.5) * width, df_plot[col], width, label=col,
                color=metric_colors[i], edgecolor="black", linewidth=0.5)

    plt.title("CyberShield SOTA Benchmark - Multi-Model Metric Comparison (%)",
              fontsize=14, weight="bold")
    plt.xticks(x, df_plot.index, rotation=25, ha="right", weight="bold")
    plt.ylabel("Score (%)", weight="bold")
    plt.ylim([65, 105])
    plt.legend(loc="upper right", frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    bench_path = os.path.join(out_dir, FIGURE_FILES[2])
    plt.savefig(bench_path, dpi=300)
    plt.close()
    return bench_path


def generate_all_figures(
    test_predictions: Mapping[str, np.ndarray],
    results: Mapping[str, Mapping[str, float]],
    y_test: np.ndarray,
    df: pd.DataFrame,
    out_dir: str,
    verbose: bool = True,
) -> Mapping[str, str]:
    """توليد الأشكال الثلاثة كلها (نفس ترتيب النوت بوك)."""
    os.makedirs(out_dir, exist_ok=True)
    apply_pub_style()
    if verbose:
        print("🎨 GENERATING SOTA BENCHMARK VISUALIZATIONS...\n")

    cm_path = plot_confusion_matrices(test_predictions, results, y_test, out_dir)
    if verbose:
        print(f"✅ Saved Confusion Matrices: {cm_path}")
    curves_path = plot_roc_pr_curves(test_predictions, y_test, out_dir)
    if verbose:
        print(f"✅ Saved ROC & PR Curves: {curves_path}")
    bench_path = plot_benchmark_barchart(df, test_predictions, out_dir)
    if verbose:
        print(f"✅ Saved Comparative Bar Chart: {bench_path}")
        print("\n🎉 ALL VISUALIZATION FIGURES SUCCESSFULLY GENERATED & SAVED IN 300 DPI!")
    return {"confusion": cm_path, "curves": curves_path, "barchart": bench_path}
