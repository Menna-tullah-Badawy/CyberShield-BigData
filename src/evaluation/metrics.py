"""
Benchmark Metrics Matrix builder.
منطق Cell 1: تجميع نتائج النماذج في مصفوفة + حفظ CSV + إعلان البطل.
"""

import os
from typing import Dict, Mapping

import pandas as pd

BENCHMARK_CSV = "production_balanced_benchmark.csv"


class CyberEvaluationMetrics:
    """بناء مصفوفة البنش مارك النهائية وحفظها."""

    @staticmethod
    def build_benchmark_matrix(results: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
        """تحويل قاموس النتائج لمصفوفة DataFrame (نفس سطر النوت بوك)."""
        return pd.DataFrame(results).T

    @classmethod
    def save_and_report(cls, results: Dict[str, Dict[str, float]], out_dir: str,
                        verbose: bool = True) -> pd.DataFrame:
        """طباعة المصفوفة + حفظ CSV + إعلان البطل (نفس سطور النوت بوك)."""
        from src.models.model_selector import ModelSelector

        df = cls.build_benchmark_matrix(results)
        if verbose:
            print("\n" + "=" * 115)
            print("🏆 SOTA CYBERSHIELD BENCHMARK MATRIX (PRECISION >= 92% & RECALL >= 90%)")
            print("=" * 115)
            print(df.to_string())
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, BENCHMARK_CSV)
        df.to_csv(csv_path)
        if verbose:
            print(f"\n📁 Saved: {csv_path}")
            best = ModelSelector.select_champion_by_f1(df)
            print(f"\n🌟 Top Balanced Model: {best} "
                  f"(Recall: {df.loc[best, 'Recall'] * 100:.2f}%, "
                  f"Precision: {df.loc[best, 'Precision'] * 100:.2f}%)")
        return df
