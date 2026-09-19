"""
Comprehensive Comparative Benchmark Execution Script.
التشغيل الشامل للمقارنة المرجعية (Distributed Spark RF vs Bi-LSTM vs Bi-GRU vs Mamba SSM).
"""

import os
import json
import pandas as pd
from pyspark.sql import SparkSession

from src.common.logger import get_logger
from src.models.model_selector import SparkModelSelector
from src.evaluation.threshold_optimizer import ThresholdOptimizer
from src.evaluation.metrics import CyberEvaluationMetrics
from src.models.sequence_dataset import spark_to_numpy_matrices, get_temporal_dataloaders
from src.models.deep_learning_models import SequenceClassifier
from src.models.dl_trainer import DLTrainer

logger = get_logger("NIDS-Benchmark")


def main():
    spark = SparkSession.builder \
        .appName("CyberShield-BigData-Benchmark") \
        .master("local[*]") \
        .config("spark.driver.memory", "8g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

    PROCESSED_DATA_PATH = "data/processed/"
    REPORTS_DIR = "reports/"
    CHECKPOINTS_DIR = "checkpoints/"
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    logger.info("📥 تحميل مجموعات البيانات الموزعة من Parquet...")
    train_df = spark.read.parquet(os.path.join(PROCESSED_DATA_PATH, "train.parquet"))
    val_df = spark.read.parquet(os.path.join(PROCESSED_DATA_PATH, "val.parquet"))
    test_df = spark.read.parquet(os.path.join(PROCESSED_DATA_PATH, "test.parquet"))

    benchmark_results = {}

    # 1. تدريب وتقييم Spark Random Forest Champion
    logger.info("🌲 [Benchmark 1/4] تدريب وتقييم Spark Random Forest...")
    rf_selector = SparkModelSelector(num_trees=100, max_depth=16)
    rf_model, _ = rf_selector.train_and_select_best(
        df=train_df,
        features_col="final_features",
        label_col="label",
        val_df=val_df,
        test_df=test_df
    )

    # ضبط العتبة وتقييم الاختبار
    val_preds = rf_model.transform(val_df)
    optimizer = ThresholdOptimizer(probability_col="probability", label_col="label")
    opt_res = optimizer.find_optimal_threshold(val_preds, metric_target="f2")
    optimal_tau = opt_res.get("optimal_threshold", 0.50)

    test_preds = optimizer.apply_custom_threshold(rf_model.transform(test_df), threshold=optimal_tau)
    evaluator = CyberEvaluationMetrics(label_col="label", prediction_col="prediction", probability_col="probability")
    rf_final_metrics = evaluator.compute_all_metrics(test_preds)
    rf_final_metrics["optimal_threshold"] = optimal_tau
    benchmark_results["Spark_Random_Forest"] = rf_final_metrics

    # 2. تجهيز بيانات التسلسل الزمني للتعلم العميق
    logger.info("⏳ تجهيز النوافذ الزمنية المنزلقة لنماذج الـ Deep Learning...")
    train_np = spark_to_numpy_matrices(train_df, feature_col="final_features", label_col="label")
    val_np = spark_to_numpy_matrices(val_df, feature_col="final_features", label_col="label")
    test_np = spark_to_numpy_matrices(test_df, feature_col="final_features", label_col="label")

    dl_data = get_temporal_dataloaders(train_np, val_np, test_np, seq_len=10, batch_size=64)

    dl_architectures = [
        ("Bi-LSTM", "bilstm", {"hidden_dim": 64, "num_layers": 2, "dropout": 0.2}),
        ("Bi-GRU", "bigru", {"hidden_dim": 64, "num_layers": 2, "dropout": 0.2}),
        ("Mamba_SSM", "mamba", {"d_model": 64, "d_state": 16, "num_layers": 2, "dropout": 0.2})
    ]

    # 3. تدريب وتقييم نماذج Deep Learning
    for idx, (name, arch_type, params) in enumerate(dl_architectures, start=2):
        logger.info(f"🧠 [Benchmark {idx}/4] تدريب نموذج {name}...")
        model = SequenceClassifier(arch_type=arch_type, input_dim=dl_data["input_dim"], **params)
        trainer = DLTrainer(model=model, lr=1e-3)
        trainer.fit(dl_data["train"], dl_data["val"], epochs=15, patience=4)
        dl_metrics = trainer.evaluate(dl_data["test"])
        benchmark_results[name] = dl_metrics
        trainer.save_checkpoint(os.path.join(CHECKPOINTS_DIR, f"{arch_type}_checkpoint.pt"))

    # 4. توليد المصفوفة المقارنة الموحدة
    summary_df = pd.DataFrame(benchmark_results).T
    cols_order = [
        "accuracy", "precision", "recall", "f1_score", "f2_score", 
        "mcc", "auc_pr", "auc_roc", "false_positive_rate", "false_negative_rate", "optimal_threshold"
    ]
    summary_df = summary_df[[c for c in cols_order if c in summary_df.columns]]

    print("\n" + "=" * 90)
    print("🏆 FINAL COMPARATIVE BENCHMARK MATRIX (CYBERSHIELD-BIGDATA)")
    print("=" * 90)
    print(summary_df.to_string())

    summary_df.to_csv(os.path.join(REPORTS_DIR, "benchmark_summary.csv"))
    with open(os.path.join(REPORTS_DIR, "benchmark_summary.json"), "w") as f:
        json.dump(benchmark_results, f, indent=4)

    logger.info(f"🎉 تم حفظ مخرجات المقارنة بنجاح في مجلد: {REPORTS_DIR}")
    spark.stop()


if __name__ == "__main__":
    main()