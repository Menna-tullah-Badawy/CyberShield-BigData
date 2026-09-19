"""
==============================================================================
CyberShield-BigData: Hybrid NIDS Orchestrator (torch track)
نفس مراحل cybershield.ipynb عبر CLI:
    extract → benchmark → diagnostics → figures → serve / report / bot
==============================================================================
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import configs.settings as cfg
from src.common.logger import get_logger

logger = get_logger("CyberShield-Master")


def print_banner():
    banner = """
    =========================================================================
      ______      __                _____ _     _      _     _
     / ____/_  __/ /_  ___  _____  / ___/| |   (_)__  | |   / /
    / /   / / / / __ \\/ _ \\/ ___/  \\__ \\ | |  / / _ \\ | |  / /
   / /___/ /_/ / /_/ /  __/ /      ___/ / | | / /  __/ | | / /
   \\____/\\__, /_.___/\\___/_/      /____/  |_|/_/\\___/  |___/
        /____/  Hybrid NIDS — Mamba SSM + XAI + MITRE RAG
    =========================================================================
    """
    print(banner)


def run_extract(args) -> dict:
    from src.feature_engineering.pipeline_builder import FeaturePipelineBuilder
    from src.feature_store.transformations import FeatureStoreManager

    store = FeatureStoreManager(args.store_dir)
    if store.exists() and not args.force:
        logger.info("✅ Feature Store already built — skipping (use --force to rebuild)")
        return store.load_metadata()
    builder = FeaturePipelineBuilder(store_dir=args.store_dir)
    return builder.build_features(csv_dir=args.data_path)


def run_benchmark(args) -> dict:
    from src.orchestrator.pipeline_orchestrator import EndToEndPipelineOrchestrator

    orch = EndToEndPipelineOrchestrator(store_dir=args.store_dir, out_dir=args.out_dir)
    return orch.run_training_pipeline()


def run_diagnostics(args, bench: dict):
    from src.evaluation.model_validator import ModelValidator

    return ModelValidator.validate_champion_model(
        models=bench["models"], test_predictions=bench["test_predictions"],
        results=bench["results"], X_tr_tab=bench["X_tr_tab"],
        X_train_t=bench["X_train_t"], y_train_t=bench["y_train_t"],
        y_test=bench["y_test"], out_dir=args.out_dir, device=bench["device"])


def run_figures(args, bench: dict):
    from src.evaluation.figures import generate_all_figures

    return generate_all_figures(
        test_predictions=bench["test_predictions"], results=bench["results"],
        y_test=bench["y_test"], df=bench["df"], out_dir=args.out_dir)


def run_serve(args):
    import uvicorn

    from serving.api.main import create_app

    app = create_app()
    logger.info(f"🚀 Serving on 0.0.0.0:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


def run_report(args):
    import numpy as np
    import torch

    from serving.api.main import build_default_ctx
    from src.feature_store.transformations import FeatureStoreManager
    from src.genai_reporting.report_generator import IncidentReportGenerator
    from src.monitoring.monitoring_engine import MLOpsMonitoringEngine

    ctx = build_default_ctx(args.store_dir)
    store = FeatureStoreManager(args.store_dir)
    _, _, _, _, X_test_raw, y_test = store.load_sequences()
    engine = MLOpsMonitoringEngine(
        ctx["mamba"], ctx["scaler"], ctx["feature_names"],
        ctx["retriever"], ctx["device"])
    audit = engine.run_batch_audit(X_test_raw, y_test)
    pdf_path = os.path.join(args.out_dir, "CyberShield_SOC_Report.pdf")
    IncidentReportGenerator().create_soc_report(
        pdf_path=pdf_path, tp=audit["tp"], fp=audit["fp"], tn=audit["tn"],
        fn=audit["fn"], roc=audit["roc"], n_samples=audit["n_samples"], cm=audit["cm"],
        xai_features=audit["xai_features"],
        threat_types_detected=audit["threat_types_detected"],
        threat_kb=ctx["threat_kb"], port=args.port,
        results=audit.get("results"))
    logger.info(f"📄 Report: {pdf_path}")


def run_bot(args):
    import torch

    from serving.api.main import build_default_ctx
    from src.feature_store.transformations import FeatureStoreManager
    from src.monitoring.monitoring_engine import MLOpsMonitoringEngine

    ctx = build_default_ctx(args.store_dir)
    store = FeatureStoreManager(args.store_dir)
    _, _, _, _, X_test_raw, y_test = store.load_sequences()
    engine = MLOpsMonitoringEngine(
        ctx["mamba"], ctx["scaler"], ctx["feature_names"],
        ctx["retriever"], ctx["device"],
        bot_token=os.environ.get("CYBERSHIELD_BOT_TOKEN", ""))
    asyncio.run(engine.run_production_audit(X_test_raw, y_test))


def main():
    parser = argparse.ArgumentParser(description="CyberShield-BigData NIDS Orchestrator")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "extract", "benchmark", "diagnostics",
                                 "figures", "serve", "report", "bot"])
    parser.add_argument("--data-path", type=str, default=None,
                        help="CSV directory (default: download via kagglehub)")
    parser.add_argument("--store-dir", type=str, default=cfg.STORE_DIR)
    parser.add_argument("--out-dir", type=str, default=cfg.OUT_DIR)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--force", action="store_true",
                        help="Rebuild stages even if outputs exist")
    args = parser.parse_args()

    print_banner()
    logger.info(f"⚡ Mode: {args.mode}")

    if args.mode in ("all", "extract"):
        run_extract(args)
        if args.mode == "extract":
            return
    if args.mode in ("all", "benchmark"):
        bench = run_benchmark(args)
        if args.mode == "benchmark":
            return
    else:
        bench = None

    if args.mode == "all":
        run_diagnostics(args, bench)
        run_figures(args, bench)
        logger.info("✅ Full pipeline complete.")
    elif args.mode == "diagnostics":
        raise SystemExit("diagnostics needs a benchmark run: use --mode all first")
    elif args.mode == "figures":
        raise SystemExit("figures needs a benchmark run: use --mode all first")
    elif args.mode == "serve":
        run_serve(args)
    elif args.mode == "report":
        run_report(args)
    elif args.mode == "bot":
        run_bot(args)


if __name__ == "__main__":
    main()
