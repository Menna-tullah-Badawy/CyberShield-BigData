"""
Run the Sweet-Spot Benchmark only (Cell 1 track).
Usage: python scripts/run_benchmark.py [--store-dir D] [--out-dir O]
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.logger import get_logger
from src.orchestrator.pipeline_orchestrator import EndToEndPipelineOrchestrator

logger = get_logger("CyberShield-Benchmark")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store-dir", default="data/feature_store/nids_features_latest")
    parser.add_argument("--out-dir", default="benchmark_results")
    args = parser.parse_args()

    orch = EndToEndPipelineOrchestrator(store_dir=args.store_dir, out_dir=args.out_dir)
    bench = orch.run_training_pipeline()
    logger.info(f"✅ Benchmark complete: {len(bench['results'])} models evaluated.")


if __name__ == "__main__":
    main()
