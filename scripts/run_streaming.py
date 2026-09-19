"""
Real-time Network Traffic Streaming Runner Script.
سكريبت تشغيل محرك الاستماع الحي لحزم الشبكة (Spark Structured Streaming).
"""

import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline.streaming import SparkStreamingPipeline
import configs.settings as cfg
from src.common.logger import get_logger

logger = get_logger("CyberShield-StreamRunner")


def main():
    logger.info("📡 تشغيل محرك استماع وتحليل تدفقات الحزم الحية في الوقت الحقيقي...")
    stream_engine = SparkStreamingPipeline()
    
    monitoring_dir = os.path.join(cfg.RAW_DATA_DIR, "live_stream_feed")
    os.makedirs(monitoring_dir, exist_ok=True)
    
    stream_df = stream_engine.create_file_stream_source(monitoring_dir)
    windowed_df = stream_engine.apply_windowed_aggregations(stream_df, window_duration="10 seconds", slide_duration="5 seconds")
    
    query = stream_engine.start_console_sink(windowed_df, query_name="ProductionLiveTrafficMonitor")
    logger.info(f"🟢 يستمع النظام الآن لأي حزم جديدة تسقط في: {monitoring_dir}")
    
    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        logger.info("🛑 تم إيقاف استعلام البث الحي بواسطة المستخدم.")
        query.stop()


if __name__ == "__main__":
    main()