"""
Real-time Network Traffic Stream Viewer & Live Ticker.
شاشة مراقبة تدفق الحزم اللحظي وكشف الهجمات فور وقوعها.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from serving.dashboard.components.charts import create_traffic_gauge


def render():
    st.title("📡 Real-time Network Traffic Surveillance")
    st.markdown("مراقبة مستمرة وتصنيف فوري لتدفق الحزم الشبكية (Spark Streaming Telemetry).")

    # بطاقات المؤشرات اللحظية
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Throughput (PPS)", "1,248 pps", "+84 pps")
    c2.metric("Bandwidth", "14.2 Mbps", "+1.1 Mbps")
    c3.metric("Inspected Micro-batches", "18,490", "Active")
    c4.metric("Avg Latency", "12.4 ms", "-1.2 ms")

    st.markdown("---")

    col_gauge, col_feed = st.columns([1, 2])
    with col_gauge:
        st.subheader("Threat Gauge")
        st.plotly_chart(create_traffic_gauge(14.8), use_container_width=True)

    with col_feed:
        st.subheader("🚨 Live Detected Threats Stream")
        sample_alerts = pd.DataFrame([
            {"Time": "02:14:10", "Src IP": "192.168.1.105", "Target": "10.0.0.1:80", "Proto": "TCP", "Type": "SQL_INJECTION", "Risk": "CRITICAL"},
            {"Time": "02:14:08", "Src IP": "192.168.1.180", "Target": "10.0.0.5:445", "Proto": "TCP", "Type": "COMMAND_INJECTION", "Risk": "CRITICAL"},
            {"Time": "02:14:02", "Src IP": "192.168.1.92", "Target": "10.0.0.8:8080", "Proto": "TCP", "Type": "XSS_ATTACK", "Risk": "HIGH"},
            {"Time": "02:13:55", "Src IP": "192.168.1.12", "Target": "10.0.0.1:53", "Proto": "UDP", "Type": "DNS_RECON", "Risk": "MEDIUM"}
        ])
        st.dataframe(sample_alerts, use_container_width=True, hide_index=True)