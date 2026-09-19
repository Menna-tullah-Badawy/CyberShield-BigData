"""
CyberShield SOC Master Operations Dashboard.
واجهة المراقبة والتحكم اللحظية للأمن السيبراني وعمليات الـ SOC.
"""

import sys
from pathlib import Path

# إضافة المجلد الرئيسي للمشروع إلى sys.path لتمكين كافة الاستيرادات
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
from serving.dashboard.pages import realtime_monitor, threat_analysis, reports

# إعدادات الصفحة الرئيسية
st.set_page_config(
    page_title="CyberShield SOC Master Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# القائمة الجانبية للتنقل
st.sidebar.title("🛡️ CyberShield NIDS")
st.sidebar.caption("Enterprise AI-Powered Security Operations")

page = st.sidebar.radio(
    "Select Operational View",
    ["📡 Real-Time Monitor", "🔬 Threat Analysis", "📑 SOC Reports & KPIs"]
)

st.sidebar.markdown("---")
st.sidebar.info("🚀 System Status: **Active Monitoring**\n\n⚡ Backend: PySpark Distributed Engine")

# توجيه الصفحات
if page == "📡 Real-Time Monitor":
    realtime_monitor.render()
elif page == "🔬 Threat Analysis":
    threat_analysis.render()
elif page == "📑 SOC Reports & KPIs":
    reports.render()