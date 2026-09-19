"""
Deep-Dive Forensic Threat Analysis & XAI Explanations.
شاشة التحقيق الجنائي وفحص الأدلة وتفسيرات الرموز والميزات المؤثرة في قرارات الذكاء الاصطناعي.
"""

import streamlit as st
from serving.dashboard.components.charts import create_feature_importance_bar


def render():
    st.title("🔍 Threat Investigation & Explainable AI (XAI)")
    st.markdown("تحليل عميق للأدلة الجنائية، تفكيك الحمولة، وإسناد الرموز الخبيثة المكتشفة بواسطة SecBERT والـ Decision Trees.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Selected Incident Details")
        st.json({
            "Incident_ID": "INC-2026-0881",
            "Source_IP": "192.168.1.105",
            "Destination_IP": "10.0.0.1:80",
            "Protocol": "TCP (6)",
            "Confidence_Score": 0.984,
            "Attack_Category": "SQL_INJECTION / PRIVILEGE_ESCALATION"
        })

        st.subheader("Payload Inspection & Token Attribution")
        st.code("POST /login HTTP/1.1\nUser-Agent: Moz-Scanner\nPayload: ' UNION SELECT username, password_hash FROM admin_users--", language="http")
        st.warning("⚠️ Highlighted Malicious Tokens: ['UNION SELECT', 'FROM admin_users--']")

    with col2:
        st.subheader("Model Feature Saliency (Decision Drivers)")
        sample_feats = [
            {"feature": "byte_rate_proxy", "importance_percentage": 34.0},
            {"feature": "packet_length", "importance_percentage": 28.0},
            {"feature": "header_ratio", "importance_percentage": 18.0},
            {"feature": "window_size", "importance_percentage": 12.0},
            {"feature": "protocol_idx", "importance_percentage": 8.0}
        ]
        st.plotly_chart(create_feature_importance_bar(sample_feats), use_container_width=True)