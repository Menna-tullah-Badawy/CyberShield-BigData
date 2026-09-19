"""
GenAI Executive Reports & MITRE ATT&CK Mitigation Runbooks.
شاشة عرض تقارير الحوادث الذكية ومصفوفة تكتيكات MITRE وخطة الاستجابة الفورية.
"""

import streamlit as st
import os
import glob
import configs.settings as cfg


def render():
    st.title("📝 GenAI Threat Intelligence & Incident Reports")
    st.markdown("تقارير أمنية استخباراتية فورية مولدة بالذكاء الاصطناعي ومربوطة بمصفوفة **MITRE ATT&CK**.")

    report_files = sorted(glob.glob(os.path.join(cfg.REPORTS_DIR, "*.md")), reverse=True)

    if report_files:
        selected_file = st.selectbox("Select Incident Report", report_files, format_func=lambda x: os.path.basename(x))
        with open(selected_file, "r", encoding="utf-8") as f:
            content = f.read()

        st.markdown("---")
        st.markdown(content)
    else:
        st.info("لا توجد تقارير Markdown محفوظة حالياً. يتم إنشاء التقارير تلقائياً عند وقوع التهديدات عبر `main.py` أو الـ API.")