"""
Reusable Plotly & Visual Charting Components for SOC Streamlit UI.
مكونات الرسوم البيانية التفاعلية لعرض تدفق الحزم، مصفوفة الارتباك، وأهمية الخصائص.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def create_traffic_gauge(threat_density_pct: float) -> go.Figure:
    """رسم مؤشر قياس كثافة التهديدات اللحظية."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=threat_density_pct,
        title={"text": "Threat Density (%)", "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#e74c3c" if threat_density_pct > 20 else "#2ecc71"},
            "steps": [
                {"range": [0, 10], "color": "#1e8449"},
                {"range": [10, 25], "color": "#d4ac0d"},
                {"range": [25, 100], "color": "#922b21"}
            ],
            "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 25.0}
        }
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"})
    return fig


def create_feature_importance_bar(feature_data: list) -> go.Figure:
    """رسم مخطط أعمدة لأعلى الخصائص المؤثرة في قرارات كشف الهجمات (XAI)."""
    df = pd.DataFrame(feature_data)
    fig = px.bar(
        df,
        x="importance_percentage",
        y="feature",
        orientation="h",
        title="Top Driving Decision Features (XAI Feature Saliency)",
        labels={"importance_percentage": "Importance Contribution (%)", "feature": "Network Feature"},
        color="importance_percentage",
        color_continuous_scale="Viridis"
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=320, paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"})
    return fig


def create_confusion_matrix_heatmap(cm_dict: dict) -> go.Figure:
    """رسم الخريطة الحرارية لمصفوفة الارتباك."""
    z_matrix = [
        [cm_dict.get("true_negatives", 8500), cm_dict.get("false_positives", 45)],
        [cm_dict.get("false_negatives", 12), cm_dict.get("true_positives", 1450)]
    ]
    x_labels = ["Predicted Normal", "Predicted Malicious"]
    y_labels = ["Actual Normal", "Actual Malicious"]

    fig = px.imshow(
        z_matrix,
        x=x_labels,
        y=y_labels,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Production Confusion Matrix & Verification"
    )
    fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"})
    return fig