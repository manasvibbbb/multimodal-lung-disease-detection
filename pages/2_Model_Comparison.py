from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Model Comparison | Lung AI", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .stApp { background: #0b1220; color: #e5edf7; }
    [data-testid="stSidebar"] { background: #101b2d; }
    .hero { padding: 1.5rem 1.7rem; border: 1px solid #263b55; border-radius: 18px;
            background: linear-gradient(120deg,#13243a,#0e3341); margin-bottom: 1.3rem; }
    .hero h1 { color: #f2f8ff; margin-bottom: .25rem; }
    .hero p { color: #a9c0d5; margin: 0; }
    div[data-testid="stMetric"] { background: #132238; border: 1px solid #29415d;
        padding: 14px; border-radius: 14px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>📊 Model Performance Comparison</h1>
  <p>Evaluation metrics loaded from the committed test-set result files.</p>
</div>
""", unsafe_allow_html=True)

st.info("Values below are read from evaluation_results/*_metrics.csv. They describe the individual X-ray and CT test evaluations; a fused-model evaluation is not currently available in these files.")

METRIC_FILES = {
    "X-ray": Path("evaluation_results/cxr_metrics.csv"),
    "CT": Path("evaluation_results/ct_metrics.csv"),
}

def load_metrics(label, path):
    if not path.exists():
        st.warning(f"{label} metrics file not found: {path}")
        return None
    frame = pd.read_csv(path)
    if not {"Metric", "Value"}.issubset(frame.columns):
        st.warning(f"{path} must contain Metric and Value columns.")
        return None
    frame["Metric"] = frame["Metric"].astype(str)
    frame["Value"] = pd.to_numeric(frame["Value"], errors="coerce")
    return frame.dropna(subset=["Value"]).set_index("Metric")["Value"]

metrics = {name: load_metrics(name, path) for name, path in METRIC_FILES.items()}
metrics = {name: values for name, values in metrics.items() if values is not None}

if metrics:
    comparable = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    chart_data = {}
    for name, values in metrics.items():
        chart_data[name] = {metric: float(values[metric]) for metric in comparable if metric in values.index}
    comparison = pd.DataFrame(chart_data)
    available = [metric for metric in comparable if metric in comparison.index]
    comparison = comparison.reindex(available)

    st.subheader("Key evaluation metrics")
    cols = st.columns(len(metrics))
    for col, (name, values) in zip(cols, metrics.items()):
        with col:
            st.markdown(f"**{name} model**")
            for metric in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "Test Samples"]:
                if metric in values.index:
                    value = values[metric]
                    shown = f"{value * 100:.2f}%" if metric != "ROC-AUC" and metric != "Test Samples" else (f"{value:.4f}" if metric == "ROC-AUC" else f"{int(value)}")
                    st.metric(metric, shown)

    if not comparison.empty:
        st.subheader("Side-by-side comparison")
        st.bar_chart(comparison, y=list(comparison.columns))
        st.dataframe((comparison * 100).round(2).rename_axis("Metric").reset_index(), use_container_width=True, hide_index=True)

    st.subheader("Evaluation plots")
    plot_cols = st.columns(2)
    plot_paths = [
        ("X-ray confusion matrix", Path("evaluation_results/cxr_confusion_matrix.png")),
        ("CT confusion matrix", Path("evaluation_results/ct_confusion_matrix.png")),
        ("X-ray ROC curve", Path("evaluation_results/cxr_roc_curve.png")),
        ("CT ROC curve", Path("evaluation_results/ct_roc_curve.png")),
    ]
    for idx, (caption, path) in enumerate(plot_paths):
        with plot_cols[idx % 2]:
            if path.exists():
                st.image(str(path), caption=caption, use_container_width=True)
            else:
                st.caption(f"{caption} not found: {path}")

    st.caption("Percentages are computed from the stored metric values for display only. No metrics are recalculated or estimated on this page.")
else:
    st.error("No evaluation metric files could be loaded. Run the evaluation scripts locally and ensure the CSV files are in evaluation_results/.")
