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

st.info(
    "This dashboard reads the saved X-ray and CT evaluation CSVs. "
    "They represent separate modality-level test evaluations; they are not fused-model metrics."
)

st.warning(
    "The CT evaluation script currently references models/ct_model_v2.keras "
    "with 128×128 ImageNet preprocessing, while the main upload app loads "
    "models/ct_model.h5 and uses the src.predict preprocessing path (224×224 "
    "and division by 255). Therefore, the saved CT metrics should not be "
    "presented as a measured performance result for the CT model loaded by "
    "the main app. Re-evaluate the app's exact model/preprocessing pair before "
    "making that comparison."
)

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
    chart_data = {
        name: {
            metric: float(values[metric])
            for metric in comparable
            if metric in values.index
        }
        for name, values in metrics.items()
    }
    comparison = pd.DataFrame(chart_data)
    comparison = comparison.reindex([m for m in comparable if m in comparison.index])

    st.subheader("Key evaluation metrics")
    metric_cols = st.columns(len(metrics))
    for col, (name, values) in zip(metric_cols, metrics.items()):
        with col:
            st.markdown(f"**{name} evaluation**")
            for metric in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "Test Samples"]:
                if metric in values.index:
                    value = float(values[metric])
                    if metric == "Test Samples":
                        shown = f"{int(value)}"
                    elif metric == "ROC-AUC":
                        shown = f"{value:.4f}"
                    else:
                        shown = f"{value * 100:.2f}%"
                    st.metric(metric, shown)

    if not comparison.empty:
        st.subheader("Side-by-side comparison")
        st.bar_chart(comparison, y=list(comparison.columns))

        rows = []
        for name, values in metrics.items():
            row = {"Model": name}
            for metric in ["Accuracy", "Precision", "Recall", "F1-score"]:
                if metric in values.index:
                    row[metric] = f"{float(values[metric]) * 100:.2f}%"
            if "ROC-AUC" in values.index:
                row["ROC-AUC"] = f"{float(values['ROC-AUC']):.4f}"
            if "Test Samples" in values.index:
                row["Test samples"] = int(values["Test Samples"])
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

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

    st.subheader("Metrics not yet available")
    st.markdown(
        "- **Fused-model accuracy, precision, recall, F1-score and ROC-AUC:** "
        "not computed in the committed evaluation artifacts. A valid fused "
        "evaluation requires paired, patient-aligned X-ray and CT test samples "
        "with a defined fusion rule and labels.\n"
        "- **Inference time:** not recorded in the saved metric CSVs, so no "
        "latency value is displayed."
    )
    st.caption(
        "All displayed figures are read from the stored CSVs and plots. "
        "This page does not recalculate, infer, or estimate evaluation results."
    )
else:
    st.error(
        "No evaluation metric files could be loaded. Ensure the CSV files are "
        "present in evaluation_results/."
    )
