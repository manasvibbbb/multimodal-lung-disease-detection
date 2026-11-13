import io
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.predict import load_model, preprocess, predict_single

st.set_page_config(page_title="Multi-slice CT | Lung AI", page_icon="🫁", layout="wide")

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
    .note { color: #9fb3c8; font-size: .88rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_ct_model():
    model_path = Path("models/ct_model.h5")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return load_model(str(model_path))

st.markdown("""
<div class="hero">
  <h1>🫁 Multi-slice CT Analysis</h1>
  <p>Review predictions across a selected set of CT image slices.</p>
</div>
""", unsafe_allow_html=True)

st.warning("Research demonstration only. This page evaluates uploaded 2D slices independently and averages their abnormality scores. It does not reconstruct a 3D volume or provide a clinical diagnosis.")

uploads = st.file_uploader(
    "Upload CT slices (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="Select slices from one study. Use a consistent series and ordering."
)

if uploads:
    st.caption(f"{len(uploads)} slice(s) selected.")
    preview_cols = st.columns(min(4, len(uploads)))
    for idx, upload in enumerate(uploads[:4]):
        with preview_cols[idx % len(preview_cols)]:
            st.image(upload, caption=upload.name, use_container_width=True)
    if len(uploads) > 4:
        st.caption(f"{len(uploads) - 4} additional slices will be included in the analysis.")

max_slices = st.number_input("Maximum slices to analyse", min_value=1, max_value=200, value=50, step=1)
run = st.button("Analyse CT slices", type="primary", disabled=not bool(uploads))

if run:
    selected = uploads[:int(max_slices)]
    results = []
    progress = st.progress(0, text="Loading CT model...")
    try:
        model = get_ct_model()
        for i, uploaded in enumerate(selected):
            suffix = Path(uploaded.name).suffix.lower() or ".png"
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
                    temp.write(uploaded.getvalue())
                    temp_path = temp.name
                image_array = preprocess(temp_path)
                abnormal_probability = float(predict_single(model, image_array, "ct"))
                results.append({
                    "Slice": i + 1,
                    "Filename": uploaded.name,
                    "Abnormal probability": abnormal_probability,
                    "Prediction": "Abnormal" if abnormal_probability >= 0.5 else "Normal"
                })
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)
            progress.progress((i + 1) / len(selected), text=f"Analysing slice {i + 1} of {len(selected)}")
        progress.empty()

        df = pd.DataFrame(results)
        mean_score = float(df["Abnormal probability"].mean())
        median_score = float(df["Abnormal probability"].median())
        positive_count = int((df["Abnormal probability"] >= 0.5).sum())
        overall_label = "Abnormal" if mean_score >= 0.5 else "Normal"

        st.subheader("Study summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Slices analysed", len(df))
        c2.metric("Mean abnormality score", f"{mean_score * 100:.1f}%")
        c3.metric("Median score", f"{median_score * 100:.1f}%")
        c4.metric("Slices above threshold", f"{positive_count}/{len(df)}")

        st.subheader("Slice-level results")
        chart_df = df.set_index("Slice")[["Abnormal probability"]]
        st.line_chart(chart_df, y="Abnormal probability")
        display_df = df.copy()
        display_df["Abnormal probability"] = display_df["Abnormal probability"].map(lambda value: f"{value:.4f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.subheader("Aggregate score")
        if overall_label == "Abnormal":
            st.error(f"Mean abnormality score is {mean_score:.3f} (threshold 0.50).")
        else:
            st.success(f"Mean abnormality score is {mean_score:.3f} (threshold 0.50).")

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download slice results (CSV)", data=csv,
                           file_name="ct_slice_analysis.csv", mime="text/csv")
    except Exception as exc:
        progress.empty()
        st.error(f"CT analysis could not be completed: {exc}")
        st.exception(exc)

st.markdown('<p class="note">Model scores are outputs of the current CT classifier and are not calibrated clinical probabilities. Aggregate score is the arithmetic mean of the selected slice scores.</p>', unsafe_allow_html=True)
