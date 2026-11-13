import os
import tempfile
from pathlib import Path

import cv2
import streamlit as st

from src.predict import load_model, preprocess, predict_single, fuse_predictions
from src.gradcam import get_gradcam_heatmap, overlay_heatmap
from src.explain import explain


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Multi-Modal Lung AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background: #0b1220; color: #e5edf7; }
    [data-testid="stSidebar"] { background: #101b2d; }
    .hero {
        padding: 1.7rem 1.9rem;
        border: 1px solid #263b55;
        border-radius: 18px;
        background: linear-gradient(120deg, #13243a, #0e3341);
        margin-bottom: 1.2rem;
    }
    .hero h1 { color: #f2f8ff; margin: 0 0 .35rem 0; }
    .hero p { color: #a9c0d5; margin: 0; }
    div[data-testid="stMetric"] {
        background: #132238;
        border: 1px solid #29415d;
        padding: 14px;
        border-radius: 14px;
    }
    .subtle { color: #9fb3c8; font-size: .9rem; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# CACHED MODEL LOADING
# --------------------------------------------------
@st.cache_resource(show_spinner="Loading trained AI models...")
def load_models():
    xray_path = Path("models/xray_model.h5")
    ct_path = Path("models/ct_model.h5")

    if not xray_path.exists():
        raise FileNotFoundError(f"X-ray model not found: {xray_path}")
    if not ct_path.exists():
        raise FileNotFoundError(f"CT model not found: {ct_path}")

    return load_model(str(xray_path)), load_model(str(ct_path))


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "analysis" not in st.session_state:
    st.session_state.analysis = None


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 🫁 Lung AI")
    st.caption("Multi-modal imaging research interface")
    st.divider()

    st.markdown("**Models**")
    st.write("ResNet50 · Chest X-ray")
    st.write("ResNet50 · CT scan")

    st.markdown("**Explainability**")
    st.write("Grad-CAM visual maps")
    st.write("Gemini explanation (optional)")

    st.markdown("**Analysis tools**")
    st.write("Single-image multimodal analysis")
    st.write("Multi-slice CT analysis")
    st.write("Model performance comparison")

    st.divider()
    st.markdown("**Recent analyses**")
    if st.session_state.history:
        for item in reversed(st.session_state.history[-5:]):
            st.write(f"• {item['prediction']} · {item['score']:.3f}")
    else:
        st.caption("No analyses in this session yet.")


# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🫁 Multi-Modal Lung Disease Detection</h1>
    <p>AI-assisted exploration of chest X-ray and CT images with score-level fusion and visual explainability.</p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "Research demonstration only — model outputs are not a diagnosis, "
    "not calibrated clinical probabilities, and must not guide treatment. "
    "A qualified healthcare professional must interpret medical images."
)

st.markdown("### Upload imaging data")
left, right = st.columns(2)

with left:
    xray_file = st.file_uploader(
        "Chest X-ray",
        type=["jpg", "jpeg", "png"],
        key="main_xray"
    )

with right:
    ct_file = st.file_uploader(
        "Chest CT slice",
        type=["jpg", "jpeg", "png"],
        key="main_ct"
    )

if xray_file or ct_file:
    preview_cols = st.columns(2)
    if xray_file:
        with preview_cols[0]:
            st.image(xray_file.getvalue(), caption="Selected chest X-ray", use_container_width=True)
    if ct_file:
        with preview_cols[1]:
            st.image(ct_file.getvalue(), caption="Selected CT slice", use_container_width=True)

analyse = st.button(
    "🔍 Analyse selected images",
    type="primary",
    disabled=(xray_file is None and ct_file is None),
    use_container_width=True
)


# --------------------------------------------------
# ANALYSIS PIPELINE
# --------------------------------------------------
if analyse:
    temp_paths = []
    analysis_result = {
        "xray_prob": None,
        "ct_prob": None,
        "xray_original": xray_file.getvalue() if xray_file else None,
        "ct_original": ct_file.getvalue() if ct_file else None,
        "xray_cam": None,
        "ct_cam": None,
        "gradcam_warnings": [],
    }

    def save_upload(upload):
        suffix = Path(upload.name).suffix.lower() or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(upload.getvalue())
            temp_paths.append(temp.name)
            return temp.name

    try:
        with st.spinner("Running image preprocessing and model inference..."):
            xray_model, ct_model = load_models()

            if xray_file:
                xray_path = save_upload(xray_file)
                xray_array = preprocess(xray_path)
                analysis_result["xray_prob"] = float(
                    predict_single(xray_model, xray_array, "xray")
                )
                try:
                    heatmap = get_gradcam_heatmap(xray_array, xray_model)
                    overlay = overlay_heatmap(xray_path, heatmap)
                    analysis_result["xray_cam"] = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                except Exception as exc:
                    analysis_result["gradcam_warnings"].append(f"X-ray Grad-CAM unavailable: {exc}")

            if ct_file:
                ct_path = save_upload(ct_file)
                ct_array = preprocess(ct_path)
                analysis_result["ct_prob"] = float(
                    predict_single(ct_model, ct_array, "ct")
                )
                try:
                    heatmap = get_gradcam_heatmap(ct_array, ct_model)
                    overlay = overlay_heatmap(ct_path, heatmap)
                    analysis_result["ct_cam"] = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                except Exception as exc:
                    analysis_result["gradcam_warnings"].append(f"CT Grad-CAM unavailable: {exc}")

            label, score = fuse_predictions(
                analysis_result["xray_prob"],
                analysis_result["ct_prob"]
            )
            analysis_result["prediction"] = label
            analysis_result["score"] = float(score)
            analysis_result["explanation"] = explain(label, float(score))
            analysis_result["modalities"] = [
                name for name, value in [
                    ("Chest X-ray", analysis_result["xray_prob"]),
                    ("CT scan", analysis_result["ct_prob"])
                ] if value is not None
            ]

            st.session_state.analysis = analysis_result
            st.session_state.history.append({
                "prediction": label,
                "score": float(score)
            })

    except Exception as exc:
        st.error(f"Analysis could not be completed: {exc}")
        st.exception(exc)

    finally:
        for path in temp_paths:
            try:
                os.remove(path)
            except OSError:
                pass


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------
result = st.session_state.analysis

if result:
    st.divider()
    st.markdown("## Analysis overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Model classification", result["prediction"])
    col2.metric("Aggregate model score", f"{result['score'] * 100:.1f}%")
    col3.metric("Input modalities", len(result["modalities"]))

    if result["prediction"] == "Abnormal":
        st.error("The model classified the supplied image score as abnormal. This requires professional clinical review.")
    else:
        st.info("The model did not classify the supplied image score as abnormal. This does not rule out disease.")

    st.markdown("### Modality-level model outputs")
    score_cols = st.columns(2)
    if result["xray_prob"] is not None:
        score_cols[0].metric("X-ray abnormality score", f"{result['xray_prob']:.4f}")
    if result["ct_prob"] is not None:
        score_cols[1].metric("CT abnormality score", f"{result['ct_prob']:.4f}")

    if result["gradcam_warnings"]:
        for warning in result["gradcam_warnings"]:
            st.caption(warning)

    st.markdown("### Explainability")
    if result["xray_original"] is not None:
        st.markdown("**Chest X-ray**")
        c1, c2 = st.columns(2)
        c1.image(result["xray_original"], caption="Original X-ray", use_container_width=True)
        if result["xray_cam"] is not None:
            c2.image(result["xray_cam"], caption="Grad-CAM overlay", use_container_width=True)

    if result["ct_original"] is not None:
        st.markdown("**CT scan**")
        c1, c2 = st.columns(2)
        c1.image(result["ct_original"], caption="Original CT slice", use_container_width=True)
        if result["ct_cam"] is not None:
            c2.image(result["ct_cam"], caption="Grad-CAM overlay", use_container_width=True)

    st.markdown("### AI-generated explanation")
    st.write(result["explanation"])

    st.markdown("### Research screening summary")
    summary = f"""MULTI-MODAL LUNG IMAGE SCREENING SUMMARY

Classification: {result['prediction']}
Aggregate model output score: {result['score']:.4f}
Input modalities: {', '.join(result['modalities'])}

X-ray abnormality score: {result['xray_prob'] if result['xray_prob'] is not None else 'Not supplied'}
CT abnormality score: {result['ct_prob'] if result['ct_prob'] is not None else 'Not supplied'}

Explanation:
{result['explanation']}

IMPORTANT:
This output is generated by an AI research demonstration. It is not a diagnosis,
not a calibrated clinical probability, and must not be used for treatment decisions.
A qualified healthcare professional must interpret the images.
"""
    st.download_button(
        "📄 Download analysis summary",
        data=summary,
        file_name="lung_image_screening_summary.txt",
        mime="text/plain"
    )
