import streamlit as st
import tempfile

from src.predict import (
    load_model,
    preprocess,
    predict_single,
    fuse_predictions
)

from src.gradcam import (
    get_gradcam_heatmap,
    overlay_heatmap
)

from src.explain import explain


# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Multi-Modal Lung Disease Detection",
    page_icon="🫁",
    layout="wide"
)


# -----------------------------------
# SESSION HISTORY
# -----------------------------------
if "history" not in st.session_state:
    st.session_state.history = []


# -----------------------------------
# SIDEBAR
# -----------------------------------
with st.sidebar:

    st.header("System Information")

    st.write("### AI Models")
    st.write("• ResNet50 - X-Ray")
    st.write("• ResNet50 - CT Scan")

    st.write("### Technologies")
    st.write("• Transfer Learning")
    st.write("• Grad-CAM")
    st.write("• Gemini AI")

    st.write("### Supported Inputs")
    st.write("• Chest X-Ray")
    st.write("• Chest CT Scan")

    st.write("---")

    st.subheader("Recent Analyses")

    for item in st.session_state.history[-5:]:
        st.write(
            f"• {item['Prediction']} ({item['Confidence']})"
        )


# -----------------------------------
# PAGE TITLE
# -----------------------------------
st.title("🫁 Multi-Modal Lung Disease Detection")

st.info(
    "AI-Powered Analysis using Chest X-Ray, CT Scan, ResNet50, Grad-CAM and Gemini AI"
)

st.write(
    "Upload Chest X-ray and/or CT Scan images for AI-based analysis."
)


# -----------------------------------
# FILE UPLOADS
# -----------------------------------
xray_file = st.file_uploader(
    "Upload Chest X-ray",
    type=["jpg", "jpeg", "png"]
)

ct_file = st.file_uploader(
    "Upload Chest CT Scan",
    type=["jpg", "jpeg", "png"]
)


# -----------------------------------
# LOAD MODELS
# -----------------------------------
xray_model = load_model("models/xray_model.h5")
ct_model = load_model("models/ct_model.h5")

xray_prob = None
ct_prob = None

xray_temp = None
ct_temp = None


# -----------------------------------
# PROCESS X-RAY
# -----------------------------------
if xray_file:

    xray_temp = tempfile.NamedTemporaryFile(delete=False)

    xray_temp.write(xray_file.read())

    xray_img = preprocess(xray_temp.name)

    xray_prob = predict_single(
        xray_model,
        xray_img,
        "xray"
    )


# -----------------------------------
# PROCESS CT
# -----------------------------------
if ct_file:

    ct_temp = tempfile.NamedTemporaryFile(delete=False)

    ct_temp.write(ct_file.read())

    ct_img = preprocess(ct_temp.name)

    ct_prob = predict_single(
        ct_model,
        ct_img,
        "ct"
    )


# -----------------------------------
# ANALYZE BUTTON
# -----------------------------------
if st.button("🔍 Analyze Images"):

    with st.spinner("Analyzing medical images..."):

        if xray_prob is None and ct_prob is None:

            st.warning(
                "Please upload at least one image."
            )

        else:

            # -----------------------------------
            # FUSION
            # -----------------------------------
            label, score = fuse_predictions(
                xray_prob,
                ct_prob
            )

            st.session_state.history.append({
                "Prediction": label,
                "Confidence": round(score, 2)
            })

            # -----------------------------------
            # KPI DASHBOARD
            # -----------------------------------
            st.subheader("Analysis Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Prediction",
                    label
                )

            with col2:
                st.metric(
                    "Confidence",
                    f"{score*100:.1f}%"
                )

            with col3:

                if score > 0.8:
                    risk = "High"

                elif score > 0.5:
                    risk = "Medium"

                else:
                    risk = "Low"

                st.metric(
                    "Risk Level",
                    risk
                )

            # -----------------------------------
            # RISK INDICATOR
            # -----------------------------------
            if score > 0.8:

                st.error(
                    "🔴 High Probability of Abnormality"
                )

            elif score > 0.5:

                st.warning(
                    "🟠 Moderate Probability of Abnormality"
                )

            else:

                st.success(
                    "🟢 Low Probability of Abnormality"
                )

            # -----------------------------------
            # MODEL SCORES
            # -----------------------------------
            st.subheader("Individual Model Scores")

            if xray_prob is not None:
                st.write(
                    f"Chest X-Ray Score: {xray_prob:.2f}"
                )

            if ct_prob is not None:
                st.write(
                    f"CT Scan Score: {ct_prob:.2f}"
                )

            # -----------------------------------
            # X-RAY ANALYSIS
            # -----------------------------------
            if xray_prob is not None:

                st.subheader("Chest X-Ray Analysis")

                heatmap = get_gradcam_heatmap(
                    xray_img,
                    xray_model
                )

                result = overlay_heatmap(
                    xray_temp.name,
                    heatmap
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.image(
                        xray_file,
                        caption="Original X-Ray",
                        use_container_width=True
                    )

                with col2:
                    st.image(
                        result,
                        caption="Grad-CAM Heatmap",
                        use_container_width=True
                    )

            # -----------------------------------
            # CT ANALYSIS
            # -----------------------------------
            if ct_prob is not None:

                st.subheader("CT Scan Analysis")

                heatmap = get_gradcam_heatmap(
                    ct_img,
                    ct_model
                )

                result = overlay_heatmap(
                    ct_temp.name,
                    heatmap
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.image(
                        ct_file,
                        caption="Original CT Scan",
                        use_container_width=True
                    )

                with col2:
                    st.image(
                        result,
                        caption="Grad-CAM Heatmap",
                        use_container_width=True
                    )

            # -----------------------------------
            # AI EXPLANATION
            # -----------------------------------
            st.subheader("🤖 AI Medical Explanation")

            explanation = explain(
                label,
                score
            )

            st.write(explanation)

            # -----------------------------------
            # MEDICAL REPORT
            # -----------------------------------
            st.subheader("📋 Medical Report Summary")

            st.markdown(f"""
### Diagnostic Summary

**Prediction:** {label}

**Confidence Score:** {score:.2f}

**Imaging Modalities:** Chest X-Ray + CT Scan

**AI Models:** ResNet50 Transfer Learning

**Explainability:** Grad-CAM Enabled

**Analysis Type:** Multi-Modal Lung Disease Detection
""")

            # -----------------------------------
            # DOWNLOAD REPORT
            # -----------------------------------
            report = f'''
MULTI-MODAL LUNG DISEASE DETECTION REPORT

Prediction: {label}

Confidence Score: {score:.2f}

X-Ray Score: {xray_prob}

CT Score: {ct_prob}

AI Explanation:

{explanation}
'''

            st.download_button(
                label="📄 Download Medical Report",
                data=report,
                file_name="lung_disease_report.txt",
                mime="text/plain"
            )
