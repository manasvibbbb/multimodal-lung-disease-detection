# Multi-Modal Lung Disease Detection

A research-oriented Streamlit application for exploring chest X-ray and CT image classification with ResNet50 models, score-level fusion, and Grad-CAM visual explanations.

> **Important:** This project is an academic/research demonstration, not a medical device. Model outputs are not diagnoses and must not be used to make treatment decisions. Clinical interpretation belongs to qualified healthcare professionals.

## Features

- Chest X-ray and CT image screening using the project's trained models.
- Score-level fusion when both modalities are supplied.
- Grad-CAM visual explanations in the main application.
- Multi-slice CT page: per-slice model scores, mean and median score summaries, and CSV export.
- Model comparison page sourced from the stored evaluation metric CSV files.
- Downloadable text report from the main application.

## Run locally

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Optional Gemini explanation: copy `.env.example` to `.env` and replace the placeholder with a newly generated API key. The app loads this local file through python-dotenv. Do not commit `.env` or expose the key.
5. Start Streamlit:

```bash
streamlit run app.py
```

Streamlit will show the main application and the pages in the `pages/` directory in its navigation.

## Repository layout

- `app.py`: primary single-image X-ray / CT workflow.
- `pages/1_Multi_Slice_CT.py`: multi-slice CT analysis.
- `pages/2_Model_Comparison.py`: saved evaluation metrics and plots.
- `src/predict.py`: model loading, preprocessing, and score-level fusion.
- `src/gradcam.py`: Grad-CAM heatmap generation.
- `src/explain.py`: optional Gemini explanation with a local fallback.
- `evaluation_results/`: stored evaluation CSVs, class mappings, and plots.
- `models/`: trained model files tracked using Git LFS.

## Evaluation metrics

The comparison page reads the committed `evaluation_results/cxr_metrics.csv` and `evaluation_results/ct_metrics.csv` files. Metrics displayed there refer to the corresponding stored test-set evaluations. They are not automatically recalculated for different model files, and they do not represent a validated fused-model result.

The committed CT evaluation script uses `models/ct_model_v2.keras` with 128×128 ImageNet preprocessing. The main upload app uses `models/ct_model.h5` and the `src.predict` preprocessing pipeline (224×224 and division by 255). The saved CT metrics therefore must not be presented as measured performance for the main app's CT model without a matching re-evaluation.

## Multi-slice CT aggregation

The multi-slice page analyses each uploaded image independently using the same CT model path and preprocessing helper as the existing app. It reports the arithmetic mean and median of the slice abnormality scores. This is a simple 2D slice aggregation utility; it does not perform 3D volume reconstruction, spatial modelling, or clinical validation.

## Model files

Large `.h5` and `.keras` files are tracked with Git LFS. Install Git LFS before cloning/downloading model artifacts.

## API key security

Use a local `.env` file for `GEMINI_API_KEY`; it is ignored by Git. If a key was ever committed or exposed, revoke/rotate it with the provider and use a new key.
