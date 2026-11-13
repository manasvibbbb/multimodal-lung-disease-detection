import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)

# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "models/ct_model_v2.keras"
TEST_PATH = "data/ct/test"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

# Fixed primary evaluation threshold
THRESHOLD = 0.5

OUTPUT_DIR = "evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("LOADING CT MODEL")
print("=" * 70)

model = load_model(MODEL_PATH)

print("Model input shape :", model.input_shape)
print("Model output shape:", model.output_shape)


# ============================================================
# LOAD TEST DATA
# ============================================================

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print("\nTEST DATA")
print("-" * 70)
print("Number of images:", test.samples)
print("Class mapping:", test.class_indices)


# ============================================================
# CLASS MAPPING
# ============================================================

class_indices = test.class_indices

print("\nClass mapping detected:")

for class_name, index in class_indices.items():
    print(f"  {index} -> {class_name}")


# ============================================================
# FIND ABNORMAL CLASS
# ============================================================

abnormal_names = [
    "ABNORMAL",
    "Abnormal",
    "abnormal",
    "PNEUMONIA",
    "Pneumonia",
    "pneumonia"
]

abnormal_index = None

for name in abnormal_names:
    if name in class_indices:
        abnormal_index = class_indices[name]
        break

if abnormal_index is None:
    raise ValueError(
        "\nCould not find the abnormal class.\n"
        f"Detected classes: {class_indices}\n"
        "Check your CT test folder names."
    )

print("\nAbnormal class index:", abnormal_index)


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating CT predictions...")

raw_output = model.predict(
    test,
    verbose=1
).reshape(-1)

true_original = test.classes


# ============================================================
# CONVERT TO ABNORMAL PROBABILITY
# ============================================================

if abnormal_index == 1:
    abnormal_probability = raw_output
else:
    abnormal_probability = 1.0 - raw_output


# ============================================================
# TRUE LABELS
# 0 = NORMAL
# 1 = ABNORMAL
# ============================================================

true = (
    true_original == abnormal_index
).astype(int)


# ============================================================
# PREDICTIONS
# ============================================================

pred = (
    abnormal_probability >= THRESHOLD
).astype(int)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    true,
    pred
)

precision = precision_score(
    true,
    pred,
    zero_division=0
)

recall = recall_score(
    true,
    pred,
    zero_division=0
)

f1 = f1_score(
    true,
    pred,
    zero_division=0
)

auc_score = roc_auc_score(
    true,
    abnormal_probability
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true,
    pred
)

tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)

specificity = tn / (tn + fp)

npv = tn / (tn + fn)

ppv = tp / (tp + fp)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("FINAL CT TEST RESULTS")
print("=" * 70)

print(f"Test samples       : {len(true)}")
print(f"Threshold          : {THRESHOLD}")

print(f"\nAccuracy           : {accuracy * 100:.2f}%")
print(f"Precision          : {precision * 100:.2f}%")
print(f"Recall             : {recall * 100:.2f}%")
print(f"F1-score           : {f1 * 100:.2f}%")
print(f"Sensitivity        : {sensitivity * 100:.2f}%")
print(f"Specificity        : {specificity * 100:.2f}%")
print(f"PPV                : {ppv * 100:.2f}%")
print(f"NPV                : {npv * 100:.2f}%")
print(f"ROC-AUC            : {auc_score:.4f}")

print("\nConfusion Matrix")
print("----------------")
print(cm)

print("\nClassification Report")
print("---------------------")

print(
    classification_report(
        true,
        pred,
        target_names=[
            "Normal",
            "Abnormal"
        ],
        digits=4,
        zero_division=0
    )
)


# ============================================================
# SAVE INDIVIDUAL PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({

    "filename": test.filenames,

    "true_label":
        np.where(
            true == 1,
            "Abnormal",
            "Normal"
        ),

    "true_label_numeric": true,

    "predicted_label":
        np.where(
            pred == 1,
            "Abnormal",
            "Normal"
        ),

    "predicted_label_numeric": pred,

    "abnormal_probability":
        abnormal_probability,

    "raw_model_output":
        raw_output
})


prediction_file = os.path.join(
    OUTPUT_DIR,
    "ct_predictions.csv"
)

prediction_df.to_csv(
    prediction_file,
    index=False
)

print("\nPrediction file saved:")
print(prediction_file)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "Sensitivity",
        "Specificity",
        "PPV",
        "NPV",
        "ROC-AUC",
        "Threshold",
        "Test Samples",
        "True Negative",
        "False Positive",
        "False Negative",
        "True Positive"
    ],

    "Value": [
        accuracy,
        precision,
        recall,
        f1,
        sensitivity,
        specificity,
        ppv,
        npv,
        auc_score,
        THRESHOLD,
        len(true),
        tn,
        fp,
        fn,
        tp
    ]
})


metrics_file = os.path.join(
    OUTPUT_DIR,
    "ct_metrics.csv"
)

metrics_df.to_csv(
    metrics_file,
    index=False
)


# ============================================================
# SAVE CLASS MAPPING
# ============================================================

with open(
    os.path.join(
        OUTPUT_DIR,
        "ct_class_mapping.txt"
    ),
    "w"
) as f:

    f.write(
        f"Class mapping: {class_indices}\n"
    )

    f.write(
        f"Abnormal index: {abnormal_index}\n"
    )

    f.write(
        f"Threshold: {THRESHOLD}\n"
    )

    f.write(
        "Probability used for evaluation: P(Abnormal)\n"
    )


# ============================================================
# CONFUSION MATRIX IMAGE
# ============================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Normal",
        "Abnormal"
    ]
)

disp.plot()

plt.title(
    "Confusion Matrix - CT Test Set"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "ct_confusion_matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    true,
    abnormal_probability
)

plt.figure(figsize=(6, 6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {auc_score:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    "k--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve - CT Test Set"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "ct_roc_curve.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# DONE
# ============================================================

print("\n")
print("=" * 70)
print("CT EVALUATION COMPLETED")
print("=" * 70)

print("\nFiles saved:")

print("1.", prediction_file)

print("2.", metrics_file)

print(
    "3.",
    os.path.join(
        OUTPUT_DIR,
        "ct_class_mapping.txt"
    )
)

print(
    "4.",
    os.path.join(
        OUTPUT_DIR,
        "ct_confusion_matrix.png"
    )
)

print(
    "5.",
    os.path.join(
        OUTPUT_DIR,
        "ct_roc_curve.png"
    )
)