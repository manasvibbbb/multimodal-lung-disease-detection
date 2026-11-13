# ============================================================
# IMPROVED CXR / X-RAY TRAINING
# ResNet50 Transfer Learning + Fine Tuning
# ============================================================

import os
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    BatchNormalization
)

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 2. PATHS
# ============================================================

TRAIN_DIR = "data/xray/train"
VAL_DIR = "data/xray/val"

MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 3. IMAGE / TRAINING SETTINGS
# ============================================================

IMG_HEIGHT = 224
IMG_WIDTH = 224

IMG_SIZE = (
    IMG_HEIGHT,
    IMG_WIDTH
)

BATCH_SIZE = 32

STAGE1_EPOCHS = 3
STAGE2_EPOCHS = 5


# ============================================================
# 4. CHECK DATASET PATHS
# ============================================================

if not os.path.exists(TRAIN_DIR):

    raise FileNotFoundError(
        f"Training directory not found: {TRAIN_DIR}"
    )


if not os.path.exists(VAL_DIR):

    raise FileNotFoundError(
        f"Validation directory not found: {VAL_DIR}"
    )


print("\n")
print("=" * 75)
print("IMPROVED X-RAY / CXR TRAINING")
print("=" * 75)

print("\nTraining directory:")
print(TRAIN_DIR)

print("\nValidation directory:")
print(VAL_DIR)


# ============================================================
# 5. TRAINING DATA GENERATOR
# ============================================================
#
# IMPORTANT:
# ResNet50 pretrained on ImageNet expects preprocess_input.
#
# Do NOT use rescale=1./255 here.
# ============================================================

train_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    rotation_range=10,

    width_shift_range=0.05,

    height_shift_range=0.05,

    zoom_range=0.10,

    horizontal_flip=True,

    fill_mode="nearest"
)


# ============================================================
# 6. VALIDATION DATA GENERATOR
# ============================================================

val_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input
)


# ============================================================
# 7. LOAD TRAINING DATA
# ============================================================

train = train_datagen.flow_from_directory(

    TRAIN_DIR,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="binary",

    shuffle=True,

    seed=SEED
)


# ============================================================
# 8. LOAD VALIDATION DATA
# ============================================================

val = val_datagen.flow_from_directory(

    VAL_DIR,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="binary",

    shuffle=False
)


# ============================================================
# 9. DISPLAY DATASET INFORMATION
# ============================================================

print("\n")
print("=" * 75)
print("DATASET INFORMATION")
print("=" * 75)

print("\nClass mapping:")
print(train.class_indices)

print("\nTraining samples:")
print(train.samples)

print("\nValidation samples:")
print(val.samples)


# ============================================================
# 10. VERIFY BINARY CLASSIFICATION
# ============================================================

if len(train.class_indices) != 2:

    raise ValueError(
        "Expected exactly 2 classes for binary classification."
    )


# ============================================================
# 11. CALCULATE CLASS WEIGHTS
# ============================================================
#
# Instead of manually forcing:
#
# 0 -> 1.0
# 1 -> 1.5
#
# calculate weights from the actual training distribution.
# ============================================================

train_labels = train.classes

unique_classes = np.unique(
    train_labels
)

class_weights_array = compute_class_weight(

    class_weight="balanced",

    classes=unique_classes,

    y=train_labels
)


class_weights = {

    int(class_id): float(weight)

    for class_id, weight
    in zip(
        unique_classes,
        class_weights_array
    )
}


print("\n")
print("=" * 75)
print("CLASS WEIGHTS")
print("=" * 75)

print(class_weights)


# ============================================================
# 12. CREATE RESNET50 BACKBONE
# ============================================================

print("\n")
print("=" * 75)
print("LOADING RESNET50")
print("=" * 75)


base_model = ResNet50(

    weights="imagenet",

    include_top=False,

    input_shape=(
        IMG_HEIGHT,
        IMG_WIDTH,
        3
    )
)


print("\nResNet50 loaded with ImageNet weights.")


# ============================================================
# 13. STAGE 1 — FREEZE RESNET50
# ============================================================

base_model.trainable = False


# ============================================================
# 14. BUILD CLASSIFICATION HEAD
# ============================================================

inputs = Input(

    shape=(
        IMG_HEIGHT,
        IMG_WIDTH,
        3
    )
)


# Keep backbone in inference mode during stage 1.

features = base_model(

    inputs,

    training=False
)


# ------------------------------------------------------------
# Global Average Pooling
# ------------------------------------------------------------

x = GlobalAveragePooling2D()(
    features
)


# ------------------------------------------------------------
# Dense layer
# ------------------------------------------------------------

x = Dense(

    256,

    activation="relu"
)(x)


# ------------------------------------------------------------
# Batch normalization
# ------------------------------------------------------------

x = BatchNormalization()(x)


# ------------------------------------------------------------
# Dropout
# ------------------------------------------------------------

x = Dropout(
    0.5
)(x)


# ------------------------------------------------------------
# Binary output
# ------------------------------------------------------------

outputs = Dense(

    1,

    activation="sigmoid"
)(x)


# ============================================================
# 15. CREATE MODEL
# ============================================================

model = Model(

    inputs=inputs,

    outputs=outputs,

    name="CXR_ResNet50_Improved"
)


# ============================================================
# 16. STAGE 1 COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=1e-4
    ),

    loss="binary_crossentropy",

    metrics=[

        tf.keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        ),

        tf.keras.metrics.AUC(
            name="auc"
        )
    ]
)


# ============================================================
# 17. DISPLAY MODEL
# ============================================================

print("\n")
print("=" * 75)
print("MODEL ARCHITECTURE")
print("=" * 75)

model.summary()


# ============================================================
# 18. STAGE 1 CALLBACKS
# ============================================================

stage1_checkpoint = ModelCheckpoint(

    filepath=(
        "models/"
        "xray_stage1_best.keras"
    ),

    monitor="val_auc",

    mode="max",

    save_best_only=True,

    verbose=1
)


stage1_early_stop = EarlyStopping(

    monitor="val_auc",

    mode="max",

    patience=4,

    restore_best_weights=True,

    verbose=1
)


stage1_reduce_lr = ReduceLROnPlateau(

    monitor="val_auc",

    mode="max",

    factor=0.3,

    patience=2,

    min_lr=1e-7,

    verbose=1
)


# ============================================================
# 19. STAGE 1 TRAINING
# ============================================================

print("\n")
print("=" * 75)
print("STAGE 1")
print("Training classification head")
print("=" * 75)


history_stage1 = model.fit(

    train,

    validation_data=val,

    epochs=STAGE1_EPOCHS,

    class_weight=class_weights,

    callbacks=[

        stage1_checkpoint,

        stage1_early_stop,

        stage1_reduce_lr
    ],

    verbose=1
)


# ============================================================
# 20. STAGE 2 — FINE TUNING
# ============================================================

print("\n")
print("=" * 75)
print("STAGE 2")
print("Fine-tuning ResNet50")
print("=" * 75)


# Unfreeze backbone.

base_model.trainable = True


# ------------------------------------------------------------
# Freeze early layers.
# Fine-tune only the final 40 layers.
# ------------------------------------------------------------

for layer in base_model.layers:

    layer.trainable = False


for layer in base_model.layers[-40:]:

    layer.trainable = True


# ------------------------------------------------------------
# Keep BatchNormalization frozen.
# ------------------------------------------------------------

for layer in base_model.layers:

    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):

        layer.trainable = False


# ============================================================
# 21. COUNT TRAINABLE PARAMETERS
# ============================================================

trainable_count = np.sum([

    np.prod(
        variable.shape
    )

    for variable in model.trainable_variables

])


print("\nTrainable parameters:")
print(trainable_count)


# ============================================================
# 22. RECOMPILE FOR FINE TUNING
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=1e-5
    ),

    loss="binary_crossentropy",

    metrics=[

        tf.keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        ),

        tf.keras.metrics.AUC(
            name="auc"
        )
    ]
)


# ============================================================
# 23. STAGE 2 CALLBACKS
# ============================================================

stage2_checkpoint = ModelCheckpoint(

    filepath=(
        "models/"
        "xray_stage2_best.keras"
    ),

    monitor="val_auc",

    mode="max",

    save_best_only=True,

    verbose=1
)


stage2_early_stop = EarlyStopping(

    monitor="val_auc",

    mode="max",

    patience=5,

    restore_best_weights=True,

    verbose=1
)


stage2_reduce_lr = ReduceLROnPlateau(

    monitor="val_auc",

    mode="max",

    factor=0.3,

    patience=2,

    min_lr=1e-7,

    verbose=1
)


# ============================================================
# 24. STAGE 2 TRAINING
# ============================================================

history_stage2 = model.fit(

    train,

    validation_data=val,

    epochs=STAGE2_EPOCHS,

    class_weight=class_weights,

    callbacks=[

        stage2_checkpoint,

        stage2_early_stop,

        stage2_reduce_lr
    ],

    verbose=1
)


# ============================================================
# 25. SAVE FINAL MODEL
# ============================================================

FINAL_MODEL_PATH = (
    "models/"
    "xray_model_v2.keras"
)


model.save(
    FINAL_MODEL_PATH
)


# ============================================================
# 26. SAVE CLASS MAPPING
# ============================================================

mapping_path = (
    "models/"
    "xray_model_v2_class_mapping.txt"
)


with open(
    mapping_path,
    "w"
) as f:

    f.write(
        "CXR Model V2 Class Mapping\n"
    )

    f.write(
        "===========================\n"
    )

    f.write(
        str(train.class_indices)
    )

    f.write("\n")


# ============================================================
# 27. FINAL INFORMATION
# ============================================================

print("\n")
print("=" * 75)
print("TRAINING COMPLETED")
print("=" * 75)

print("\nFinal model:")
print(FINAL_MODEL_PATH)

print("\nClass mapping file:")
print(mapping_path)

print("\nStage 1 best model:")
print(
    "models/xray_stage1_best.keras"
)

print("\nStage 2 best model:")
print(
    "models/xray_stage2_best.keras"
)

print("\n")
print("=" * 75)
print("IMPORTANT")
print("=" * 75)

print(
    "The test set was NOT used during training."
)

print(
    "Evaluate the final model on the untouched "
    "data/xray/test directory."
)

print(
    "Do not tune the test threshold using the test set."
)