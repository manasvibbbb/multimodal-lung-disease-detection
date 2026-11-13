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
# CONFIGURATION
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

TRAIN_DIR = "data/ct/train"
VAL_DIR = "data/ct/val"

MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

STAGE1_EPOCHS = 3
STAGE2_EPOCHS = 5


# ============================================================
# CHECK DIRECTORIES
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
print("IMPROVED CT TRAINING")
print("=" * 75)


# ============================================================
# TRAINING DATA
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
# VALIDATION DATA
# ============================================================

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)


# ============================================================
# LOAD TRAINING DATA
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
# LOAD VALIDATION DATA
# ============================================================

val = val_datagen.flow_from_directory(

    VAL_DIR,

    target_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="binary",

    shuffle=False
)


# ============================================================
# DATASET INFORMATION
# ============================================================

print("\n")
print("=" * 75)
print("CT DATASET INFORMATION")
print("=" * 75)

print("\nClass mapping:")
print(train.class_indices)

print("\nTraining samples:")
print(train.samples)

print("\nValidation samples:")
print(val.samples)


# ============================================================
# CLASS WEIGHTS
# ============================================================

train_labels = train.classes

unique_classes = np.unique(train_labels)

weights = compute_class_weight(

    class_weight="balanced",

    classes=unique_classes,

    y=train_labels
)

class_weights = {

    int(class_id): float(weight)

    for class_id, weight
    in zip(unique_classes, weights)

}

print("\nClass weights:")
print(class_weights)


# ============================================================
# RESNET50
# ============================================================

base_model = ResNet50(

    weights="imagenet",

    include_top=False,

    input_shape=(128, 128, 3)
)


# ============================================================
# STAGE 1 — FREEZE BACKBONE
# ============================================================

base_model.trainable = False


inputs = Input(
    shape=(128, 128, 3)
)


features = base_model(
    inputs,
    training=False
)


# GlobalAveragePooling instead of Flatten

x = GlobalAveragePooling2D()(features)

x = Dense(
    256,
    activation="relu"
)(x)

x = BatchNormalization()(x)

x = Dropout(
    0.5
)(x)

outputs = Dense(
    1,
    activation="sigmoid"
)(x)


model = Model(
    inputs=inputs,
    outputs=outputs,
    name="CT_ResNet50_Improved"
)


# ============================================================
# STAGE 1 COMPILE
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
# STAGE 1 CALLBACKS
# ============================================================

checkpoint1 = ModelCheckpoint(

    "models/ct_stage1_best.keras",

    monitor="val_auc",

    mode="max",

    save_best_only=True,

    verbose=1
)


early_stop1 = EarlyStopping(

    monitor="val_auc",

    mode="max",

    patience=4,

    restore_best_weights=True,

    verbose=1
)


reduce_lr1 = ReduceLROnPlateau(

    monitor="val_auc",

    mode="max",

    factor=0.3,

    patience=2,

    min_lr=1e-7,

    verbose=1
)


# ============================================================
# STAGE 1 TRAINING
# ============================================================

print("\n")
print("=" * 75)
print("STAGE 1 — CT CLASSIFICATION HEAD")
print("=" * 75)


history1 = model.fit(

    train,

    validation_data=val,

    epochs=STAGE1_EPOCHS,

    class_weight=class_weights,

    callbacks=[

        checkpoint1,

        early_stop1,

        reduce_lr1
    ],

    verbose=1
)


# ============================================================
# STAGE 2 — FINE TUNING
# ============================================================

print("\n")
print("=" * 75)
print("STAGE 2 — CT RESNET50 FINE-TUNING")
print("=" * 75)


base_model.trainable = True


# Freeze all layers first

for layer in base_model.layers:

    layer.trainable = False


# Unfreeze final 40 layers

for layer in base_model.layers[-40:]:

    layer.trainable = True


# Keep BatchNorm frozen

for layer in base_model.layers:

    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):

        layer.trainable = False


# ============================================================
# RECOMPILE
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
# STAGE 2 CALLBACKS
# ============================================================

checkpoint2 = ModelCheckpoint(

    "models/ct_stage2_best.keras",

    monitor="val_auc",

    mode="max",

    save_best_only=True,

    verbose=1
)


early_stop2 = EarlyStopping(

    monitor="val_auc",

    mode="max",

    patience=5,

    restore_best_weights=True,

    verbose=1
)


reduce_lr2 = ReduceLROnPlateau(

    monitor="val_auc",

    mode="max",

    factor=0.3,

    patience=2,

    min_lr=1e-7,

    verbose=1
)


# ============================================================
# STAGE 2 TRAINING
# ============================================================

history2 = model.fit(

    train,

    validation_data=val,

    epochs=STAGE2_EPOCHS,

    class_weight=class_weights,

    callbacks=[

        checkpoint2,

        early_stop2,

        reduce_lr2
    ],

    verbose=1
)


# ============================================================
# SAVE MODEL
# ============================================================

FINAL_MODEL = "models/ct_model_v2.keras"

model.save(FINAL_MODEL)


# ============================================================
# SAVE CLASS MAPPING
# ============================================================

with open(
    "models/ct_model_v2_class_mapping.txt",
    "w"
) as f:

    f.write(
        str(train.class_indices)
    )

    f.write("\n")


# ============================================================
# DONE
# ============================================================

print("\n")
print("=" * 75)
print("CT TRAINING COMPLETE")
print("=" * 75)

print("\nFinal model:")
print(FINAL_MODEL)

print("\nThe test set was NOT used during training.")