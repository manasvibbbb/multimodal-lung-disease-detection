print("CT Scan training started...")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
  
# -----------------------------------
# DATA GENERATORS
# -----------------------------------
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(
    rescale=1./255
)

# -----------------------------------
# LOAD DATASET
# -----------------------------------
train = train_gen.flow_from_directory(
    "data/ct/train",
    target_size=(128,128),
    batch_size=32,
    class_mode='binary'
)

val = val_gen.flow_from_directory(
    "data/ct/val",
    target_size=(128,128),
    batch_size=32,
    class_mode='binary'
)

# -----------------------------------
# DEBUG LABELS
# -----------------------------------
print("Class labels:")
print(train.class_indices)

print("Train samples:", train.samples)
print("Val samples:", val.samples)

# -----------------------------------
# RESNET MODEL
# -----------------------------------
base = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(128,128,3)
)

# Freeze most layers
for layer in base.layers[:-20]:
    layer.trainable = False

# Fine-tune top layers
for layer in base.layers[-20:]:
    layer.trainable = True

# -----------------------------------
# CUSTOM HEAD
# -----------------------------------
x = Flatten()(base.output)

x = Dense(128, activation='relu')(x)

x = Dropout(0.5)(x)

out = Dense(1, activation='sigmoid')(x)

model = Model(
    inputs=base.input,
    outputs=out
)

# -----------------------------------
# COMPILE
# -----------------------------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -----------------------------------
# EARLY STOPPING
# -----------------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# -----------------------------------
# TRAIN
# -----------------------------------
model.fit(
    train,
    validation_data=val,
    epochs=25,
    callbacks=[early_stop],
    class_weight={
        0:1.0,
        1:1.5
    }
)

# -----------------------------------
# SAVE MODEL
# -----------------------------------
model.save("models/ct_model.h5")

print("CT model saved successfully!")