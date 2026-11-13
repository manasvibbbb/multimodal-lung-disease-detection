from functools import lru_cache

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

IMG_SIZE = (224, 224)


@lru_cache(maxsize=4)
def load_model(path):
    """Load each model path once per Python process, shared by all app pages."""
    return tf.keras.models.load_model(path)


def preprocess(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)


def predict_single(model, img_array, model_type):
    prob = float(model.predict(img_array, verbose=0)[0][0])

    # X-ray: NORMAL=0, PNEUMONIA=1.
    if model_type.lower() == "xray":
        return prob

    # CT: ABNORMAL=0, NORMAL=1.
    if model_type.lower() == "ct":
        return 1 - prob

    raise ValueError("model_type must be 'xray' or 'ct'")


def fuse_predictions(xray_prob=None, ct_prob=None):
    """Average supplied abnormality scores; single modality passes through."""
    if xray_prob is not None and ct_prob is not None:
        score = (xray_prob + ct_prob) / 2
    elif xray_prob is not None:
        score = xray_prob
    elif ct_prob is not None:
        score = ct_prob
    else:
        return "No Prediction", 0.0

    label = "Abnormal" if score >= 0.5 else "Normal"
    return label, score
