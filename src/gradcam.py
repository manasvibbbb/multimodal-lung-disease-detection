import cv2
import numpy as np
import tensorflow as tf


def get_gradcam_heatmap(img_array, model, last_conv_layer_name="conv5_block3_out"):
    """Generate a normalized Grad-CAM heatmap for the selected output."""
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        target_score = predictions[:, 0]

    grads = tape.gradient(target_score, conv_outputs)
    if grads is None:
        raise ValueError("Could not calculate gradients for the selected convolutional layer.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    feature_maps = conv_outputs[0]
    heatmap = tf.reduce_sum(feature_maps * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)
    heatmap = tf.math.divide_no_nan(heatmap, max_value)
    return heatmap.numpy()


def overlay_heatmap(img_path, heatmap):
    """Overlay a Grad-CAM heatmap on an image read by OpenCV."""
    image = cv2.imread(str(img_path))
    if image is None:
        raise ValueError(f"Could not read image at: {img_path}")

    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.ndim != 2:
        raise ValueError("Grad-CAM heatmap must be a two-dimensional array.")

    resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    colored = cv2.applyColorMap(
        np.uint8(np.clip(resized, 0, 1) * 255),
        cv2.COLORMAP_JET
    )
    return cv2.addWeighted(image, 0.6, colored, 0.4, 0)
