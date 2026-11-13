import numpy as np
import tensorflow as tf
import cv2

# -----------------------------------
# Generate Grad-CAM Heatmap
# -----------------------------------
def get_gradcam_heatmap(img_array, model, last_conv_layer_name="conv5_block3_out"):

    # Create gradient model
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    # Compute gradients
    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    # Average gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    # Weight feature maps
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    # Normalize
    heatmap = np.maximum(heatmap, 0)

    heatmap /= tf.math.reduce_max(heatmap)

    return heatmap.numpy()


# -----------------------------------
# Overlay Heatmap on Original Image
# -----------------------------------
def overlay_heatmap(img_path, heatmap):

    # Read original image
    img = cv2.imread(img_path)

    # Resize heatmap
    heatmap = cv2.resize(
        heatmap,
        (img.shape[1], img.shape[0])
    )

    # Convert heatmap to color
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    # Blend images
    superimposed = cv2.addWeighted(
        img,
        0.6,
        heatmap,
        0.4,
        0
    )

    return superimposed