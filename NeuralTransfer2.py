import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import cv2
import PIL

from tensorflow.keras import Model


def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    gram_matrix = tf.expand_dims(result, axis=0)
    input_shape = tf.shape(input_tensor)
    i_j = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return gram_matrix/i_j


def LoadVgg():
    model = tf.keras.applications.vgg19.VGG19(
        include_top=True, weights=None)
    model.load_weights("./modelWeights/VGG19_Model_Weights.h5")
    model.trainable = False
    # Model layers for content and style transfer using A Neural Algorithm of Artistic Style
    content_layers = ["block4_conv2"]
    style_layers = ["block1_conv1", "block2_conv1",
                    "block3_conv1", "block4_conv1", "block5_conv1"]
    content_output = model.get_layer(content_layers[0]).output
    style_output = [model.get_layer(
        style_layer).output for style_layer in style_layers]
    gram_style_output = [gram_matrix(
        output) for output in style_output]

    StylizationModel = Model(
        [model.input], [content_output, gram_style_output])

    return StylizationModel


def optimizer():
    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    return opt


def lossFunction(style_outputs, content_outputs, style_target, content_target, style_weight, content_weight):
    content_loss = tf.reduce_mean((content_outputs - content_target)**2)
    style_loss = tf.add_n([tf.reduce_mean((output_ - target_)**2)
                          for output_, target_ in zip(style_outputs, style_target)])
    total_loss = content_weight*content_loss + style_weight*style_loss
    return total_loss


def train_step(image, epoch, style_target, content_target, style_weight, content_weight):
    with tf.GradientTape() as tape:
        output = LoadVgg(image*255)
        loss = lossFunction(
            output[1], output[0], style_target, content_target, style_weight, content_weight)
        gradient = tape.gradient(loss, image)
        optimizer.apply_gradients([(gradient, image)])
        image.assign(tf.clip_by_value(
            image, clip_value_min=0.0, clip_value_max=1.0))
    if epoch % 10 == 0:
        tf.print(f"loss -> {loss}")


def predict(content_image_path, style_image_path, EPOCHS=100, style_weight=1e-2, content_weight=1e-1):

    content_image = cv2.resize(
        cv2.imread(content_image_path), (224, 224))
    style_image = cv2.resize(
        cv2.imread(style_image_path), (224, 224))
    image = tf.image.convert_image_dtype(content_image, tf.float32)
    image = tf.Variable([image])
    content_target = LoadVgg(np.array([content_image*255]))[0]
    style_target = LoadVgg(np.array([style_image*255]))[1]
    for i in range(EPOCHS):
        train_step(image, i, style_target, content_target,
                   style_weight, content_weight)
    tensor = image*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 1:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    tensor = PIL.Image.fromarray(tensor)
    return tensor
