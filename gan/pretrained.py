import tensorflow_hub as hub
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()
generate = hub.load("https://tfhub.dev/google/progan-128/1").signatures['default']

for _ in range(10000):
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        image = sess.run(generate(tf.random.normal([20, 512])))['default']

    image = image*255
    image = image.astype(np.uint8)

    for i in range(image.shape[0]):
        plt.imsave(f"./generated_images/generated_image{_}_{i}.jpg", image[i])