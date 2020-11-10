import argparse
import tensorflow_hub as hub
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

from gan.wgan import GAN
from SuperResolution.model.common import resolve_single
from SuperResolution.utils import load_image, plot_sample
from SuperResolution.model.srgan import sr_generator

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, default="progan", choices=["progan", "resnet", "wgan"], help="decides which GAN model to use")
parser.add_argument("num_images", type=int, default=1, help="number of images to generate")
parser.add_argument("save_path", type=str, default="./data", help="the path where to save generated images")
args = parser.parse_args()

model_name = args.model
num_images = args.num_images
path = args.save_path

if model_name == "progan":
    tf.compat.v1.disable_eager_execution()
    model = hub.load("https://tfhub.dev/google/progan-128/1").signatures['default']

    if num_images <= 20:
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            image = sess.run(model(tf.random.normal([20, 512])))['default']

            for i in range(num_images):
                plt.imsave(path+f"/generated_image_{i}.jpg", image[i])
    elif num_images > 20:
        for j in range(num_images//20):
            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                image = sess.run(model(tf.random.normal([20, 512])))['default']

                for i in range(20):
                    if i < num_images//20:
                        plt.imsave(path + f"/generated_image_{j}_{i}.jpg", image[i])
                    else:
                        break

elif model_name == "resnet":
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.enable_resource_variables()
    model = hub.Module("https://tfhub.dev/google/compare_gan/model_9_celebahq128_resnet19/1")

    if num_images <= 64:
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())

            z_values = tf.random.uniform(minval=-1, maxval=1, shape=[64, 128])
            images = model(z_values, signature="generator")
            sess.run([images])

            for i in range(num_images):
                plt.imsave(path+f"/generated_image{i}.jpg", images.eval()[i])
    elif num_images >= 64:
        for j in range(num_images//64):
            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())

                z_values = tf.random.uniform(minval=-1, maxval=1, shape=[64, 128])
                images = model(z_values, signature="generator")
                sess.run([images])

                for i in range(64):
                    if i < num_images//64:
                        plt.imsave(path + f"/generated_image{j}_{i}.jpg", images.eval()[i])
                    else:
                        break

elif model_name == "wgan":
    model = GAN()
    model.weight_save_dir = "./data/weight/wgan/"
    model.img_save_dir = "./data/"
    model.load_weight()

    for i in range(num_images):
        model.save_image(i)
