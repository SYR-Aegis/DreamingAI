import os
import cv2
import tensorflow as tf

from tensorflow.keras.models import load_model
from model import Classifier

classifier = load_model("classifier.h5")

for img in os.listdir("./data/generated_images"):
    img = cv2.imread(os.path.join("./data/generated_images", img))
    img = tf.expand_dims(img, axis=0)
    print(classifier.predict(img))

for img in os.listdir("./data/raw_image"):
    img = cv2.resize(cv2.imread(os.path.join("./data/raw_image", img)), (128, 128))
    img = tf.expand_dims(img, axis=0)
    print(classifier.predict(img))