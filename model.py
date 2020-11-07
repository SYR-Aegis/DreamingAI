import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dropout, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50V2
from utils.dataset import data_generator


class Classifier:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def model(self):
        input_tensor = Input(shape=self.input_shape)

        #resnet = ResNet50V2(include_top=False, input_shape=self.input_shape, weights="imagenet")(input_tensor)
        x = Conv2D(64, (3, 3), padding="same")(input_tensor)
        x = LeakyReLU()(x)
        x = Dropout(0.4)(x)

        x = Conv2D(128, (3, 3), padding="same")(x)
        x = LeakyReLU()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.4)(x)

        x = BatchNormalization()(x)

        x = Conv2D(128, (3, 3), padding="same")(x)
        x = Conv2D(256, (3, 3), padding="same")(x)
        x = MaxPooling2D((2, 2))(x)

        x = Dropout(0.4)(x)

        x = Flatten()(x)
        # x = Dense(512, activation="relu")(x)
        # x = Dense(256, activation="relu")(x)
        x = Dense(1, activation="sigmoid")(x)

        model = Model(inputs=input_tensor, outputs=x)

        return model

    def fit(self, X, Y, batch_size, epochs=30, save=True):
        model = self.model()
        model.compile(tf.keras.optimizers.Adam(), loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(data_generator(X, Y, batch_size=batch_size), epochs=epochs, verbose=1, steps_per_epoch=Y.shape[0]//batch_size)

        if save:
            model.save("classifier.h5")
