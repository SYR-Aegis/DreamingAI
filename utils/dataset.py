import numpy as np
import os
import cv2


def create_dataset(real_img_path, fake_img_path, samples, shuffle=True, train_test_ratio=0.8):
    '''
    returns a list of images and their labels
    '''
    real_imgs = []
    fake_imgs = []

    for img in os.listdir(real_img_path):
        real_imgs.append([cv2.resize(cv2.imread(os.path.join(real_img_path, img)), (128, 128)), 1])

        if len(real_imgs) == samples:
            break

    for img in os.listdir(fake_img_path):
        fake_imgs.append([cv2.imread(os.path.join(fake_img_path, img)), 0])

        if len(fake_imgs) == samples:
            break

    imgs = real_imgs+fake_imgs
    imgs = np.array(imgs)

    if shuffle:
        np.random.shuffle(imgs)

    X_train = []
    X_test = []
    Y_train = []
    Y_test = []

    for img, i in zip(imgs, range(len(imgs))):
        if i < int(train_test_ratio*len(imgs)):
            X_train.append(img[0])
            Y_train.append([img[1]])
        else:
            X_test.append(img[0])
            Y_test.append([img[1]])

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)

    return X_train, X_test, Y_train, Y_test


def data_generator(X, Y, batch_size):
    '''
    data generator for training classifier
    '''
    while True:
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i+batch_size, :, :, :]
            Y_batch = Y[i:i+batch_size, :]

            yield X_batch, Y_batch
