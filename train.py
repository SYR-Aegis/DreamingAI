from model import Classifier
from utils.dataset import create_dataset

real_img_path = "./data/raw_image"
fake_img_path = "./data/generated_images"

X_train, X_test, Y_train, Y_test = create_dataset(real_img_path, fake_img_path, 2000)

classifier = Classifier(input_shape=(128, 128, 3))
classifier.fit(X_train, Y_train, batch_size=32, epochs=5)
