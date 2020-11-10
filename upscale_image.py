import matplotlib.pyplot as plt
import cv2
import glob
import argparse

from SuperResolution.model.srgan import sr_generator
from SuperResolution.model.common import resolve_single
from SuperResolution.utils import load_image

model = sr_generator()
model.load_weights("superRes.h5")

parser = argparse.ArgumentParser()
parser.add_argument("save_path", type=str, default="./data", help="the path where to save generated images")
args = parser.parse_args()

path = args.save_path

for img_name, i in zip(glob.glob(path+"/*.png"), range(len(glob.glob(path+"/*.png")))):
    img = load_image(img_name)
    result = resolve_single(model, img).numpy()
    result = cv2.resize(result, (512, 512))

    plt.imsave(img_name, result)
